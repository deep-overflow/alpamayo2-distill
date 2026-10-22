"""Teacher-forced sequences for importance passes, student training and student evaluation.

Every sequence is built in the teacher's token space with the teacher's own tokenizer,
processor and trajectory tokenizers (from its config, no weights), exactly as
teacher_lib builds prompts, and only then remapped to a pruned student's vocabulary
(`vocab_map.npy`: old id -> new id, -1 = dropped):

  driving  meta_action prompt (+ fused history-trajectory tokens) + teacher CoC/meta-action
           ids ending with <|traj_future_start|>; GT future for the expert's flow loss
  det      vqa-layout prompt with the detection question + teacher JSON ids ending <|im_end|>

`DistillDataset` pairs the cached frames with the P1 teacher labels and returns both
sequences of one clip plus the teacher's top-64 distribution (remapped, renormalised over
the ids the student keeps) at every target position. With `det_source="gt"` the det target is
the projected obstacle.offline answer (gt_det_labels.py) and carries no KD targets.
"""

import json
from pathlib import Path

import distill_cache as dc
import hydra.utils as hyu
import numpy as np
import slice_lib as pl
import torch
from det_utils import PROMPTS

from alpamayo2_super.config import Alpamayo2SuperConfig, build_alpamayo2_super_tokenizer
from alpamayo2_super.helper import get_processor
from alpamayo2_super.models.utils import fuse_traj_tokens
from alpamayo2_super.text_tasks import build_text_task_messages

REPO = Path(__file__).resolve().parents[2]
TEACHER_LABELS = REPO / "outputs" / "distill_teacher"
GT_DET_LABELS = REPO / "outputs" / "distill_gtdet"
DET_QUESTION = PROMPTS["p2_release_style"]
VISION_KEYS = ("pixel_values", "image_grid_thw")


class SeqBuilder:
    """Prompt/target sequences in the teacher's id space, optionally remapped for a student."""

    def __init__(self, vocab_map=None):
        self.cfg = Alpamayo2SuperConfig.from_pretrained(pl.TEACHER_SNAPSHOT)
        self.cfg._name_or_path = pl.TEACHER_SNAPSHOT  # get_processor resolves files from it
        self.tokenizer = build_alpamayo2_super_tokenizer(
            pl.TEACHER_SNAPSHOT, self.cfg.history_vocab_size, self.cfg.future_vocab_size)
        self.processor = get_processor(self.tokenizer, self.cfg)
        self.hist_tok = hyu.instantiate(self.cfg.hist_traj_tokenizer_cfg, load_weights=False)
        self.fut_tok = hyu.instantiate(self.cfg.future_traj_tokenizer_cfg, load_weights=False)
        self.vmap = None if vocab_map is None else torch.as_tensor(vocab_map, dtype=torch.long)

    def _tokenize(self, data, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, add_vision_id=False)
        images = data["image_frames"].flatten(0, 1).float() / 255.0  # (24, 3, 320, 576)
        return dict(self.processor(text=text, images=images, videos=None, padding=False,
                                   return_tensors="pt", do_rescale=False))

    def remap(self, ids):
        """Teacher ids -> student ids (identity without a vocab map)."""
        if self.vmap is None:
            return ids
        new = self.vmap[ids]
        assert (new >= 0).all(), "id outside the student vocabulary"
        return new

    def _insert_lidar(self, tok):
        """Teacher-space ids with the 258-token LiDAR block after the last camera image."""
        import lidar_lib as ll

        vc = self.cfg.vlm_config
        tok["input_ids"], _ = ll.insert_lidar_tokens(tok["input_ids"], vc.vision_start_token_id,
                                                     vc.vision_end_token_id, vc.video_token_id)
        tok["attention_mask"] = torch.ones_like(tok["input_ids"])
        return tok

    def _finish(self, tok, target_ids):
        prompt_len = tok["input_ids"].shape[1]
        ids = tok["input_ids"]
        if target_ids is not None:
            tgt = torch.as_tensor(np.asarray(target_ids), dtype=torch.long).view(1, -1)
            ids = torch.cat([ids, tgt], dim=1)  # (1, P + n)
        return {"input_ids": self.remap(ids), "attention_mask": torch.ones_like(ids),
                **{k: tok[k] for k in VISION_KEYS}, "prompt_len": prompt_len,
                "n_target": 0 if target_ids is None else len(target_ids)}

    def driving(self, data, target_ids=None, lidar=False):
        tok = self._tokenize(data, build_text_task_messages(data, self.cfg, "meta_action"))
        traj = {"ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]}
        tok["input_ids"] = fuse_traj_tokens(self.hist_tok, self.fut_tok, tok["input_ids"], traj,
                                            self.cfg.traj_ids)
        if lidar:
            tok = self._insert_lidar(tok)
        seq = self._finish(tok, target_ids)
        seq["traj"] = {**traj, "ego_future_xyz": data["ego_future_xyz"],
                       "ego_future_rot": data["ego_future_rot"]}
        return seq

    def det(self, data, target_ids=None, question=DET_QUESTION, lidar=False):
        d = {**data, "question": question}
        tok = self._tokenize(d, build_text_task_messages(d, self.cfg, "vqa"))
        if lidar:
            tok = self._insert_lidar(tok)
        return self._finish(tok, target_ids)

    def kd_targets(self, label):
        """Teacher top-64 at each target step -> (ids (n, k) student space, probs (n, k)).

        Probabilities are the teacher's full-vocabulary ones, restricted to the top-64 ids the
        student keeps and renormalised; dropped ids get probability 0 (id set to 0).
        """
        ids = torch.as_tensor(label["topk_ids"], dtype=torch.long)  # (n, k)
        p = torch.exp(torch.as_tensor(label["topk_logits"], dtype=torch.float32)
                      - torch.as_tensor(label["lse"])[:, None])  # (n, k)
        if self.vmap is not None:
            ids = self.vmap[ids]
            p = torch.where(ids >= 0, p, torch.zeros_like(p))
            ids = ids.clamp(min=0)
        return ids, p / p.sum(-1, keepdim=True).clamp(min=1e-12)


def expert_fm(model, traj, vlm_out):
    """`ExpertModel.forward` (one flow-matching draw on the GT future) that also works on a
    model split over cards: the prediction is moved back to the action's card before the
    loss, and the expert's tokens are cropped out of the cache again. -> loss (scalar)."""
    ex = model.expert
    cache = vlm_out.past_key_values
    cache_len = cache.get_seq_length()
    fut = ex._process_traj_future_training(traj)
    emb = ex.action_in_proj(fut["noisy_x"], fut["timesteps"])  # (B, n_tok, d_e)
    pos = ex._process_mrope_position_ids(vlm_out, vlm_out.logits.shape[0], emb.shape[1],
                                         emb.device)
    kw = {"is_causal": False} if ex.config.expert_non_causal_attention else {}
    eo = ex.expert(inputs_embeds=emb, position_ids=pos, past_key_values=cache, use_cache=True,
                   **kw)
    cache.crop(cache_len)
    pred = ex.action_out_proj(eo.last_hidden_state).to(emb.device)
    pred = pred.view(-1, *ex.action_space.get_action_space_dims())
    return ex.diffusion.compute_loss_from_pred(fut, pred.float())


def load_label(set_name, task, clip_id, t0_us):
    z = np.load(TEACHER_LABELS / set_name / task / f"{clip_id}__t0_{int(t0_us)}.npz",
                allow_pickle=True)
    out = {k: z[k] for k in ("gen_ids", "topk_ids", "topk_logits", "lse")}
    out["text"] = json.loads(str(z["text"]))
    if "traj" in z.files:
        out["traj"] = z["traj"]
    return out


def label_path(set_name, task, clip_id, t0_us, det_source="a2"):
    root = GT_DET_LABELS if task == "det" and det_source == "gt" else TEACHER_LABELS
    return root / set_name / task / f"{clip_id}__t0_{int(t0_us)}.npz"


class DistillDataset(torch.utils.data.Dataset):
    """One item = one clip: its driving and detection sequences with KD targets."""

    def __init__(self, set_name, items, builder, lidar=False, det_source="a2"):
        self.set_name, self.items, self.b, self.lidar = set_name, list(items), builder, lidar
        self.det_source = det_source

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip_id, t0_us = self.items[i]
        data = dc.load_cached(dc.path_for(clip_id, t0_us, self.set_name))
        out = {"clip_id": clip_id, "t0_us": t0_us}
        if self.lidar:
            import build_lidar_distill as bl

            out["lidar"] = bl.load_cached(bl.path_for(clip_id, t0_us, self.set_name))
        for task in ("driving", "det"):
            if task == "det" and self.det_source == "gt":
                ids = np.load(label_path(self.set_name, task, clip_id, t0_us, "gt"))["gen_ids"]
                out[task] = self.b.det(data, ids, lidar=self.lidar)
                continue
            lab = load_label(self.set_name, task, clip_id, t0_us)
            seq = (self.b.driving(data, lab["gen_ids"], lidar=self.lidar) if task == "driving"
                   else self.b.det(data, lab["gen_ids"], lidar=self.lidar))
            seq["kd_ids"], seq["kd_p"] = self.b.kd_targets(lab)
            out[task] = seq
        return out


def labelled_items(set_name, tasks=("driving", "det"), lidar=False, det_source="a2"):
    """(clip_id, t0_us) of the set's manifest that have a cache (and a LiDAR cache when asked)
    and every requested label."""
    import pandas as pd

    man = pd.read_parquet(REPO / "outputs" / "distill_sets" / f"{set_name}.parquet")
    keep = []
    for r in man.itertuples():
        if not dc.path_for(r.clip_id, r.t0_us, set_name).exists():
            continue
        if lidar:
            import build_lidar_distill as bl

            if not bl.path_for(r.clip_id, r.t0_us, set_name).exists():
                continue
        if all(label_path(set_name, t, r.clip_id, r.t0_us, det_source).exists() for t in tasks):
            keep.append((r.clip_id, int(r.t0_us)))
    return keep
