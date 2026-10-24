"""P2 pruning importance: activation statistics of A2 (or an S1 student) on the three tasks.

Forward-only hooks (Minitron-style) on every decoder layer of both towers, accumulated per
task over a bucket-stratified calibration subset of the labelled train clips:

  res    mean |x| of the RMSNorm outputs (input_layernorm + post_attention_layernorm) per
         residual channel, per layer                                  (L, d)
  head   mean L2 norm of each query head's attention output (o_proj input)   (L, nQ)
  mlp    mean |act(gate) * up| per MLP channel (down_proj input)             (L, m)
  bi     Block Influence 1 - cos(h_in, h_out) per layer                      (L,)
  final  mean |x| of the final norm output                                   (d,)

Tasks: `driving` (meta_action prompt + teacher CoC, VLM tower), `det` (detection prompt +
teacher JSON, VLM tower) and `expert` (flow-matching passes at random timesteps on the GT
future over the driving sequence's cache, expert tower). VLM statistics are kept twice, over
all positions and over the generated (target) positions only. Each sequence is averaged over
its tokens first, so every sequence weighs the same.

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=1,2,3 python experiments/distill/importance_distill.py --n 128
  CUDA_VISIBLE_DEVICES=1 python experiments/distill/importance_distill.py --n 128 \
      --student /mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/s1/final --exp-id distill_importance_s1
"""

import os

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES, e.g. 1,2,3")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

import distill_cache as dc
import distill_data as dd
import teacher_lib as tl
from run_oodval import seed_all, set_determinism

REPO = Path(__file__).resolve().parents[2]
TASKS = ("driving", "det", "expert")


class ActStats:
    """Hooks that accumulate per-sequence-averaged activation statistics."""

    def __init__(self, model):
        self.task, self.target_from = None, None
        self.towers = {"vlm": model.vlm.model.language_model.layers,
                       "exp": model.expert.expert.layers}
        self.acc, self.handles = {}, []
        for tower, layers in self.towers.items():
            for i, layer in enumerate(layers):
                h = self.handles
                h.append(layer.input_layernorm.register_forward_hook(self._res(tower, i)))
                h.append(layer.post_attention_layernorm.register_forward_hook(self._res(tower, i)))
                h.append(layer.self_attn.o_proj.register_forward_pre_hook(self._head(tower, i)))
                h.append(layer.mlp.down_proj.register_forward_pre_hook(self._mlp(tower, i)))
                h.append(layer.register_forward_hook(self._bi(tower, i), with_kwargs=True))
        self.handles.append(model.vlm.model.language_model.norm.register_forward_hook(
            self._final("vlm")))
        self.handles.append(model.expert.expert.norm.register_forward_hook(self._final("exp")))
        self.head_dim = 128

    def _active(self, tower):
        return self.task is not None and (tower == "exp") == (self.task == "expert")

    def _add(self, tower, key, i, x):
        """x: (L_tok, ...) per-token values of one sequence -> += token mean (and target mean)."""
        for scope, sl in (("all", slice(None)), ("tgt", slice(self.target_from, None))):
            if scope == "tgt" and (tower == "exp" or self.target_from is None):
                continue
            k = (self.task, tower, scope, key)
            v = x[sl].float().mean(0).cpu()
            store = self.acc.setdefault(k, {})
            store[i] = store[i] + v if i in store else v

    def _res(self, tower, i):
        def hook(module, args, out):
            if self._active(tower):
                self._add(tower, "res", i, out[0].abs())  # (L_tok, d)
        return hook

    def _head(self, tower, i):
        def hook(module, args):
            if self._active(tower):
                x = args[0][0]  # (L_tok, nQ * 128)
                self._add(tower, "head", i, x.view(x.shape[0], -1, self.head_dim).float().norm(dim=-1))
        return hook

    def _mlp(self, tower, i):
        def hook(module, args):
            if self._active(tower):
                self._add(tower, "mlp", i, args[0][0].abs())  # (L_tok, m)
        return hook

    def _bi(self, tower, i):
        def hook(module, args, kwargs, out):
            if self._active(tower):
                h_out = (out[0] if isinstance(out, tuple) else out)[0]
                # accelerate moves inputs inside the wrapped forward, so the hook still sees
                # the hidden state on the previous layer's card
                h_in = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])[0]
                h_in = h_in.to(h_out.device)
                cos = torch.nn.functional.cosine_similarity(h_in.float(), h_out.float(), dim=-1)
                self._add(tower, "bi", i, (1 - cos)[:, None])  # (L_tok, 1)
        return hook

    def _final(self, tower):
        def hook(module, args, out):
            if self._active(tower):
                self._add(tower, "final", 0, out[0].abs())
        return hook

    def arrays(self, n_seq):
        """-> {"task/tower/scope/key": (n_layers, ...) mean over sequences}."""
        out = {}
        for (task, tower, scope, key), store in self.acc.items():
            arr = torch.stack([store[i] for i in sorted(store)]).numpy() / n_seq[task]
            out[f"{task}/{tower}/{scope}/{key}"] = arr.squeeze(-1) if key == "bi" else arr
        return out


def calib_items(n, seed):
    """Bucket-stratified clips of train_10k that have both teacher labels."""
    man = pd.read_parquet(REPO / "outputs" / "distill_sets" / "train_10k.parquet")
    have = set(dd.labelled_items("train_10k"))
    man = man[[(c, int(t)) in have for c, t in zip(man.clip_id, man.t0_us)]]
    rng = np.random.default_rng(seed)
    per = int(np.ceil(n / man.bucket.nunique()))
    picks = []
    for _, g in man.groupby("bucket"):
        idx = rng.permutation(len(g))[:per]
        picks += [(r.clip_id, int(r.t0_us)) for r in g.iloc[idx].itertuples()]
    rng.shuffle(picks)
    return picks[:n]


def load_model(student):
    if student is None:
        model, _, per_gpu = tl.load_teacher()
        return model, None, f"teacher, layers per card {per_gpu}"
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    model = Alpamayo2Super.from_pretrained(student, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    vmap_p = Path(student) / "vocab_map.npy"
    return model, (np.load(vmap_p) if vmap_p.exists() else None), f"student {student}"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n-expert", type=int, default=4, help="flow-matching passes per clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student", default=None, help="student checkpoint dir (default: teacher)")
    ap.add_argument("--exp-id", default="distill_importance_teacher")
    args = ap.parse_args()
    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    set_determinism()
    items = calib_items(args.n, args.seed)
    model, vmap, desc = load_model(args.student)
    builder = dd.SeqBuilder(vocab_map=vmap)
    stats = ActStats(model)
    n_seq = dict.fromkeys(TASKS, 0)
    losses = {"driving_ce": [], "det_ce": [], "fm": []}
    t0 = time.time()
    print(f"{desc} | {len(items)} calibration clips", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for j, (clip_id, t0_us) in enumerate(items):
            data = dc.load_cached(dc.path_for(clip_id, t0_us, "train_10k"))
            for task in ("driving", "det"):
                lab = dd.load_label("train_10k", task, clip_id, t0_us)
                seq = (builder.driving(data, lab["gen_ids"]) if task == "driving"
                       else builder.det(data, lab["gen_ids"]))
                ids = seq["input_ids"].to("cuda:0")  # (1, L)
                stats.task, stats.target_from = task, seq["prompt_len"]
                out = model.vlm(input_ids=ids, attention_mask=seq["attention_mask"].to("cuda:0"),
                                use_cache=(task == "driving"),
                                **{k: seq[k].to("cuda:0") for k in dd.VISION_KEYS})
                n_seq[task] += 1
                logits = out.logits[0, seq["prompt_len"] - 1:-1].float()  # (n, V)
                losses[f"{task}_ce"].append(float(torch.nn.functional.cross_entropy(
                    logits, ids[0, seq["prompt_len"]:].to(logits.device))))
                if task == "driving":
                    stats.task, stats.target_from = "expert", None
                    traj = {k: v.to("cuda:0") for k, v in seq["traj"].items()}
                    for k in range(args.n_expert):
                        seed_all(args.seed * 1000 + j * 10 + k)
                        losses["fm"].append(float(dd.expert_fm(model, traj, out)))
                        n_seq["expert"] += 1
                stats.task = None
                del out
            if (j + 1) % 16 == 0:
                print(f"[{j + 1}/{len(items)}] {(time.time() - t0) / (j + 1):.1f} s/clip  CE drv "
                      f"{np.mean(losses['driving_ce']):.3f} det {np.mean(losses['det_ce']):.3f} "
                      f"FM {np.mean(losses['fm']):.4f}", flush=True)

    arrays = stats.arrays(n_seq)
    np.savez(out_dir / "stats.npz", **arrays)
    (out_dir / "calib_items.json").write_text(json.dumps(items))
    metrics = {k: float(np.mean(v)) for k, v in losses.items()}
    metrics.update({"n_seq": n_seq, "sec_per_clip": (time.time() - t0) / len(items)})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "model": desc, "n_clips": len(items), "n_expert": args.n_expert, "seed": args.seed,
        "calib": "bucket-stratified train_10k clips with driving + det teacher labels",
        "statistics": ["res", "head", "mlp", "bi", "final"], "scopes": ["all", "tgt"],
        "gpus": os.environ["CUDA_VISIBLE_DEVICES"]}, indent=2))
    lines = [f"importance stats ({desc}), {len(items)} clips",
             (f"teacher-forced CE on own labels: driving {metrics['driving_ce']:.4f}, det "
             f"{metrics['det_ce']:.4f}; expert FM loss {metrics['fm']:.4f}"),
             "arrays: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in sorted(arrays.items()))]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
