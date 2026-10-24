"""Alpamayo 2 Super as the teacher: prompts, generation with top-k logits, teacher-forced
forwards and seeded denoising, shared by the G0 gates and the labeling run.

Driving sequences use the release `meta_action` task (CoC + meta-action, stopped right
after `<|traj_future_start|>`, sampling as `text_tasks.generate_text`: temperature 0.6,
top_p 0.98). Detection sequences use the `vqa` message layout the release grounding
notebook uses, on the same six-camera driving profile. Placement and seeding are those of
oodval/run_oodval.py (layer i of both towers on the same card).
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oodval"))

from run_oodval import MODEL_ID, MODEL_REV, layer_device_map, seed_all

from alpamayo2_super import helper
from alpamayo2_super.models.alpamayo2_super import (
    Alpamayo2Super,
    MaskDiscreteTrajectoryLogitsProcessor,
)
from alpamayo2_super.models.expert_utils import (
    StopAfterEOS,
    build_expert_pos_ids_and_attn_mask,
    find_eos_offset,
    replace_padding_after_eos,
)
from alpamayo2_super.models.token_utils import extract_text_tokens
from alpamayo2_super.models.utils import fuse_traj_tokens
from alpamayo2_super.text_tasks import build_text_task_messages

TOPK = 64
VISION_KEYS = ("pixel_values", "image_grid_thw")


def bf16(fn):
    """Run under inference mode + bf16 autocast, as every earlier A2 run did (run_oodval)."""
    def wrapped(*a, **kw):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return fn(*a, **kw)
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped


def load_teacher(first_share=0.47):
    """A2 in bf16 over the visible cards (2 cards: `first_share` of the layers on card 0)."""
    n_gpus = torch.cuda.device_count()
    share = first_share if n_gpus == 2 else 1.0 / n_gpus - 0.03
    device_map, per_gpu = layer_device_map(n_gpus, share)
    model = Alpamayo2Super.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16,
                                           device_map=device_map)
    model.eval()
    processor = helper.get_processor(model.tokenizer, model.config)
    return model, processor, per_gpu


def _tokenize(model, processor, data, messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         add_vision_id=False)
    images = data["image_frames"].flatten(0, 1).float() / 255.0  # (24, 3, H, W)
    return dict(processor(text=text, images=images, videos=None, padding=False,
                          return_tensors="pt", do_rescale=False))


def driving_inputs(model, processor, data):
    """meta_action prompt with the history trajectory tokens fused in. Tensors on cuda:0."""
    tok = _tokenize(model, processor, data,
                    build_text_task_messages(data, model.config, "meta_action"))
    traj = {"ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]}
    tok["input_ids"] = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer,
                                        tok["input_ids"], traj, model.config.traj_ids)
    return helper.to_device({"tok": tok, **traj}, "cuda:0")


def det_inputs(model, processor, data, question):
    """vqa-layout prompt (images + question), as the release grounding notebook builds it."""
    tok = _tokenize(model, processor, {**data, "question": question},
                    build_text_task_messages({**data, "question": question}, model.config, "vqa"))
    return helper.to_device({"tok": tok}, "cuda:0")


def _gen_config(model, max_new, temperature, top_p):
    gc = copy.deepcopy(model.vlm.generation_config)
    gc.do_sample = True
    gc.temperature = temperature
    gc.top_p = top_p
    gc.top_k = None
    gc.num_return_sequences = 1
    gc.max_new_tokens = max_new
    gc.output_logits = True
    gc.return_dict_in_generate = True
    gc.pad_token_id = model.tokenizer.pad_token_id
    return gc


def _stop_ids(model, driving):
    if driving:
        return {model.config.traj_ids["future_start"]}
    return set(np.atleast_1d(model.vlm.generation_config.eos_token_id).tolist())


def _extract(model, gen_row, logits_rows, stop_ids, topk):
    """One generated row -> label fields, cut after the first stop token."""
    gen = gen_row.tolist()
    n = len(gen)
    for i, t_id in enumerate(gen):  # keep up to and including the first stop token
        if t_id in stop_ids:
            n = i + 1
            break
    gen = gen[:n]
    logits = logits_rows[:n].float()  # (n, V) raw, pre-processor
    lse = torch.logsumexp(logits, dim=-1)  # (n,)
    vals, ids = logits.topk(topk, dim=-1)  # (n, k)
    text = extract_text_tokens(model.tokenizer, torch.tensor([gen]))
    return {
        "gen_ids": np.asarray(gen, dtype=np.int32),
        "hit_stop": bool(gen) and gen[-1] in stop_ids,
        "cot": text["cot"][0], "meta_action": text["meta_action"][0], "answer": text["answer"][0],
        "box": text["box"][0], "raw": model.tokenizer.decode(gen, skip_special_tokens=False),
        "topk_ids": ids.cpu().numpy().astype(np.int32),
        "topk_logits": vals.cpu().numpy().astype(np.float16),
        "lse": lse.cpu().numpy().astype(np.float32),
    }


def _generate_kwargs(model, driving, max_new, temperature, top_p):
    tids = model.config.traj_ids
    lp = LogitsProcessorList([MaskDiscreteTrajectoryLogitsProcessor(
        traj_token_offset=min(tids["history_id0"], tids["future_id0"]),
        traj_vocab_size=model.config.traj_vocab_size)])
    kw = {"generation_config": _gen_config(model, max_new, temperature, top_p),
          "logits_processor": lp}
    if driving:
        kw["stopping_criteria"] = StoppingCriteriaList([StopAfterEOS(eos_token_id=tids["future_start"])])
    return kw


@bf16
def generate(model, inputs, seed, driving, max_new=512, temperature=0.6, top_p=0.98,
             topk=TOPK):
    """One sampled continuation plus the teacher's raw next-token distribution per step.

    Returns a dict with gen_ids (n,), text fields, topk_ids (n, k) int32, topk_logits
    (n, k) float16, lse (n,) float32 (log-sum-exp of the full raw logits, so
    p = exp(logit - lse) exactly), and `out` (the HF output, its cache included).
    """
    tok = inputs["tok"]
    prompt_len = tok["input_ids"].shape[1]
    t = time.time()
    seed_all(seed)
    out = model.vlm.generate(**tok, **_generate_kwargs(model, driving, max_new, temperature, top_p))
    torch.cuda.synchronize()
    sec = time.time() - t
    seq = out.sequences  # (1, prompt_len + n_raw)
    if driving:
        seq = replace_padding_after_eos(token_ids=seq, eos_token_id=model.config.traj_ids["future_start"],
                                        pad_token_id=model.tokenizer.pad_token_id)
    logits = torch.stack([lg[0] for lg in out.logits])  # (n_raw, V)
    res = _extract(model, seq[0, prompt_len:], logits, _stop_ids(model, driving), topk)
    return {**res, "prompt_len": prompt_len, "sec": sec, "out": out}


@bf16
def generate_batch(model, inputs_list, seed, driving, max_new=512, temperature=0.6, top_p=0.98,
                   topk=TOPK):
    """`generate` for several samples at once (no cache returned, so no denoising).

    The prompts of one task have identical token layouts (same template, same image grid),
    so rows are stacked without padding; this is asserted. Sampling shares one RNG stream,
    seeded once per batch. -> list of label dicts (each with `sec` = batch time / B).
    """
    toks = [x["tok"] for x in inputs_list]
    lens = {t["input_ids"].shape[1] for t in toks}
    assert len(lens) == 1, f"prompt lengths differ: {sorted(lens)}"
    prompt_len = lens.pop()
    batch = {"input_ids": torch.cat([t["input_ids"] for t in toks]),  # (B, L)
             "attention_mask": torch.cat([t["attention_mask"] for t in toks]),  # (B, L)
             "pixel_values": torch.cat([t["pixel_values"] for t in toks]),  # (B*N_patch, D)
             "image_grid_thw": torch.cat([t["image_grid_thw"] for t in toks])}  # (B*24, 3)
    t = time.time()
    seed_all(seed)
    out = model.vlm.generate(**batch, **_generate_kwargs(model, driving, max_new, temperature,
                                                          top_p))
    torch.cuda.synchronize()
    sec = (time.time() - t) / len(toks)
    seq = out.sequences  # (B, prompt_len + n_raw)
    if driving:
        seq = replace_padding_after_eos(token_ids=seq, eos_token_id=model.config.traj_ids["future_start"],
                                        pad_token_id=model.tokenizer.pad_token_id)
    logits = torch.stack(out.logits, dim=1)  # (B, n_raw, V)
    stop = _stop_ids(model, driving)
    return [{**_extract(model, seq[b, prompt_len:], logits[b], stop, topk),
             "prompt_len": prompt_len, "sec": sec} for b in range(len(toks))]


@bf16
def forced_forward(model, inputs, gen_ids):
    """Teacher-forced pass over prompt + gen_ids. -> (log-probs (n, V) for each gen token's
    prediction step, KV cache, rope_deltas, total length)."""
    tok = inputs["tok"]
    gen = torch.as_tensor(gen_ids, device=tok["input_ids"].device).view(1, -1)  # (1, n)
    ids = torch.cat([tok["input_ids"], gen], dim=1)  # (1, L)
    out = model.vlm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                    **{k: tok[k] for k in VISION_KEYS})
    p = tok["input_ids"].shape[1]
    logp = torch.log_softmax(out.logits[0, p - 1:-1].float(), dim=-1)  # (n, V)
    return logp, out.past_key_values, model.vlm.model.rope_deltas, ids.shape[1]


@bf16
def denoise(model, cache, rope_deltas, seq_ids, kv_len, inputs, seeds, diffusion_kwargs=None):
    """K seeded flow-matching samples on one cache; the body of run_oodval's second half.
    seq_ids (1, L) is prompt + generated ids. -> (K, 64, 3) xyz in the t0 ego frame."""
    device = seq_ids.device
    n_diff = model.expert.action_space.get_action_space_dims()[0]
    offset = find_eos_offset(sequences=seq_ids, eos_token_id=model.config.traj_ids["future_start"],
                             device=device)  # (1,)
    position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=kv_len, n_diffusion_tokens=n_diff,
        b_star=1, device=device)
    fkw = {"is_causal": False} if model.expert.config.expert_non_causal_attention else {}

    def step_fn(x, t):
        emb = model.expert.action_in_proj(x, t)
        if emb.dim() == 2:
            emb = emb.view(1, n_diff, -1)
        eo = model.expert.expert(inputs_embeds=emb, position_ids=position_ids,
                                 past_key_values=cache, attention_mask=attention_mask,
                                 use_cache=True, **fkw)
        cache.crop(kv_len)
        pred = model.expert.action_out_proj(eo.last_hidden_state)
        return pred.view(-1, *model.expert.action_space.get_action_space_dims()).to(x.device)

    hist_xyz = inputs["ego_history_xyz"][:, -1]  # (1, 16, 3)
    hist_rot = inputs["ego_history_rot"][:, -1]  # (1, 16, 3, 3)
    preds = []
    for s in seeds:
        seed_all(s)
        action = model.expert.diffusion.sample(batch_size=1, step_fn=step_fn, device=device,
                                               return_all_steps=False, **(diffusion_kwargs or {}))
        xyz, _ = model.expert.action_space.action_to_traj(action, hist_xyz, hist_rot)
        preds.append(xyz[0].float().cpu().numpy())  # (64, 3)
    return np.stack(preds)


@bf16
def denoise_batched(model, cache, rope_deltas, seq_ids, kv_len, inputs, k, seed,
                    diffusion_kwargs=None):
    """`denoise` with the K samples in one batch (one RNG seed), as the release
    `sample_trajectories_from_data` does: the cache is repeated K times in place.
    -> (K, 64, 3) xyz in the t0 ego frame."""
    device = seq_ids.device
    n_diff = model.expert.action_space.get_action_space_dims()[0]
    offset = find_eos_offset(sequences=seq_ids, eos_token_id=model.config.traj_ids["future_start"],
                             device=device).repeat(k)  # (K,)
    cache.batch_repeat_interleave(k)
    position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas.repeat(k, 1), kv_cache_seq_len=kv_len,
        n_diffusion_tokens=n_diff, b_star=k, device=device)
    fkw = {"is_causal": False} if model.expert.config.expert_non_causal_attention else {}

    def step_fn(x, t):
        emb = model.expert.action_in_proj(x, t)
        if emb.dim() == 2:
            emb = emb.view(k, n_diff, -1)
        eo = model.expert.expert(inputs_embeds=emb, position_ids=position_ids,
                                 past_key_values=cache, attention_mask=attention_mask,
                                 use_cache=True, **fkw)
        cache.crop(kv_len)
        pred = model.expert.action_out_proj(eo.last_hidden_state)
        return pred.view(-1, *model.expert.action_space.get_action_space_dims()).to(x.device)

    hist_xyz = inputs["ego_history_xyz"][:, -1].repeat(k, 1, 1)  # (K, 16, 3)
    hist_rot = inputs["ego_history_rot"][:, -1].repeat(k, 1, 1, 1)  # (K, 16, 3, 3)
    seed_all(seed)
    action = model.expert.diffusion.sample(batch_size=k, step_fn=step_fn, device=device,
                                           return_all_steps=False, **(diffusion_kwargs or {}))
    xyz, _ = model.expert.action_space.action_to_traj(action, hist_xyz, hist_rot)
    return xyz.float().cpu().numpy()  # (K, 64, 3)

