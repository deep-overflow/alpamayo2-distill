"""Open-loop evaluation of Alpamayo 2 Super on the OOD-val 262 clips.

Matched to `alpamayo-model-compression/experiments/evaluation/run_baseline.py`, the
runner the Alpamayo-1.5 rows on disk were produced with, so the two models join on
`clip_id` under the same estimator:

  clip -> 1 CoC rollout (do_sample, temperature 0.6, top_p 0.98)
       -> K=8 denoisings on that one CoC's KV cache, seed = clip_seed + k
       -> per-sample ADE/FDE against ego_future_xyz[..., :2]   (64 steps, 6.4 s)

The release API (`sample_trajectories_from_data(num_traj_samples=K)`) draws K
independent CoCs instead, which is a different estimator: CoC diversity alone lowers
minADE@K. `rollout_then_k_traj` below is that method's body with one generation and a
loop of seeded `diffusion.sample(batch_size=1)` calls over the same cache (the release
`step_fn` already crops the cache back after every expert forward). Nothing under
`src/` is edited.

Placement: 71.6 GB of bf16 weights do not fit one 48 GB card. VLM and expert both have
64 layers and expert layer i attends to VLM KV-cache layer i, so the split is by layer
index with `expert.expert.layers[i]` on the same card as the VLM's layer i.

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=4,5 python experiments/oodval/run_oodval.py --shard 0 --n-shards 2
"""

import os

# The cards come from the shell, never from an argument: the imports below initialise
# CUDA, after which CUDA_VISIBLE_DEVICES is ignored and the process lands on card 0 of
# a shared box. Refuse to start rather than guess.
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES to this replica's cards, e.g. 4,5")

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
COMPRESSION = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(Path(__file__).parent))
# numpy-only; imported rather than copied so bucket / ADE / degeneracy cannot drift
sys.path.insert(0, str(COMPRESSION / "experiments" / "head_analysis"))

import cache_a2 as ca
import eval_lib as el
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

from alpamayo2_super import helper
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.models.alpamayo2_super import (
    Alpamayo2Super,
    MaskDiscreteTrajectoryLogitsProcessor,
    _append_text_eos_mask,
)
from alpamayo2_super.models.expert_utils import (
    StopAfterEOS,
    build_expert_pos_ids_and_attn_mask,
    find_eos_offset,
    replace_padding_after_eos,
)
from alpamayo2_super.models.token_utils import extract_text_tokens
from alpamayo2_super.models.utils import fuse_traj_tokens

MODEL_ID = "nvidia/Alpamayo2-Super"
# pinned so a repo update cannot silently change the weights under a run
MODEL_REV = "00554695e729a6ff0b6281fd2c81b18d06e33dbe"
N_LAYERS = 64


def set_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def layer_device_map(n_gpus, first_share):
    """Split the 64 layer pairs over `n_gpus` visible cards.

    The first card also carries the vision tower and the embeddings, the last one the
    final norms and lm_head, so `first_share` (< 1/n_gpus) keeps them level.
    """
    n_first = round(N_LAYERS * first_share)
    rest = N_LAYERS - n_first
    bounds = [n_first + round(rest * j / (n_gpus - 1)) for j in range(n_gpus)]
    dev_of = [next(g for g, b in enumerate(bounds) if i < b) for i in range(N_LAYERS)]
    last = n_gpus - 1
    dm = {
        "vlm.model.visual": 0,
        "vlm.model.language_model.embed_tokens": 0,
        "vlm.model.language_model.rotary_emb": 0,
        "vlm.model.language_model.norm": last,
        "vlm.lm_head": last,
        "expert.expert.rotary_emb": 0,
        "expert.expert.norm": last,
        "expert.action_in_proj": 0,
        "expert.action_out_proj": last,
        "expert.action_space": 0,
        "expert.diffusion": 0,
        "history_traj_tokenizer": 0,
        "future_traj_tokenizer": 0,
    }
    for i, g in enumerate(dev_of):
        dm[f"vlm.model.language_model.layers.{i}"] = g
        dm[f"expert.expert.layers.{i}"] = g  # reads KV-cache layer i, so it lives with it
    return dm, [dev_of.count(g) for g in range(n_gpus)]


def build_inputs(model, processor, data):
    """`helper.prepare_model_inputs` with the processor built once instead of per clip."""
    messages = helper.create_messages(data, model.config)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         add_vision_id=False, continue_final_message=False)
    images = data["image_frames"].flatten(0, 1).float() / 255.0  # (24, 3, H, W)
    tokenized = dict(processor(text=text, images=images, videos=None, padding=False,
                               return_tensors="pt", do_rescale=False))
    return helper.to_device({"tokenized_data": tokenized,
                             "ego_history_xyz": data["ego_history_xyz"],
                             "ego_history_rot": data["ego_history_rot"]}, "cuda:0")


@torch.inference_mode()
def rollout_then_k_traj(model, model_inputs, base_seed, traj_seeds, max_gen, diffusion_kwargs):
    """One CoC rollout, then one denoising per seed on that rollout's KV cache.

    Returns (pred_k (K, T, 2) numpy, cot text, n generated tokens up to and including
    <traj_future_start>, has_eos, seconds for generation, seconds for denoising).
    """
    tokenized_data = dict(model_inputs["tokenized_data"])
    traj_data = {"ego_history_xyz": model_inputs["ego_history_xyz"],
                 "ego_history_rot": model_inputs["ego_history_rot"]}
    input_ids = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer,
                                 tokenized_data["input_ids"], traj_data, model.config.traj_ids)
    tokenized_data["input_ids"] = input_ids  # (1, L_prompt)
    prompt_len = input_ids.shape[1]

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.top_p = 0.98
    generation_config.temperature = 0.6
    generation_config.do_sample = True
    generation_config.max_new_tokens = max_gen
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    traj_ids = model.config.traj_ids
    logits_processor = LogitsProcessorList([MaskDiscreteTrajectoryLogitsProcessor(
        traj_token_offset=min(traj_ids["history_id0"], traj_ids["future_id0"]),
        traj_vocab_size=model.config.traj_vocab_size)])
    eos_token_id = traj_ids["future_start"]
    _append_text_eos_mask(logits_processor, generation_config.eos_token_id,
                          preserved_token_id=eos_token_id)
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])

    t_gen = time.time()
    seed_all(base_seed)
    vlm_outputs = model._generate_with_shared_prefill(
        tokenized_data, generation_config=generation_config, n_samples_total=1,
        stopping_criteria=stopping_criteria, logits_processor=logits_processor)
    vlm_outputs.sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences, eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id)  # (1, L_prompt + L_gen)
    torch.cuda.synchronize()
    t_gen = time.time() - t_gen

    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    device = input_ids.device
    n_diffusion_tokens = model.expert.action_space.get_action_space_dims()[0]
    has_eos = bool((vlm_outputs.sequences == eos_token_id).any())
    offset = find_eos_offset(sequences=vlm_outputs.sequences, eos_token_id=eos_token_id,
                             device=device)  # (1,)
    position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=vlm_outputs.rope_deltas, kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tokens, b_star=1, device=device,
        prefix_mask=tokenized_data.get("attention_mask"))

    forward_kwargs = {}
    if model.expert.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def step_fn(x, t):
        future_token_embeds = model.expert.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(1, n_diffusion_tokens, -1)
        expert_outputs = model.expert.expert(
            inputs_embeds=future_token_embeds, position_ids=position_ids,
            past_key_values=prompt_cache, attention_mask=attention_mask, use_cache=True,
            **forward_kwargs)
        prompt_cache.crop(prefill_seq_len)
        pred = model.expert.action_out_proj(expert_outputs.last_hidden_state)
        # the last layer sits on the last card; the Euler state lives on the first
        return pred.view(-1, *model.expert.action_space.get_action_space_dims()).to(x.device)

    hist_xyz = traj_data["ego_history_xyz"][:, -1]  # (1, 16, 3)
    hist_rot = traj_data["ego_history_rot"][:, -1]  # (1, 16, 3, 3)
    t_den = time.time()
    preds = []
    for s in traj_seeds:
        seed_all(s)
        action = model.expert.diffusion.sample(batch_size=1, step_fn=step_fn, device=device,
                                               return_all_steps=False, **diffusion_kwargs)
        pred_xyz, _ = model.expert.action_space.action_to_traj(action, hist_xyz, hist_rot)
        preds.append(pred_xyz[0, :, :2].float().cpu().numpy())  # (T, 2)
    torch.cuda.synchronize()
    t_den = time.time() - t_den

    cot = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)["cot"][0]
    gen_len = int(offset[0].item()) - prompt_len
    return np.stack(preds), cot, gen_len, has_eos, t_gen, t_den, prompt_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="a2_oodval")
    ap.add_argument("--k", type=int, default=8, help="trajectory samples per clip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--diffusion-steps", type=int, default=None, help="default: the model's own")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N clips of the shard")
    ap.add_argument("--first-share", type=float, default=0.47,
                    help="fraction of the 64 layers on the first card (2-card split)")
    ap.add_argument("--tag", default="alpamayo2")
    args = ap.parse_args()

    # with CUDA_DEVICE_ORDER=PCI_BUS_ID (env.sh) these are the indices nvidia-smi shows
    args.gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    n_gpus = torch.cuda.device_count()
    assert n_gpus == len(args.gpus.split(",")) >= 2, (n_gpus, args.gpus)

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{args.tag}_s{args.shard}of{args.n_shards}.json"

    man = pd.read_parquet(ca.MANIFEST)
    man = [{"clip_id": r.clip_id, "t0_us": int(r.t0_us), "gt_coc": r.gt_coc,
            "cluster": r.cluster, "split": r.split} for r in man.itertuples()]
    man = man[args.shard::args.n_shards]  # strided, the same rule as run_baseline.py
    if args.limit:
        man = man[: args.limit]

    set_determinism()
    share = args.first_share if n_gpus == 2 else 1.0 / n_gpus - 0.03
    device_map, per_gpu = layer_device_map(n_gpus, share)
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    print(f"{args.exp_id} {args.tag} | shard {args.shard}/{args.n_shards} | {len(man)} clips | "
          f"cards {args.gpus} ({gpu_names[0]}) | layers per card {per_gpu}", flush=True)

    t_load = time.time()
    model = Alpamayo2Super.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16,
                                           device_map=device_map)
    model.eval()
    processor = helper.get_processor(model.tokenizer, model.config)
    print(f"model loaded in {time.time() - t_load:.0f}s | attn "
          f"{model.vlm.config._attn_implementation} | mem GB "
          f"{[round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(n_gpus)]}",
          flush=True)

    diffusion_kwargs = {}
    if args.diffusion_steps:
        diffusion_kwargs["inference_step"] = args.diffusion_steps

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / f"config_s{args.shard}of{args.n_shards}.json").write_text(json.dumps({
        "model": MODEL_ID, "model_revision": MODEL_REV, "set": "ood", "manifest": str(ca.MANIFEST),
        "cache": str(ca.CACHE), "n_clips": len(man), "k": args.k, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample",
        "protocol": "1 CoC rollout, k denoisings on its KV cache (matched to run_baseline.py)",
        "sampling": {"do_sample": True, "temperature": 0.6, "top_p": 0.98},
        "max_gen": args.max_gen,
        "diffusion_steps": args.diffusion_steps or model.expert.diffusion.num_inference_steps,
        "input_profile": "trajectory: cameras [0,1,2,3,5,6] x 4 frames",
        "shard": [args.shard, args.n_shards], "gpus": args.gpus, "gpu_names": gpu_names,
        "layers_per_gpu": per_gpu, "attn_implementation": model.vlm.config._attn_implementation,
        "deterministic": {"use_deterministic_algorithms": True, "warn_only": True,
                          "cudnn_deterministic": True, "tf32": False,
                          "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
    }, indent=2))

    t_start = time.time()
    for i, m in enumerate(man):
        if m["clip_id"] in done:
            continue
        t0 = time.time()
        data = select_task_input(ca.load_cached(ca.path_for(m["clip_id"], m["t0_us"])),
                                 "trajectory")  # image_frames (6, 4, 3, H, W)
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].numpy()  # (64, 2)

        model_inputs = build_inputs(model, processor, data)

        base = ca.clip_seed(args.seed, m["clip_id"])
        traj_seeds = [base + k for k in range(args.k)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_k, gen_coc, gen_len, has_eos, t_gen, t_den, prompt_len = rollout_then_k_traj(
                model, model_inputs, base, traj_seeds, args.max_gen, diffusion_kwargs)

        ade_k, fde_k = el.ade_fde(pred_k, gt_xy)  # (K,), (K,)
        rec = {"clip_id": m["clip_id"], "bucket": el.bucket(gt_xy), "seed": base,
               "minADE_rollout": float(ade_k.min()), "minFDE_rollout": float(fde_k.min()),
               "ade_rollout_k": [round(float(x), 6) for x in ade_k],
               "fde_rollout_k": [round(float(x), 6) for x in fde_k],
               "gen_coc": gen_coc, "gen_len": gen_len, "has_eos": has_eos,
               **{f"coc_{k}": v for k, v in el.coc_degenerate(gen_coc).items()}}
        for h in (16, 32):
            a_h, f_h = el.ade_fde(pred_k[:, :h], gt_xy[:h])
            rec.update({f"ade_rollout_k_h{h}": [round(float(x), 6) for x in a_h],
                        f"fde_rollout_k_h{h}": [round(float(x), 6) for x in f_h]})
        rec["pred_xy_k"] = np.round(pred_k.astype(float), 3).tolist()  # (K, 64, 2), mm
        rec.update({"gt_coc": m["gt_coc"], "cluster": m["cluster"], "split": m["split"],
                    "prompt_len": prompt_len, "sec_gen": round(t_gen, 2),
                    "sec_denoise": round(t_den, 2), "sec_total": round(time.time() - t0, 2)})
        if not np.isfinite(pred_k).all():
            raise RuntimeError(f"non-finite trajectory on {m['clip_id']}")

        rows.append(rec)
        if len(rows) % 5 == 0 or i + 1 == len(man):
            rows_path.write_text(json.dumps(rows, indent=2))
        peak = [round(torch.cuda.max_memory_allocated(g) / 2**30, 1) for g in range(n_gpus)]
        print(f"[{i + 1}/{len(man)}] {m['clip_id'][:8]} {rec['bucket']:10s} "
              f"ade@8={rec['minADE_rollout']:.3f} gen={gen_len}tok {t_gen:.1f}s "
              f"den={t_den:.1f}s tot={time.time() - t0:.1f}s peakGB={peak} | {gen_coc[:90]}",
              flush=True)

    rows_path.write_text(json.dumps(rows, indent=2))
    a = np.array([r["minADE_rollout"] for r in rows])
    line = (f"{args.tag} ood-val shard {args.shard}/{args.n_shards}: {len(rows)} clips, "
            f"minADE@{args.k}(rollout) mean {a.mean():.4f} median {np.median(a):.4f}, "
            f"CoC degen {np.mean([r['coc_degenerate'] for r in rows]):.3f}")
    (out_dir / f"summary_s{args.shard}of{args.n_shards}.txt").write_text(line + "\n")
    print(f"\n{line}\n{(time.time() - t_start) / 60:.1f} min -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()
