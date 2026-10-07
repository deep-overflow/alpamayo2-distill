"""Dual-objective Taylor importance on Alpamayo 2 Super, with the token-type split.

plans/2026-09-27_a2-dual-importance.md. The per-clip protocol is
alpamayo-model-compression's run_importance.py (the source of the paper's Fig. 1), and
the VLM gates are its gradient-anatomy TypedUnitGates, so one pass gives both:

  1. no-grad CoC rollout (release decoding, seed = sha256(f"42:{clip_id}")[:4])
  2. one teacher-forced VLM forward with grad on prompt + generated CoC -> KV cache
  3. CoC NLL backward (retain_graph)                          -> I_CoC by token type
  4. 10-step FM loss through the expert onto detached cache leaves, then ONE VLM backward
     seeded with the accumulated cache gradient               -> I_traj by token type
  5. expert gates read per denoising step                     -> I_traj(expert) per step

Summing the typed gradients over token types gives the plain gate gradient, so
|sum_types| is the shipped score (importance.npz); the typed arrays are the anatomy.

Memory: the 34B weights take 67 GB and the activations saved for the two backwards are
estimated at 110-115 GB. `--strategy offload` keeps them in host RAM
(torch.autograd.graph.save_on_cpu) so two cards suffice; `--strategy none` keeps them on
the cards and needs four. Gradient checkpointing is not an option: transformers turns
use_cache off under it, and the recompute would re-append to the KV cache this pass needs.

Per-clip results go to <out>/perclip/<clip_id>.{npz,json} (resumable, shard-safe);
`--finalize` stacks them into the compression repo's file layout.

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=4,5 python experiments/importance/run_importance_a2.py \
      --strategy offload --shard 0 --n-shards 2
  python experiments/importance/run_importance_a2.py --finalize
"""

import os
import sys

FINALIZE = "--finalize" in sys.argv
if not FINALIZE and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    # the imports below initialise CUDA; set after that, this is ignored and the process
    # lands on card 0 of a shared box
    raise SystemExit("set CUDA_VISIBLE_DEVICES to this replica's cards, e.g. 4,5")
if FINALIZE:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "oodval"))

import a2_grad_lib as gl
import cache_a2 as ca

MANIFEST = Path("/mnt/nvme1n1/ad_vla/outputs/chan/eval_sets/calib_100.parquet")
CACHE = ca.AV / "pre_processed" / "calib_a2"
NT = len(gl.TYPES)


def host_gb():
    vm = psutil.virtual_memory()
    return psutil.Process().memory_info().rss / 2**30, vm.available / 2**30


def process_clip(model, processor, data, args, seed, dims):
    from run_oodval import build_inputs, seed_all

    n_vlm, H, D, I, _, EH, ED, EI = dims
    vlm_layers = model.vlm.model.language_model.layers
    dev0 = gl.module_device(model.vlm.model.language_model.embed_tokens)
    t_all = time.time()
    model_inputs = build_inputs(model, processor, data)

    seed_all(seed)
    t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        seq, prompt_len, has_eos, cot = gl.rollout_sequence(model, model_inputs, args.max_gen)
    seq = seq.clone()  # out of inference mode, so autograd may save what derives from it
    sec_roll = time.time() - t
    coc_start, coc_end = prompt_len, seq.shape[1]
    sp = gl.special_ids(model)
    # Alpamayo 2 writes the CoC straight into the assistant turn and closes it with
    # <|traj_future_start|> alone (1.5 wrote <|cot_end|><|traj_future_start|>); the CoC span
    # is everything generated under both, so the NLL includes the closing token either way
    gen_ids = seq[0, prompt_len:].tolist()
    tail_ok = gen_ids[-1:] == [sp["traj_future_start"]] and sp["cot_end"] not in gen_ids
    type_idx = gl.token_types(model, seq, prompt_len)  # (T,)
    n_tok = np.bincount(type_idx.numpy(), minlength=NT)
    x1 = gl.gt_actions(model, data, dev0)  # (1, 64, 2) fp32

    gates = gl.TypedUnitGates(vlm_layers, H, D, I, NT)
    gates.set_types(type_idx)
    # Q heads only, as 1.5's --verify: a plain MLP gate would save one more fp32 (T, 25600)
    # tensor per layer (~30 GB of host RAM here)
    ref = gl.UnitGates(vlm_layers, H, D, I, mlp=False) if args.verify else None
    exp_gates = gl.UnitGates(model.expert.expert.layers, EH, ED, EI)
    # G0b: a second, never-zeroed set of expert gates on the same pass; its accumulated
    # grad must equal |sum_s| of what exp_gates records step by step
    exp_ref = gl.UnitGates(model.expert.expert.layers, EH, ED, EI) if args.verify else None

    ram_before = host_gb()
    if args.strategy == "hybrid":
        offload = gl.GpuFirstOffload(args.keep_ptrs, args.budget, pin=True)
    elif args.strategy.startswith("offload"):
        offload = torch.autograd.graph.save_on_cpu(pin_memory=args.strategy == "offload")
    else:
        offload = nullcontext()
    t = time.time()
    with offload, torch.autocast("cuda", dtype=torch.bfloat16):
        hidden, cache, rope_deltas = gl.vlm_forward_with_grad(
            model, seq, model_inputs["tokenized_data"])
    sec_fwd = time.time() - t
    ram_after_fwd = host_gb()
    if ram_after_fwd[1] < args.min_free_ram_gb:
        raise SystemExit(f"host RAM available {ram_after_fwd[1]:.0f} GB < "
                         f"{args.min_free_ram_gb} GB after the forward; stopping")
    cache_t = [gl.cache_layer_kv(cache, i) for i in range(n_vlm)]
    if not all(k.requires_grad and v.requires_grad for k, v in cache_t):
        raise RuntimeError("cache k/v carry no graph; enable_input_require_grads() missing")
    for k, v in cache_t:
        k.retain_grad()
        v.retain_grad()
    prefill = cache.get_seq_length()

    verify = []

    def read():
        q, mlp = gates.q_signed(), gates.mlp_signed()  # (5, L, H), (5, L, I)
        if ref is not None:
            s, p = q.sum(0), ref.q_signed()
            den = np.linalg.norm(p, axis=1)
            ok = den > 0
            verify.append(float((np.linalg.norm(s - p, axis=1)[ok] / den[ok]).max()))
            ref.zero_grads()
        gates.zero_grads()
        return q, mlp

    # ---- objective 1: CoC NLL ----
    t = time.time()
    nll = gl.coc_nll(model, hidden, seq, coc_start, coc_end)
    nll.backward(retain_graph=True)
    ce_q, ce_mlp = read()
    coc_kv = gl.kv_group_scores(cache_t)
    for k, v in cache_t:
        k.grad = None
        v.grad = None
    nll_val = nll.item()
    del hidden, nll
    sec_ce = time.time() - t

    # ---- objective 2: trajectory flow matching ----
    t = time.time()
    fm_loss, grads, leaves, steps = gl.expert_fm_grads(
        model, cache, rope_deltas, x1, args.fm_steps, seed, prefill,
        step_gates=None if args.no_step_record else exp_gates)
    if steps is None:
        exp_q, exp_mlp = exp_gates.q_signed(), exp_gates.mlp_signed()  # (L, EH), (L, EI)
    else:
        exp_q, exp_mlp = steps["q"].sum(0), steps["mlp"].sum(0)
    g0b = None
    if exp_ref is not None:
        g0b = {}
        for name, rec_s, acc_s in (("q", exp_q, exp_ref.q_signed()), ("mlp", exp_mlp, exp_ref.mlp_signed())):
            den = np.linalg.norm(acc_s, axis=1)
            g0b[name] = float((np.linalg.norm(np.abs(rec_s) - np.abs(acc_s), axis=1) / den).max())
    direct = gl.cache_read(leaves, grads, type_idx, NT)  # (L, 5)
    sec_exp = time.time() - t
    t = time.time()
    ts, gs = [], []
    for (k, v), (gk, gv) in zip(cache_t, grads):
        for t_, g_ in ((k, gk), (v, gv)):
            if g_ is not None:
                ts.append(t_)
                gs.append(g_.to(t_.dtype))
    torch.autograd.backward(ts, gs)
    fm_q, fm_mlp = read()
    traj_kv = gl.kv_group_scores(cache_t)
    sec_fm = time.time() - t

    peak = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
            for i in range(torch.cuda.device_count())]
    gates.remove()
    exp_gates.remove()
    if ref is not None:
        ref.remove()
    if exp_ref is not None:
        exp_ref.remove()
    del cache, cache_t, leaves, grads, ts, gs
    torch.cuda.empty_cache()

    arrays = {
        "ce_q_type": ce_q.astype(np.float32), "fm_q_type": fm_q.astype(np.float32),
        "ce_mlp_type": ce_mlp.astype(np.float32), "fm_mlp_type": fm_mlp.astype(np.float32),
        "traj_exp_q_signed": exp_q.astype(np.float32),
        "traj_exp_mlp_signed": exp_mlp.astype(np.float32),
        "coc_kv_k": coc_kv[0], "coc_kv_v": coc_kv[1],
        "traj_kv_k": traj_kv[0], "traj_kv_v": traj_kv[1],
        "direct": direct, "n_tok": n_tok,
    }
    if steps is not None:
        arrays["exp_q_step"] = steps["q"].astype(np.float32)  # (S, L, EH) signed
        arrays["exp_mlp_step"] = steps["mlp"].astype(np.float32)  # (S, L, EI) signed
    rec = {"coc_len": int(coc_end - coc_start), "prompt_len": int(prompt_len),
           "has_eos": has_eos, "tail_ok": tail_ok,
           "cot": cot, "nll": nll_val, "fm_loss": fm_loss,
           "n_tok": {k: int(n) for k, n in zip(gl.TYPES, n_tok)},
           "peak_gb": peak,
           "saved_to_host_gb": round(offload.to_host / 2**30, 1) if args.strategy == "hybrid" else None,
           "saved_kept_on_gpu_gb": [round(v / 2**30, 1) for v in offload.used.values()]
           if args.strategy == "hybrid" else None, "host_rss_gb_after_fwd": round(ram_after_fwd[0], 1),
           "host_rss_gb_before_fwd": round(ram_before[0], 1),
           "host_avail_gb_after_fwd": round(ram_after_fwd[1], 1),
           "sec": {"rollout": round(sec_roll, 1), "forward": round(sec_fwd, 1),
                   "ce_backward": round(sec_ce, 1), "fm_expert": round(sec_exp, 1),
                   "fm_backward": round(sec_fm, 1), "total": round(time.time() - t_all, 1)}}
    if args.verify:
        rec["typed_vs_plain_relerr_q"] = verify  # [CE, FM], max over layers
        rec["g0b_steps_vs_accumulated_relerr"] = g0b
    return arrays, rec


def finalize(out_dir, allow_partial):
    """Stack the per-clip files into the layouts of importance_v2 / gradanat_v1 / stepimp_*."""
    clips = pd.read_parquet(MANIFEST)["clip_id"].tolist()
    have = [c for c in clips if (out_dir / "perclip" / f"{c}.npz").exists()]
    if len(have) != len(clips) and not allow_partial:
        raise SystemExit(f"{len(have)}/{len(clips)} clips done; pass --allow-partial to finalize anyway")
    recs, stack = [], {}
    for c in have:
        z = np.load(out_dir / "perclip" / f"{c}.npz")
        for k in z.files:
            stack.setdefault(k, []).append(z[k])
        r = json.loads((out_dir / "perclip" / f"{c}.json").read_text())
        r["clip_id"] = c
        recs.append(r)
    s = {k: np.stack(v) for k, v in stack.items()}  # (N, ...)
    pooled = {"coc_vlm_q": np.abs(s["ce_q_type"].sum(1)), "traj_vlm_q": np.abs(s["fm_q_type"].sum(1)),
              "coc_vlm_mlp": np.abs(s["ce_mlp_type"].sum(1)),
              "traj_vlm_mlp": np.abs(s["fm_mlp_type"].sum(1)),
              "traj_exp_q": np.abs(s["traj_exp_q_signed"]),
              "traj_exp_mlp": np.abs(s["traj_exp_mlp_signed"])}
    # the CoC NLL has no path into the expert; the 1.5 files carry these keys as zeros
    pooled["coc_exp_q"] = np.zeros_like(pooled["traj_exp_q"])
    pooled["coc_exp_mlp"] = np.zeros_like(pooled["traj_exp_mlp"])
    for k in ("coc_kv_k", "coc_kv_v", "traj_kv_k", "traj_kv_v"):
        pooled[k] = s[k]
    np.savez(out_dir / "importance.npz", **{k: v.mean(0) for k, v in pooled.items()})
    np.savez(out_dir / "importance_perclip.npz", **{k: v.astype(np.float32) for k, v in pooled.items()})
    np.savez(out_dir / "anatomy_perclip_q.npz", ce=s["ce_q_type"], fm_full=s["fm_q_type"])
    np.savez(out_dir / "anatomy_perclip_mlp.npz", ce_type=s["ce_mlp_type"], fm_type=s["fm_mlp_type"])
    np.savez(out_dir / "anatomy_direct.npz", direct=s["direct"], n_tok=s["n_tok"])
    if "exp_q_step" in s:
        st = {}
        for ax in ("q", "mlp"):
            g = s[f"exp_{ax}_step"]  # (N, S, L, U) signed
            st[f"{ax}_abs_step"] = np.abs(g).mean(0)
            st[f"{ax}_signed_step"] = g.mean(0)
            st[f"{ax}_shipped"] = np.abs(g.sum(1)).mean(0)
        np.savez(out_dir / "step_importance.npz", **st)
        np.savez(out_dir / "step_importance_perclip.npz",
                 **{f"{ax}_mass": np.abs(s[f"exp_{ax}_step"]).sum((2, 3)) for ax in ("q", "mlp")})
    (out_dir / "metrics.json").write_text(json.dumps({"n_clips": len(recs), "per_clip": recs}, indent=2))
    print(f"finalized {len(recs)} clips -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="a2_importance_v1")
    ap.add_argument("--strategy", choices=["hybrid", "offload", "offload_nopin", "none"],
                    default="hybrid",
                    help="hybrid: saved tensors stay on their card up to its budget, the rest "
                         "goes to pinned host RAM, parameters are never copied (4 cards); "
                         "offload: torch's save_on_cpu, which also copies the weights")
    ap.add_argument("--gpu-reserve-gb", type=float, default=12.0,
                    help="hybrid: headroom left free on every card for the backward itself")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N clips of the shard")
    ap.add_argument("--clip", default=None, help="run this one clip id only (smoke)")
    ap.add_argument("--first-share", type=float, default=0.47,
                    help="fraction of the 64 layers on the first card (2-card split)")
    ap.add_argument("--min-free-ram-gb", type=float, default=60.0,
                    help="stop if host RAM available drops below this after a forward")
    ap.add_argument("--verify", action="store_true",
                    help="G0: plain gates next to the typed ones; checks sum_types == plain")
    ap.add_argument("--no-step-record", action="store_true",
                    help="G0b: accumulate the expert gates over steps (shipped path) instead")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    if args.finalize:
        finalize(out_dir, args.allow_partial)
        return
    (out_dir / "perclip").mkdir(parents=True, exist_ok=True)

    from run_oodval import MODEL_ID, MODEL_REV, layer_device_map

    from alpamayo2_super import helper
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    man = pd.read_parquet(MANIFEST)
    man = [(r.clip_id, int(r.t0_us)) for r in man.itertuples()]
    if args.clip:
        man = [m for m in man if m[0] == args.clip]
    man = man[args.shard::args.n_shards]
    if args.limit:
        man = man[: args.limit]

    n_gpus = torch.cuda.device_count()
    share = args.first_share if n_gpus == 2 else 1.0 / n_gpus - 0.03
    device_map, per_gpu = layer_device_map(n_gpus, share)
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    print(f"{args.exp_id} | shard {args.shard}/{args.n_shards} | {len(man)} clips | cards "
          f"{os.environ['CUDA_VISIBLE_DEVICES']} ({gpu_names[0]}) | layers per card {per_gpu} | "
          f"strategy {args.strategy}", flush=True)

    t = time.time()
    model = Alpamayo2Super.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16,
                                           device_map=device_map)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # every weight is frozen and the gates sit downstream of k/v, so without this the early
    # layers' cache tensors would carry no graph
    model.vlm.enable_input_require_grads()
    processor = helper.get_processor(model.tokenizer, model.config)
    args.keep_ptrs = gl.param_storage_ptrs(model)
    args.budget = {d: max(0, torch.cuda.get_device_properties(d).total_memory
                          - torch.cuda.memory_allocated(d) - int(args.gpu_reserve_gb * 2**30))
                   for d in range(n_gpus)}
    print("hybrid budget per card GB:", [round(b / 2**30, 1) for b in args.budget.values()], flush=True)
    tc, ec = model.vlm.config.text_config, model.expert.expert.config
    dims = (tc.num_hidden_layers, tc.num_attention_heads, tc.head_dim, tc.intermediate_size,
            ec.num_hidden_layers, ec.num_attention_heads, ec.head_dim, ec.intermediate_size)
    print(f"model loaded in {time.time() - t:.0f}s | VLM L{dims[0]} Q{dims[1]} I{dims[3]} | "
          f"expert L{dims[4]} Q{dims[5]} I{dims[7]} | attn {model.vlm.config._attn_implementation}",
          flush=True)

    (out_dir / f"config_s{args.shard}of{args.n_shards}.json").write_text(json.dumps({
        "model": MODEL_ID, "model_revision": MODEL_REV,
        "plan": "plans/2026-09-27_a2-dual-importance.md",
        "protocol": "alpamayo-model-compression run_importance.py (importance_v2) + "
                    "TypedUnitGates (gradanat_v1) + per-step expert gates (stepimp_fm_perstep)",
        "objectives": {"coc": "own-rollout CoC NLL", "traj": "flow-matching MSE vs GT action"},
        "fm_estimator": "t_s=(s+0.5)/S, fresh CPU-generator noise per step seeded with the clip "
                        "seed, x_t=t*x1+(1-t)*eps, target x1-eps, x_t fed in fp32 under bf16 "
                        "autocast (Alpamayo 2's own Euler-state dtype)",
        "types": gl.TYPES, "manifest": str(MANIFEST), "cache": str(CACHE),
        "n_clips_shard": len(man), "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]", "fm_steps": args.fm_steps,
        "max_gen": args.max_gen, "strategy": args.strategy, "gpu_reserve_gb": args.gpu_reserve_gb,
        "verify": args.verify,
        "step_record": not args.no_step_record, "gpus": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_names": gpu_names, "layers_per_gpu": per_gpu,
        "shapes": {"vlm_q": [dims[0], dims[1]], "vlm_mlp": [dims[0], dims[3]],
                   "exp_q": [dims[4], dims[5]], "exp_mlp": [dims[4], dims[7]]},
        "torch": torch.__version__,
    }, indent=2))

    t_start = time.time()
    for i, (clip_id, t0_us) in enumerate(man):
        dst = out_dir / "perclip" / f"{clip_id}.npz"
        if dst.exists():
            continue
        for d in range(n_gpus):
            torch.cuda.reset_peak_memory_stats(d)
        data = select_task_input(ca.load_cached(ca.path_for(clip_id, t0_us, CACHE)), "trajectory")
        arrays, rec = process_clip(model, processor, data, args, ca.clip_seed(args.seed, clip_id),
                                   dims)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        (out_dir / "perclip" / f"{clip_id}.json").write_text(json.dumps(rec, indent=2))
        tmp.rename(dst)
        v = (f" verify {rec['typed_vs_plain_relerr_q']} g0b {rec['g0b_steps_vs_accumulated_relerr']}"
             if args.verify else "")
        print(f"[{i + 1}/{len(man)}] {clip_id[:8]} coc={rec['coc_len']} nll={rec['nll']:.3f} "
              f"fm={rec['fm_loss']:.4f} tail_ok={rec['tail_ok']} "
              f"peakGB={rec['peak_gb']} rss={rec['host_rss_gb_after_fwd']}GB "
              f"avail={rec['host_avail_gb_after_fwd']}GB tohost={rec['saved_to_host_gb']}GB "
              f"ongpu={rec['saved_kept_on_gpu_gb']} sec={rec['sec']}{v} | {rec['cot'][:70]}",
              flush=True)
    print(f"done {len(man)} clips in {(time.time() - t_start) / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
