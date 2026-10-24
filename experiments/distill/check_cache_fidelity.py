"""Gate G0a: does storing frames at model-input size (320x576, JPEG q95) change the teacher?

Four input variants of the same sample go through A2:
  raw       frames decoded straight from the dataset video, full resolution (reference)
  jpg_full  raw -> JPEG q95 at full resolution (the cache every earlier A2/A1.5 run used)
  small_u8  raw -> processor resize -> uint8 (what a lossless 320x576 cache would hold)
  cached    the distill cache itself (small_u8 -> JPEG q95)

Per variant: pixel_values difference to raw; teacher-forced log-probs of one CoC sampled
from raw (KL(raw || variant) per generated token, top-1 agreement); and K=8 denoisings with
the same seeds on that variant's cache (same-seed ADE to raw). Seed noise = ADE between
raw denoisings with different seeds.

Pass (fixed before the run): mean same-seed ADE(raw, cached) < 0.5 x mean seed-noise ADE,
and mean KL(raw || cached) <= 3 x mean KL(raw || jpg_full). On failure the cache is rebuilt
lossless (small_u8), which this script also scores.

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=0,1 python experiments/distill/check_cache_fidelity.py --n 20
"""

import os

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES, e.g. 0,1")

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import distill_cache as dc
import teacher_lib as tl
from run_oodval import set_determinism

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_g0a"
VARIANTS = ("jpg_full", "small_u8", "cached")


def jpeg_roundtrip(frames):
    """(C, F, 3, H, W) uint8 -> same through JPEG q95."""
    out = torch.empty_like(frames)
    for c in range(frames.shape[0]):
        for f in range(frames.shape[1]):
            buf = io.BytesIO()
            Image.fromarray(frames[c, f].permute(1, 2, 0).numpy()).save(buf, format="JPEG",
                                                                         quality=95)
            out[c, f] = torch.from_numpy(np.asarray(Image.open(buf))).permute(2, 0, 1)
    return out


def load_raw(clip_id, t0_us, avdi):
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

    return dc.with_retry(load_physical_aiavdataset, clip_id, t0_us=t0_us, avdi=avdi,
                         maybe_stream=True, camera_features=list(dc.CAMERA_FEATURES),
                         include_calibration=False)


def ade(a, b):
    """(64, 3) x2 -> mean xy displacement."""
    return float(np.linalg.norm(a[:, :2] - b[:, :2], axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    man = pd.read_parquet(REPO / "outputs" / "distill_sets" / "calib_100.parquet")
    man = [r for r in man.itertuples() if dc.path_for(r.clip_id, r.t0_us, "calib_100").exists()]
    man = man[:args.n]
    avdi = dc.make_avdi()
    pool = ThreadPoolExecutor(max_workers=3)
    raw_futs = [pool.submit(load_raw, r.clip_id, int(r.t0_us), avdi) for r in man]

    set_determinism()
    t = time.time()
    model, processor, per_gpu = tl.load_teacher()
    print(f"teacher loaded in {time.time() - t:.0f}s, layers per card {per_gpu}", flush=True)

    rows = []
    for r, fut in zip(man, raw_futs):
        raw = fut.result()
        cached = dc.load_cached(dc.path_for(r.clip_id, r.t0_us, "calib_100"))
        small = dc.resize_like_processor(processor.image_processor, raw["image_frames"].flatten(0, 1))
        data = {"raw": raw,
                "jpg_full": {**raw, "image_frames": jpeg_roundtrip(raw["image_frames"])},
                "small_u8": {**raw, "image_frames": small.view(*raw["image_frames"].shape[:2],
                                                               *small.shape[1:])},
                "cached": {**raw, "image_frames": cached["image_frames"]}}
        assert np.allclose(cached["ego_history_xyz"].numpy(), raw["ego_history_xyz"].numpy())
        inputs = {k: tl.driving_inputs(model, processor, v) for k, v in data.items()}
        seed = args.seed
        g = tl.generate(model, inputs["raw"], seed=seed, driving=True)
        del g["out"]
        row = {"clip_id": r.clip_id, "gen_len": len(g["gen_ids"]), "hit_stop": g["hit_stop"],
               "cot": g["cot"], "meta_action": g["meta_action"]}

        seeds = [seed + 1000 + j for j in range(args.k)]
        px_raw = inputs["raw"]["tok"]["pixel_values"].float()
        logp, trajs = {}, {}
        for name in ("raw", *VARIANTS):
            lp, cache, rope, kv_len = tl.forced_forward(model, inputs[name], g["gen_ids"])
            seq = torch.cat([inputs[name]["tok"]["input_ids"],
                             torch.as_tensor(g["gen_ids"], device="cuda:0").view(1, -1)], dim=1)
            trajs[name] = tl.denoise(model, cache, rope, seq, kv_len, inputs[name], seeds)
            logp[name] = lp.cpu()
            del cache
            if name != "raw":
                px = inputs[name]["tok"]["pixel_values"].float()
                row[f"px_maxabs_{name}"] = float((px - px_raw).abs().max())
                row[f"px_meanabs_{name}"] = float((px - px_raw).abs().mean())
        p_raw = logp["raw"].exp()
        gt = raw["ego_future_xyz"][0, 0].numpy()  # (64, 3)
        row["seed_noise_ade"] = float(np.mean([ade(trajs["raw"][j], trajs["raw"][(j + 1) % args.k])
                                               for j in range(args.k)]))
        row["minade_raw"] = min(ade(x, gt) for x in trajs["raw"])
        for name in VARIANTS:
            kl = (p_raw * (logp["raw"] - logp[name])).sum(-1)  # (n,)
            row[f"kl_mean_{name}"] = float(kl.mean())
            row[f"kl_max_{name}"] = float(kl.max())
            row[f"top1_agree_{name}"] = float((logp["raw"].argmax(-1) == logp[name].argmax(-1))
                                              .float().mean())
            row[f"same_seed_ade_{name}"] = float(np.mean([ade(trajs["raw"][j], trajs[name][j])
                                                          for j in range(args.k)]))
            row[f"minade_{name}"] = min(ade(x, gt) for x in trajs[name])
        rows.append(row)
        print(f"[{len(rows)}/{len(man)}] {r.clip_id[:8]} gen {row['gen_len']} | KL jpg_full "
              f"{row['kl_mean_jpg_full']:.2e} small {row['kl_mean_small_u8']:.2e} cached "
              f"{row['kl_mean_cached']:.2e} | same-seed ADE cached {row['same_seed_ade_cached']:.3f} "
              f"vs seed noise {row['seed_noise_ade']:.3f}", flush=True)
        (OUT / "rows.json").write_text(json.dumps(rows, indent=1))

    df = pd.DataFrame(rows)
    metrics = {"n": len(df)}
    for name in VARIANTS:
        for m in ("kl_mean", "top1_agree", "same_seed_ade", "px_meanabs", "px_maxabs", "minade"):
            metrics[f"{m}_{name}"] = float(df[f"{m}_{name}"].mean())
    metrics["seed_noise_ade"] = float(df.seed_noise_ade.mean())
    metrics["minade_raw"] = float(df.minade_raw.mean())
    crit_traj = metrics["same_seed_ade_cached"] < 0.5 * metrics["seed_noise_ade"]
    crit_kl = metrics["kl_mean_cached"] <= 3 * metrics["kl_mean_jpg_full"]
    metrics["pass_traj"], metrics["pass_kl"] = bool(crit_traj), bool(crit_kl)
    metrics["pass"] = bool(crit_traj and crit_kl)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "config.json").write_text(json.dumps({
        "set": "calib_100", "n": len(df), "k": args.k, "seed": args.seed,
        "task": "meta_action (CoC + meta-action), temperature 0.6, top_p 0.98",
        "variants": ["raw", *VARIANTS], "cache": str(dc.ROOT / "calib_100"),
        "criteria": {"traj": "same_seed_ade_cached < 0.5 * seed_noise_ade",
                     "kl": "kl_mean_cached <= 3 * kl_mean_jpg_full"},
        "gpus": os.environ["CUDA_VISIBLE_DEVICES"]}, indent=2))
    lines = [f"G0a cache fidelity, {len(df)} calib clips, K={args.k} same-seed denoisings",
             (f"{'variant':10s} {'KL mean':>10s} {'top1':>7s} {'sameseedADE':>12s} "
             f"{'px mean':>9s} {'px max':>8s} {'minADE@K':>9s}")]
    for name in VARIANTS:
        lines.append(f"{name:10s} {metrics[f'kl_mean_{name}']:10.2e} "
                     f"{metrics[f'top1_agree_{name}']:7.4f} {metrics[f'same_seed_ade_{name}']:12.4f} "
                     f"{metrics[f'px_meanabs_{name}']:9.4f} {metrics[f'px_maxabs_{name}']:8.3f} "
                     f"{metrics[f'minade_{name}']:9.3f}")
    lines += [(f"seed-noise ADE (raw, different seeds): {metrics['seed_noise_ade']:.4f}   "
              f"minADE@K raw {metrics['minade_raw']:.3f}"),
              f"criterion traj (cached < 0.5 x seed noise): {'PASS' if crit_traj else 'FAIL'}",
              f"criterion KL (cached <= 3 x jpg_full): {'PASS' if crit_kl else 'FAIL'}",
              f"G0a: {'PASS' if metrics['pass'] else 'FAIL'}"]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
