"""Diagnostic: is the student's trajectory gap in its CoC or in its expert?

For report-set clips the camera student's trajectories are compared three ways (same K and
same denoising seed rule as eval_student.py):
  free     the student's own CoC (outputs/distill_eval/<tag>, from eval_student.py)
  forced   the teacher's CoC fed to the student (teacher-forced), then the student's expert
  teacher  A2's trajectories (outputs/distill_teacher)
If forced ~ teacher, the gap comes from the student's CoC (exposure bias); if forced ~ free, the
student's expert / VLM features are the bottleneck.

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=0 python experiments/distill/diag_traj_forcing.py --n 100
"""

import os

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES, e.g. 0")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oodval"))

import distill_cache as dc
import distill_data as dd
import teacher_lib as tl
from analyze_p4 import boot_ci, gt_xy, traj_metrics
from eval_student import load_student

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_diag_forcing"


@tl.bf16
def forced_traj(model, seq, data, k, seed, device):
    ids = seq["input_ids"].to(device)
    out = model.vlm(input_ids=ids, attention_mask=seq["attention_mask"].to(device), use_cache=True,
                    pixel_values=seq["pixel_values"].to(device),
                    image_grid_thw=seq["image_grid_thw"].to(device), logits_to_keep=1)
    cache = out.past_key_values
    inp = {kk: data[kk].to(device) for kk in ("ego_history_xyz", "ego_history_rot")}
    return tl.denoise_batched(model, cache, model.vlm.model.rope_deltas, ids, cache.get_seq_length(),
                              inp, k=k, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/s2_wide/distill_s2_wide_final/best")
    ap.add_argument("--tag", default="cam_s2")
    ap.add_argument("--set", default="val_500")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    model, vmap, _, _ = load_student(args.model, device)
    b = dd.SeqBuilder(vocab_map=vmap)
    free_dir = REPO / "outputs" / "distill_eval" / args.tag / args.set
    rows = []
    for f in sorted(free_dir.glob("*.npz"))[:args.n]:
        z = np.load(f, allow_pickle=True)
        clip, t0, seed = str(z["clip_id"]), int(z["t0_us"]), int(z["seed"])
        lab = dd.load_label(args.set, "driving", clip, t0)
        data = dc.load_cached(dc.path_for(clip, t0, args.set))
        forced = forced_traj(model, b.driving(data, lab["gen_ids"]), data, z["traj"].shape[0],
                             seed + 1000, device)
        gt = gt_xy(args.set, clip, t0)
        row = {"clip_id": clip}
        for name, tr in (("free", z["traj"]), ("forced", forced), ("teacher", lab["traj"])):
            for kk, v in traj_metrics(tr[..., :2], gt).items():
                row[f"{name}/{kk}"] = float(v)
        rows.append(row)
    df = pd.DataFrame(rows)
    res = {"n": len(df), "model": args.model}
    for name in ("free", "forced", "teacher"):
        res[name] = {k: float(df[f"{name}/{k}"].mean()) for k in ("minADE@1", "minADE@6", "minADE@8", "minFDE@8")}
    for a, bb in (("forced", "free"), ("forced", "teacher")):
        d = (df[f"{a}/minADE@6"] - df[f"{bb}/minADE@6"]).to_numpy()
        res[f"{a}-{bb} minADE@6"] = {"mean": float(d.mean()), "ci95": boot_ci(d)}
    (OUT / f"{args.tag}_{args.set}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
