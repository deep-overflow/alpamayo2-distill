"""Gate: `run_oodval.rollout_then_k_traj` reproduces the release sampling path.

The runner re-implements `Alpamayo2Super.sample_trajectories_from_data` so that one CoC
can be denoised K times. This checks the re-implementation against the release method
itself on a few clips: same seed before generation, and the release `diffusion.sample`
wrapped so it is reseeded exactly as the runner reseeds its sample k=0. The CoC text and
the trajectory must then agree.

Two details make the release method runnable across two cards unmodified: the inputs
come from `helper.prepare_model_inputs` (the release tokenisation, so this also covers
`run_oodval.build_inputs`), and `expert.action_out_proj` is mapped to card 0 so the
velocity it returns is on the Euler state's device.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python experiments/oodval/check_native_parity.py --n 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

import cache_a2 as ca
import run_oodval as ro

from alpamayo2_super import helper
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ro.set_determinism()
    device_map, _ = ro.layer_device_map(torch.cuda.device_count(), 0.47)
    device_map["expert.action_out_proj"] = 0
    model = Alpamayo2Super.from_pretrained(ro.MODEL_ID, revision=ro.MODEL_REV,
                                           dtype=torch.bfloat16, device_map=device_map)
    model.eval()
    processor = helper.get_processor(model.tokenizer, model.config)

    native_sample = model.expert.diffusion.sample
    man = pd.read_parquet(ca.MANIFEST).head(args.n)
    ok = True
    for r in man.itertuples():
        data = select_task_input(ca.load_cached(ca.path_for(r.clip_id, int(r.t0_us))),
                                 "trajectory")
        base = ca.clip_seed(args.seed, r.clip_id)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_mine, cot_mine, *_ = ro.rollout_then_k_traj(
                model, ro.build_inputs(model, processor, data), base, [base], 256, {})

        def reseeded(*a, _base=base, **kw):
            ro.seed_all(_base)  # what the runner does before its sample k=0
            return native_sample(*a, **kw)

        model.expert.diffusion.sample = reseeded
        inputs = helper.to_device(
            helper.prepare_model_inputs(data, model.config, model.tokenizer), "cuda:0")
        ro.seed_all(base)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, _, extra = model.sample_trajectories_from_data(
                data=inputs, top_p=0.98, temperature=0.6, num_traj_samples=1,
                max_generation_length=256, return_extra=True)
        model.expert.diffusion.sample = native_sample
        pred_native = pred_xyz[0, 0, 0, :, :2].float().cpu().numpy()  # (64, 2)
        cot_native = str(extra["cot"][0, 0, 0])

        diff = float(np.abs(pred_native - pred_mine[0]).max())
        same = cot_native == cot_mine
        ok &= same and diff < 1e-3
        print(f"{r.clip_id[:8]} | CoC identical {same} | max|pred_native - pred_runner| "
              f"{diff:.2e} m | {cot_native[:70]}", flush=True)
    print("PARITY OK" if ok else "PARITY FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
