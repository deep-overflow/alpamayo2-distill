"""P4 trajectory metrics of every model on the same clips: the ones a given student was scored on.

analyze_p4.py averages each model over its own clips, so a LiDAR student (only clips with a LiDAR
sweep) and the camera models (all clips) are averaged over slightly different sets. Here A2, A1.5
and the students are all restricted to the clips the --on student has.

Usage:
  python experiments/distill/analyze_p4_common.py --on lidar_s2 --students cam_s2 lidar_s2
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import analyze_p4 as p4

KEYS = ("minADE@1", "minADE@6", "minADE@8", "minFDE@8", "avgADE@8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["val_500", "ood_262"])
    ap.add_argument("--on", default="lidar_s2", help="student whose clips define the common set")
    ap.add_argument("--students", nargs="*", default=["cam_s2", "lidar_s2"])
    ap.add_argument("--repo", default=str(p4.REPO), help="checkout whose outputs/ holds the results")
    args = ap.parse_args()
    p4.REPO = Path(args.repo)
    res = {}
    for s in args.sets:
        models = {"a2": p4.load_a2(s), "a15": p4.load_a15(s), **{t: p4.load_student(t, s) for t in args.students}}
        common = set(models[args.on])
        for rows in models.values():
            common &= set(rows)
        r = res[s] = {"n_common": len(common)}
        for name, rows in models.items():
            tm = pd.DataFrame([p4.traj_metrics(rows[k]["traj"], p4.gt_xy(s, *k)) for k in sorted(common)])
            r[name] = {k: float(tm[k].mean()) for k in KEYS}
    out = p4.REPO / "outputs" / "distill_p4" / f"common_{args.on}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
