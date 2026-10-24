"""G0d follow-up: are A2's missed boxes the ones the projected references should not count?

The projected references keep occluded objects (no LiDAR visibility test). A box-level proxy
for occlusion: rasterise every projected box of the frame (kept or not) far-to-near on a
250x250 grid of the [0, 1000] image; a reference's visible fraction is the share of its area
not covered by any nearer box. Recall of each prompt is re-scored on references with visible
fraction >= {0.0, 0.3, 0.5, 0.7} and by box height. Nothing here changes the G0d criteria.

Usage:
  python experiments/distill/analyze_detprobe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from det_utils import PROMPTS, match, to_unit1000

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "outputs" / "distill_p0_detprobe"
REF = REPO / "outputs" / "distill_refboxes" / "calib_100.parquet"
G = 250  # raster cells per side


def visible_fraction(boxes):
    """boxes: DataFrame of one frame (all projected boxes) -> Series of visible fractions."""
    order = boxes.sort_values("depth_m").index  # near first
    covered = np.zeros((G, G), bool)
    vis = {}
    for i in order:
        b = boxes.loc[i]
        x1, y1 = int(b.x1 / 1000 * G), int(b.y1 / 1000 * G)
        x2, y2 = max(int(np.ceil(b.x2 / 1000 * G)), x1 + 1), max(int(np.ceil(b.y2 / 1000 * G)), y1 + 1)
        cell = covered[y1:y2, x1:x2]
        vis[i] = 1.0 - cell.mean() if cell.size else 0.0
        covered[y1:y2, x1:x2] = True
    return pd.Series(vis)


def main():
    rows = pd.DataFrame(json.loads((PROBE / "rows.json").read_text()))
    ref = pd.read_parquet(REF)
    ref["vis_occ"] = 1.0
    for _, g in ref.groupby(["clip_id", "t0_us"]):
        ref.loc[g.index, "vis_occ"] = visible_fraction(g)
    out = {}
    for pname in PROMPTS:
        sub = rows[rows.prompt == pname]
        res = {}
        for thr_vis in (0.0, 0.3, 0.5, 0.7):
            hit = tot = 0
            for x in sub.itertuples():
                r = ref[(ref.clip_id == x.clip_id) & (ref.t0_us == x.t0_us)]
                kept = r[r.keep]
                eval_ref = kept[kept.vis_occ >= thr_vis]
                ign = pd.concat([r[~r.keep], kept[kept.vis_occ < thr_vis]])
                pb = [to_unit1000(b, "qwen_1000") for b in x.boxes_raw]
                _, _, mref = match(pb, eval_ref[["x1", "y1", "x2", "y2"]].to_numpy(),
                                   ign[["x1", "y1", "x2", "y2"]].to_numpy(), 0.5)
                hit += int(mref.sum())
                tot += len(eval_ref)
            res[f"recall_vis>={thr_vis}"] = [hit / max(tot, 1), tot]
        for lo, hi in ((12, 30), (30, 60), (60, 1e9)):
            hit = tot = 0
            for x in sub.itertuples():
                r = ref[(ref.clip_id == x.clip_id) & (ref.t0_us == x.t0_us)]
                kept = r[r.keep & (r.vis_occ >= 0.5)]
                band = kept[(kept.h_px_small >= lo) & (kept.h_px_small < hi)]
                ign = pd.concat([r[~r.keep], kept[(kept.h_px_small < lo) | (kept.h_px_small >= hi)]])
                pb = [to_unit1000(b, "qwen_1000") for b in x.boxes_raw]
                _, _, mref = match(pb, band[["x1", "y1", "x2", "y2"]].to_numpy(),
                                   ign[["x1", "y1", "x2", "y2"]].to_numpy(), 0.5)
                hit += int(mref.sum())
                tot += len(band)
            res[f"recall_vis>=0.5_h{lo}-{hi if hi < 1e9 else 'inf'}px"] = [hit / max(tot, 1), tot]
        out[pname] = res
    (PROBE / "occlusion_analysis.json").write_text(json.dumps(out, indent=2))
    for p, res in out.items():
        print(p)
        for k, (v, n) in res.items():
            print(f"  {k:32s} {v:.3f} (n={n})")


if __name__ == "__main__":
    main()
