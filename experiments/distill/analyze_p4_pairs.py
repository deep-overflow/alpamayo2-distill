"""P4 follow-up: camera vs camera + LiDAR student on exactly the same clips, plus gate G0e.

analyze_p4.py scores each model on its own clip set (the LiDAR student only has the clips with a
LiDAR cache). Here every metric is restricted to the clips both students have, and per-clip
differences get paired bootstrap 95% CIs:
  detection   TP / FP / reference counts vs the projected obstacle boxes (IoU 0.5; clips with
              references only, as in analyze_p4.det_scores), and F1 vs A2 (all common clips)
  reasoning   meta-action all-three agreement with A2, CoC ROUGE-L vs A2
G0e: share of A2's generated tokens on the report sets that fall outside the student vocabulary.
Also plots the two training runs' dev curves side by side (dev_curves_<a>_vs_<b>.png).

Usage:
  python experiments/distill/analyze_p4_pairs.py --a lidar_s2 --b cam_s2
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import distill_data as dd
from analyze_p4 import FIELDS, boot_ci, load_a2, load_student, meta_fields, rouge_l
from det_utils import match, parse_boxes, to_unit1000

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_p4"


def per_clip(rows, a2, ref):
    have = set(zip(ref.clip_id, ref.t0_us))
    out = {}
    for k, r in rows.items():
        refc = ref[(ref.clip_id == k[0]) & (ref.t0_us == k[1])]
        kept, ign = refc[refc.keep], refc[~refc.keep]
        pb = [to_unit1000(b, "qwen_1000") for b in parse_boxes(r["det"])[0]]
        tp, fp, _ = match(pb, kept[["x1", "y1", "x2", "y2"]].to_numpy(),
                          ign[["x1", "y1", "x2", "y2"]].to_numpy(), 0.5)
        tb = [to_unit1000(b, "qwen_1000") for b in parse_boxes(a2[k]["det"])[0]]
        atp, afp, _ = match(pb, np.asarray(tb).reshape(-1, 4), np.zeros((0, 4)), 0.5)
        ms, mt = meta_fields(r["meta"]), meta_fields(a2[k]["meta"])
        out[k] = {"has_ref": k in have, "tp": tp, "fp": fp, "n_ref": len(kept), "a2_tp": atp, "a2_fp": afp, "a2_n": len(tb),
                  "meta3": float(all(ms[f] is not None and ms[f] == mt[f] for f in FIELDS)),
                  "rougeL": rouge_l(r["cot"], a2[k]["cot"])}
    return pd.DataFrame(out).T


def plot_dev(runs, path):
    keys = ("loss", "ce_drv", "kl_drv", "ce_det", "kl_det", "fm")
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3))
    for name, exp in runs.items():
        dv = [x for x in json.loads((REPO / "outputs" / exp / "metrics.json").read_text()) if x["kind"] == "dev"]
        for ax, k in zip(axes, keys):
            ax.plot([x["step"] for x in dv], [x[k] for x in dv], marker=".", label=name)
            ax.set_title(f"dev {k}", fontsize=9)
            ax.set_xlabel("step")
            if k != "loss":
                ax.set_ylim(top=min(ax.get_ylim()[1], 2 * min(x[k] for x in dv)))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="lidar_s2")
    ap.add_argument("--b", default="cam_s2")
    ap.add_argument("--sets", nargs="+", default=["val_500", "ood_262"])
    ap.add_argument("--runs", nargs=2, default=["distill_s2_wide_lidar", "distill_s2_wide_final"],
                    help="training exp ids of --a and --b (for the dev-curve plot)")
    args = ap.parse_args()
    plot_dev(dict(zip((args.a, args.b), args.runs)), OUT / f"dev_curves_{args.a}_vs_{args.b}.png")
    res = {}
    for s in args.sets:
        a2 = load_a2(s)
        ref = pd.read_parquet(REPO / "outputs" / "distill_refboxes" / f"{s}.parquet")
        A, B = load_student(args.a, s), load_student(args.b, s)
        common = sorted(set(A) & set(B) & set(a2))
        da = per_clip({k: A[k] for k in common}, a2, ref)
        db = per_clip({k: B[k] for k in common}, a2, ref)
        hr = da.has_ref.astype(bool)
        r = res[s] = {"n_common": len(common), "n_common_with_refboxes": int(hr.sum())}
        for name, d in ((args.a, da), (args.b, db)):
            dr = d[hr]
            p = dr.tp.sum() / max(dr.tp.sum() + dr.fp.sum(), 1)
            rc = dr.tp.sum() / max(dr.n_ref.sum(), 1)
            ap_, ar = d.a2_tp.sum() / max(d.a2_tp.sum() + d.a2_fp.sum(), 1), d.a2_tp.sum() / max(d.a2_n.sum(), 1)
            r[name] = {"det_P": float(p), "det_R": float(rc), "det_F1_vs_a2": float(2 * ap_ * ar / max(ap_ + ar, 1e-9)),
                       "meta3_vs_a2": float(d.meta3.mean()), "rougeL_vs_a2": float(d.rougeL.mean())}
        for key in ("meta3", "rougeL", "tp", "fp"):
            sel = hr if key in ("tp", "fp") else slice(None)
            diff = (da[key][sel] - db[key][sel]).astype(float).to_numpy()
            r[f"{args.a}-{args.b} {key}/clip"] = {"mean": float(diff.mean()), "ci95": boot_ci(diff)}
        # G0e: teacher-generated tokens outside the student vocabulary
        vmap = np.load("/mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/s2_wide/distill_s2_wide_final/best/vocab_map.npy")
        n = oov = 0
        for task in ("driving", "det"):
            for c, t in dd.labelled_items(s, tasks=(task,)):
                g = dd.load_label(s, task, c, t)["gen_ids"]
                n += len(g)
                oov += int((vmap[np.clip(g, 0, len(vmap) - 1)] < 0).sum())
        r["G0e_teacher_tokens"] = n
        r["G0e_oov_rate"] = oov / max(n, 1)
    (OUT / f"pairs_{args.a}_vs_{args.b}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
