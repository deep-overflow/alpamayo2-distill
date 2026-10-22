"""P4 metrics: teacher (A2), Alpamayo 1.5, and the distilled students on the report sets.

Models (each optional, used when its files exist):
  a2        outputs/distill_teacher/<set>/{driving,det}   (K = 8 trajectories, batched)
  a15       outputs/chan/baseline_pred_{indist,oodval}    (val_500 = indist_500; trajectories only)
  <tag>     outputs/distill_eval/<tag>/<set>               (eval_student.py)
Metrics per set:
  trajectory  minADE@1/@6/@8, minFDE@8, avgADE@8 against the cached GT future (xy, 6.4 s)
  detection   front-wide boxes vs the projected obstacle boxes (project_obstacles; IoU 0.5 and
              0.3, filtered boxes ignored), by class and distance; and F1@0.5 against A2's boxes
  reasoning   meta-action agreement with A2 (each field and all three), CoC ROUGE-L (word LCS)
              against A2's CoC, share of outputs that ended with the stop token
  latency     prefill / CoC generation / denoising / detection seconds per sample (students)
Paired comparisons (bootstrap 95% CI of the mean difference, 2,000 resamples) on the clips
both models have: every student vs a2, and --pair A B (e.g. lidar_s2 cam_s2).

Usage:
  python experiments/distill/analyze_p4.py --students cam_s2 lidar_s2 --pair lidar_s2 cam_s2
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import distill_cache as dc
from det_utils import DIST_BINS, match, parse_boxes, to_unit1000

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_p4"
A15 = {"val_500": "/mnt/nvme1n1/ad_vla/outputs/chan/baseline_pred_indist",
       "ood_262": "/mnt/nvme1n1/ad_vla/outputs/chan/baseline_pred_oodval"}
FIELDS = ("Longitudinal", "Lateral", "Lane")


def gt_xy(set_name, clip_id, t0_us):
    z = np.load(dc.path_for(clip_id, t0_us, set_name), allow_pickle=True)
    return z["ego_future_xyz"][0, 0, :, :2]  # (64, 2)


def traj_metrics(pred_xy, gt):
    """pred_xy (K, 64, 2) -> dict of per-sample trajectory errors."""
    d = np.linalg.norm(pred_xy - gt[None], axis=-1)  # (K, 64)
    ade, fde = d.mean(1), d[:, -1]
    return {"minADE@1": ade[0], "minADE@6": ade[:6].min(), "minADE@8": ade[:8].min(),
            "minFDE@8": fde[:8].min(), "avgADE@8": ade[:8].mean()}


def meta_fields(text):
    out = {}
    for f in FIELDS:
        m = re.search(rf"{f}\s*:\s*([^.\n]+)", text or "")
        out[f] = m.group(1).strip().lower() if m else None
    return out


def rouge_l(a, b):
    x, y = (a or "").lower().split(), (b or "").lower().split()
    if not x or not y:
        return 0.0
    dp = np.zeros((len(x) + 1, len(y) + 1), np.int32)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            dp[i + 1, j + 1] = dp[i, j] + 1 if xi == yj else max(dp[i, j + 1], dp[i + 1, j])
    lcs = dp[-1, -1]
    p, r = lcs / len(y), lcs / len(x)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def load_a2(set_name):
    rows = {}
    root = REPO / "outputs" / "distill_teacher" / set_name
    for f in (root / "driving").glob("*.npz"):
        z = np.load(f, allow_pickle=True)
        if "traj" not in z.files:
            continue
        txt = json.loads(str(z["text"]))
        rows[(str(z["clip_id"]), int(z["t0_us"]))] = {"traj": z["traj"][..., :2], "cot": txt["cot"],
                                                      "meta": txt["meta_action"],
                                                      "drv_stop": bool(z["hit_stop"])}
    for f in (root / "det").glob("*.npz"):
        z = np.load(f, allow_pickle=True)
        k = (str(z["clip_id"]), int(z["t0_us"]))
        if k in rows:
            rows[k]["det"] = json.loads(str(z["text"]))["answer"]
            rows[k]["det_stop"] = bool(z["hit_stop"])
    return rows


def load_student(tag, set_name):
    rows = {}
    for f in (REPO / "outputs" / "distill_eval" / tag / set_name).glob("*.npz"):
        z = np.load(f, allow_pickle=True)
        txt = json.loads(str(z["text"]))
        rows[(str(z["clip_id"]), int(z["t0_us"]))] = {
            "traj": z["traj"][..., :2], "cot": txt["cot"], "meta": txt["meta_action"],
            "det": txt["det_answer"], "drv_stop": txt["drv_hit_stop"], "det_stop": txt["det_hit_stop"],
            "timing": z["timing"]}
    return rows


def load_a15(set_name):
    """Alpamayo-1.5 rows are keyed by clip only (same clip and t0 as our manifest)."""
    man = pd.read_parquet(REPO / "outputs" / "distill_sets" / f"{set_name}.parquet")
    t0_of = dict(zip(man.clip_id, man.t0_us.astype(int)))
    rows = {}
    for f in sorted(Path(A15[set_name]).glob("baseline_s*of*.json")):
        for r in json.loads(Path(f).read_text()):
            if r["clip_id"] not in t0_of:
                continue
            pred = r["pred_xy_k"]
            rows[(r["clip_id"], t0_of[r["clip_id"]])] = {
                "traj": np.asarray(json.loads(pred) if isinstance(pred, str) else pred, dtype=np.float32)}
    return rows


def boot_ci(diff, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def det_scores(rows, ref, set_name):
    """Matching against the projected obstacle boxes, LiDAR clips with references only."""
    acc = {thr: {"tp": 0, "fp": 0, "n_ref": 0, "cls": {}, "dist": {}} for thr in (0.5, 0.3)}
    have = set(zip(ref.clip_id, ref.t0_us))
    n = 0
    for k, r in rows.items():
        if k not in have or "det" not in r:
            continue
        n += 1
        refc = ref[(ref.clip_id == k[0]) & (ref.t0_us == k[1])]
        kept, ign = refc[refc.keep], refc[~refc.keep]
        rb, ib = kept[["x1", "y1", "x2", "y2"]].to_numpy(), ign[["x1", "y1", "x2", "y2"]].to_numpy()
        pb = [to_unit1000(b, "qwen_1000") for b in parse_boxes(r["det"])[0]]
        for thr in (0.5, 0.3):
            tp, fp, mref = match(pb, rb, ib, thr)
            a = acc[thr]
            a["tp"] += tp
            a["fp"] += fp
            a["n_ref"] += len(rb)
            for cls, hit in zip(kept.coarse, mref):
                a["cls"].setdefault(cls, [0, 0])
                a["cls"][cls][0] += int(hit)
                a["cls"][cls][1] += 1
            for d, hit in zip(kept.dist_m, mref):
                lo, hi = next(b for b in DIST_BINS if b[0] <= d < b[1])
                key = f"{lo}-{hi if hi < 1e9 else 'inf'}m"
                a["dist"].setdefault(key, [0, 0])
                a["dist"][key][0] += int(hit)
                a["dist"][key][1] += 1
    out = {"n_clips": n}
    for thr, a in acc.items():
        out[f"iou{thr}"] = {"precision": a["tp"] / max(a["tp"] + a["fp"], 1),
                            "recall": a["tp"] / max(a["n_ref"], 1), "tp": a["tp"], "fp": a["fp"],
                            "n_ref": a["n_ref"],
                            "recall_by_class": {c: [v[0] / max(v[1], 1), v[1]] for c, v in a["cls"].items()},
                            "recall_by_dist": {c: [v[0] / max(v[1], 1), v[1]] for c, v in a["dist"].items()}}
    return out


def agreement_with_a2(rows, a2):
    keys = [k for k in rows if k in a2 and "det" in rows[k] and "det" in a2[k]]
    tp = fp = n_ref = 0
    meta = {f: [] for f in FIELDS}
    meta_all, rl = [], []
    for k in keys:
        pb = [to_unit1000(b, "qwen_1000") for b in parse_boxes(rows[k]["det"])[0]]
        tb = [to_unit1000(b, "qwen_1000") for b in parse_boxes(a2[k]["det"])[0]]
        t, f, _ = match(pb, np.asarray(tb).reshape(-1, 4), np.zeros((0, 4)), 0.5)
        tp, fp, n_ref = tp + t, fp + f, n_ref + len(tb)
        ms, mt = meta_fields(rows[k]["meta"]), meta_fields(a2[k]["meta"])
        for fld in FIELDS:
            meta[fld].append(ms[fld] is not None and ms[fld] == mt[fld])
        meta_all.append(all(ms[fld] is not None and ms[fld] == mt[fld] for fld in FIELDS))
        rl.append(rouge_l(rows[k]["cot"], a2[k]["cot"]))
    p, r = tp / max(tp + fp, 1), tp / max(n_ref, 1)
    return {"n_clips": len(keys), "det_f1_vs_a2@0.5": 2 * p * r / max(p + r, 1e-9),
            "det_precision_vs_a2": p, "det_recall_vs_a2": r,
            **{f"meta_agree_{f}": float(np.mean(v)) for f, v in meta.items()},
            "meta_agree_all3": float(np.mean(meta_all)), "coc_rougeL_vs_a2": float(np.mean(rl))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["val_500", "ood_262"])
    ap.add_argument("--students", nargs="*", default=[])
    ap.add_argument("--pair", nargs=2, default=None, metavar=("A", "B"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, lines = {}, []
    for s in args.sets:
        models = {"a2": load_a2(s), "a15": load_a15(s), **{t: load_student(t, s) for t in args.students}}
        ref = pd.read_parquet(REPO / "outputs" / "distill_refboxes" / f"{s}.parquet")
        m = metrics[s] = {}
        per = {}
        for name, rows in models.items():
            if not rows:
                continue
            tm = pd.DataFrame([{"key": k, **traj_metrics(r["traj"], gt_xy(s, *k))} for k, r in rows.items()]).set_index("key")
            per[name] = tm
            m[name] = {"n": len(tm), "traj": tm.mean().to_dict()}
            if name != "a15":
                m[name]["det_vs_refboxes"] = det_scores(rows, ref, s)
                m[name]["stop_rate"] = {"driving": float(np.mean([r["drv_stop"] for r in rows.values()])),
                                        "det": float(np.mean([r.get("det_stop", False) for r in rows.values()]))}
            if name not in ("a2", "a15"):
                m[name]["vs_a2"] = agreement_with_a2(rows, models["a2"])
                tim = np.stack([r["timing"] for r in rows.values()])
                m[name]["latency_s"] = dict(zip(("prefill", "coc_gen", "denoise", "det"), tim.mean(0).round(3).tolist()))
        pairs = [(t, "a2") for t in args.students] + ([tuple(args.pair)] if args.pair else [])
        for a, b in pairs:
            if a not in per or b not in per:
                continue
            common = per[a].index.intersection(per[b].index)
            m[f"{a}-{b}"] = {"n": len(common), **{
                k: {"mean_diff": float((per[a].loc[common, k] - per[b].loc[common, k]).mean()),
                    "ci95": boot_ci((per[a].loc[common, k] - per[b].loc[common, k]).to_numpy())}
                for k in ("minADE@1", "minADE@6", "minADE@8", "minFDE@8", "avgADE@8")}}
        lines.append(f"== {s}")
        for name, v in m.items():
            if "traj" in v:
                t = v["traj"]
                det = v.get("det_vs_refboxes", {}).get("iou0.5", {})
                agr = v.get("vs_a2", {})
                lines.append(f"{name:10s} n={v['n']:4d} minADE@1 {t['minADE@1']:.3f} @6 {t['minADE@6']:.3f} "
                             f"@8 {t['minADE@8']:.3f} minFDE@8 {t['minFDE@8']:.3f} avgADE {t['avgADE@8']:.3f}"
                             + (f" | det P {det['precision']:.3f} R {det['recall']:.3f}" if det else "")
                             + (f" | vs A2: detF1 {agr['det_f1_vs_a2@0.5']:.3f} meta3 {agr['meta_agree_all3']:.3f}"
                                f" rougeL {agr['coc_rougeL_vs_a2']:.3f}" if agr else ""))
            else:
                lines.append(f"{name:18s} n={v['n']} " + " ".join(
                    f"{k} {d['mean_diff']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]"
                    for k, d in v.items() if k != "n"))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
