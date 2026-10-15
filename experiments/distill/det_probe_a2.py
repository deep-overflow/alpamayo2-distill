"""Gate G0d: can Alpamayo 2 Super detect every object on the front-wide camera?

A2 was released with single-object grounding only (notebooks/vqa.ipynb). This probe asks it
for all vehicles / pedestrians / cyclists on the front-wide t0 frame with three phrasings,
on the six-camera grounding profile and the release grounding sampling (temperature 0.1,
top_p 1.0), and scores the parsed boxes against the projected obstacle.offline boxes
(project_obstacles.py) on the calib_100 clips.

Coordinates are scored under both readings, Qwen3-VL relative 0..1000 and model-input pixels
(576x320); the better-aligned one is reported as A2's convention. Matching is class-agnostic
Hungarian on IoU at 0.5 and 0.3; a prediction overlapping (IoU >= 0.3) a projected box that
the reference filters dropped (too small / far / clipped) is ignored, not a false positive.

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=2,3 python experiments/distill/det_probe_a2.py --limit 10   # smoke
  CUDA_VISIBLE_DEVICES=2,3 python experiments/distill/det_probe_a2.py
"""

import os

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES, e.g. 2,3")

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import distill_cache as dc
import teacher_lib as tl
from run_oodval import set_determinism

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_p0_detprobe"
REF = REPO / "outputs" / "distill_refboxes" / "calib_100.parquet"
from det_utils import DIST_BINS, PROMPTS, match, parse_boxes, to_unit1000


def overlay(clip_id, t0_us, pred, ref, title, path):
    data = dc.load_cached(dc.path_for(clip_id, t0_us, "calib_100"))
    img = data["image_frames"][dc.FRONT_WIDE_SLOT, -1].permute(1, 2, 0).numpy()  # (320, 576, 3)
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.imshow(img)
    for boxes, color, ls in ((ref, "#2a78d6", "-"), (pred, "#e8452c", "--")):
        for b in boxes:
            x1, y1, x2, y2 = b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.5, ls=ls,
                                       ec=color))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--overlays", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)

    ref_all = pd.read_parquet(REF)
    man = pd.read_parquet(REPO / "outputs" / "distill_sets" / "calib_100.parquet")
    info = pd.read_parquet(REF.with_name("calib_100_info.parquet"))
    have = set(info[info.status == "ok"].clip_id)  # clips with zero projected boxes count too
    man = [r for r in man.itertuples()
           if r.clip_id in have and dc.path_for(r.clip_id, r.t0_us, "calib_100").exists()]
    if args.limit:
        man = man[:args.limit]

    set_determinism()
    t = time.time()
    model, processor, per_gpu = tl.load_teacher()
    print(f"teacher loaded in {time.time() - t:.0f}s, layers per card {per_gpu}, "
          f"{len(man)} clips x {len(PROMPTS)} prompts", flush=True)

    rows_path = OUT / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {(r["clip_id"], r["prompt"]) for r in rows}
    for i, r in enumerate(man):
        data = dc.load_cached(dc.path_for(r.clip_id, r.t0_us, "calib_100"))
        for pname, q in PROMPTS.items():
            if (r.clip_id, pname) in done:
                continue
            inp = tl.det_inputs(model, processor, data, q)
            g = tl.generate(model, inp, seed=args.seed, driving=False, max_new=args.max_new,
                            temperature=0.1, top_p=1.0)
            boxes, how = parse_boxes(g["answer"])
            rows.append({"clip_id": r.clip_id, "t0_us": int(r.t0_us), "prompt": pname,
                         "answer": g["answer"], "parse": how, "n_pred": len(boxes),
                         "boxes_raw": [list(b) for b in boxes], "gen_len": len(g["gen_ids"]),
                         "hit_stop": g["hit_stop"], "sec": g["sec"]})
        rows_path.write_text(json.dumps(rows, indent=1))
        last = [x for x in rows if x["clip_id"] == r.clip_id]
        print(f"[{i + 1}/{len(man)}] {r.clip_id[:8]} " + "  ".join(
            f"{x['prompt'][:2]}: {x['n_pred']} boxes/{x['gen_len']} tok/{x['sec']:.1f}s"
            for x in last), flush=True)

    # ---- scoring ----
    df = pd.DataFrame(rows)
    metrics, per_clip = {"n_clips": int(df.clip_id.nunique())}, []
    for pname in PROMPTS:
        sub = df[df.prompt == pname]
        for mode in ("qwen_1000", "pixel"):
            acc = {thr: {"tp": 0, "fp": 0, "n_ref": 0, "n_pred": 0} for thr in (0.5, 0.3)}
            by_cls = {thr: {} for thr in (0.5, 0.3)}
            by_dist = {thr: {} for thr in (0.5, 0.3)}
            for x in sub.itertuples():
                refc = ref_all[(ref_all.clip_id == x.clip_id) & (ref_all.t0_us == x.t0_us)]
                kept, ign = refc[refc.keep], refc[~refc.keep]
                rb = kept[["x1", "y1", "x2", "y2"]].to_numpy()
                ib = ign[["x1", "y1", "x2", "y2"]].to_numpy()
                pb = [to_unit1000(b, mode) for b in x.boxes_raw]
                for thr in (0.5, 0.3):
                    tp, fp, mref = match(pb, rb, ib, thr)
                    a = acc[thr]
                    a["tp"] += tp
                    a["fp"] += fp
                    a["n_ref"] += len(rb)
                    a["n_pred"] += len(pb)
                    for cls, hit in zip(kept.coarse, mref):
                        c = by_cls[thr].setdefault(cls, [0, 0])
                        c[0] += int(hit)
                        c[1] += 1
                    for d, hit in zip(kept.dist_m, mref):
                        lo, hi = next(b for b in DIST_BINS if b[0] <= d < b[1])
                        c = by_dist[thr].setdefault(f"{lo}-{hi if hi < 1e9 else 'inf'}m", [0, 0])
                        c[0] += int(hit)
                        c[1] += 1
                    if thr == 0.5 and mode == "qwen_1000":
                        per_clip.append({"clip_id": x.clip_id, "prompt": pname, "tp": tp,
                                         "fp": fp, "n_ref": len(rb), "n_pred": len(pb)})
            for thr in (0.5, 0.3):
                a = acc[thr]
                key = f"{pname}/{mode}/iou{thr}"
                metrics[key] = {
                    "precision": a["tp"] / max(a["tp"] + a["fp"], 1),
                    "recall": a["tp"] / max(a["n_ref"], 1),
                    "n_pred": a["n_pred"], "n_ref": a["n_ref"], "tp": a["tp"], "fp": a["fp"],
                    "recall_by_class": {k: [v[0] / max(v[1], 1), v[1]] for k, v in by_cls[thr].items()},
                    "recall_by_dist": {k: [v[0] / max(v[1], 1), v[1]] for k, v in by_dist[thr].items()},
                }
        metrics[f"{pname}/gen"] = {
            "parse": sub.parse.value_counts().to_dict(), "mean_pred": float(sub.n_pred.mean()),
            "mean_gen_len": float(sub.gen_len.mean()), "hit_stop": float(sub.hit_stop.mean()),
            "sec_per_clip": float(sub.sec.mean()),
            "labels": pd.Series([b[4] for bs in sub.boxes_raw for b in bs]).str.lower()
                        .value_counts().head(12).to_dict(),
        }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(per_clip).to_csv(OUT / "per_clip_qwen1000_iou05.csv", index=False)
    (OUT / "config.json").write_text(json.dumps({
        "set": "calib_100 (clips with projected reference boxes)", "n_clips": len(man),
        "prompts": PROMPTS, "sampling": {"temperature": 0.1, "top_p": 1.0, "seed": args.seed,
                                         "max_new_tokens": args.max_new},
        "input_profile": "grounding: cameras [0,1,2,3,5,6] x 4 frames, vqa message layout",
        "reference": str(REF), "coordinate_modes": ["qwen_1000", "pixel (576x320)"],
        "matching": "class-agnostic Hungarian IoU, thresholds 0.5/0.3; preds on filtered refs "
                    "(IoU>=0.3) ignored", "gpus": os.environ["CUDA_VISIBLE_DEVICES"]}, indent=2))

    lines = [f"G0d A2 detection probe, {len(man)} calib clips, front-wide t0 frame"]
    for pname in PROMPTS:
        gm = metrics[f"{pname}/gen"]
        lines.append(f"\n{pname}: mean {gm['mean_pred']:.1f} boxes, {gm['mean_gen_len']:.0f} tok, "
                     f"{gm['sec_per_clip']:.1f} s/clip, parse {gm['parse']}, labels {gm['labels']}")
        for mode in ("qwen_1000", "pixel"):
            for thr in (0.5, 0.3):
                m = metrics[f"{pname}/{mode}/iou{thr}"]
                cls = ", ".join(f"{k} {v[0]:.2f} (n={v[1]})" for k, v in m["recall_by_class"].items())
                dist = ", ".join(f"{k} {v[0]:.2f}" for k, v in m["recall_by_dist"].items())
                lines.append(f"  {mode:9s} IoU{thr}: P {m['precision']:.3f} R {m['recall']:.3f} "
                             f"(pred {m['n_pred']}, ref {m['n_ref']}) | {cls} | {dist}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    for x in [x for x in rows if x["prompt"] == "p1_qwen_json"][:args.overlays]:
        refc = ref_all[(ref_all.clip_id == x["clip_id"]) & ref_all.keep]
        overlay(x["clip_id"], x["t0_us"], [to_unit1000(b, "qwen_1000") for b in x["boxes_raw"]],
                refc[["x1", "y1", "x2", "y2"]].to_numpy().tolist(),
                f"{x['clip_id'][:8]} blue = projected reference, red dashed = A2 (p1, 0..1000)",
                OUT / "plots" / f"{x['clip_id'][:8]}_p1.png")


if __name__ == "__main__":
    main()
