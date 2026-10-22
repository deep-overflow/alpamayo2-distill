"""Detection labels from the projected obstacle.offline auto-labels, with a LiDAR visibility test.

For every sample of a set (front-wide t0 frame, clips with a LiDAR cache):
  1. project_obstacles.project_sample -> image boxes at the frame time T (old filter: `keep`)
  2. points back from the cached range image (build_lidar_distill), in rig@t0: range r from the
     sensor, rig z, column-centre azimuth phi; the horizontal radius rho solves
     |(rho cos phi, rho sin phi, z) - s_c| = r with s_c the LiDAR position at the column's
     capture time (t0 - dt_col) moved to rig@t0
  3. each track's cuboid interpolated to the capture time of its azimuth and moved to rig@t0
     with the egomotion (the transform of build_lidar_distill.deskew); points inside it (0.2 m
     margin, from 0.2 m above the bottom face) -> `n_lidar`; `n_lidar_notc` is the same count
     with the cuboid at T (no time compensation)
  4. keep_v2 = keep & n_lidar >= k & coarse class in {vehicle, pedestrian, cyclist}
Label sequences: A2's JSON answer format, nearest first, at most --max-boxes boxes, teacher ids
ending with <|im_end|>; no box -> "[ ]" (A2 never answers empty, and "[]" is not a student token).

Outputs:
  outputs/distill_gtboxes/<set>.parquet, <set>_info.parquet, <set>_stats.json, overlays/
  outputs/distill_gtdet/<set>/det/<clip>__t0_<t0>.npz  (gen_ids, text)

Usage (after `source experiments/oodval/env.sh`):
  python experiments/distill/gt_det_labels.py --set val_500 --overlays 20
  python experiments/distill/gt_det_labels.py --set val_500 --labels-only --k 3
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt

sys.path.insert(0, str(Path(__file__).parent))

import build_lidar_distill as bl
import distill_cache as dc
import project_obstacles as po
from det_utils import parse_boxes

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_gtboxes"
LABELS = REPO / "outputs" / "distill_gtdet"
CLASSES = ("vehicle", "pedestrian", "cyclist")
NAMES = {"automobile": "Car", "other_vehicle": "Car", "heavy_truck": "Truck", "bus": "Bus",
         "trailer": "Trailer", "train_or_tram_car": "Tram", "person": "Pedestrian",
         "stroller": "Pedestrian", "rider": "Cyclist"}
DIST_BINS = ((0, 30), (30, 60), (60, 1e9))
VOCAB_MAP = Path("/mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/s2_wide/distill_s2_wide_final/best/vocab_map.npy")


def label_path(set_name, clip_id, t0_us):
    return LABELS / set_name / "det" / f"{clip_id}__t0_{int(t0_us)}.npz"


def lidar_points(rimg, s_col):
    """Range image (4, H, W) + LiDAR position per column in rig@t0 (W, 3) -> points (N, 3)."""
    b, c = np.nonzero(rimg[3] > 0.5)
    r = np.expm1(rimg[0, b, c] * np.log1p(bl.MAX_RANGE))
    z = rimg[2, b, c] * 5.0
    phi = (c + 0.5) / bl.W * 2 * np.pi - np.pi
    s = s_col[c]  # (N, 3)
    cos, sin = np.cos(phi), np.sin(phi)
    bb = s[:, 0] * cos + s[:, 1] * sin
    cc = s[:, 0] ** 2 + s[:, 1] ** 2 + (z - s[:, 2]) ** 2 - r ** 2
    rho = bb + np.sqrt(np.maximum(bb ** 2 - cc, 0.0))
    return np.stack([rho * cos, rho * sin, z], 1)


def to_t0(tr, t_us, ego, t0_us):
    """One interpolated track row (rig@t_us) -> centre (3,), axes (3, 3) as rows, in rig@t0."""
    ctr = np.array([tr.cx, tr.cy, tr.cz])
    rot = spt.Rotation.from_quat([tr.qx, tr.qy, tr.qz, tr.qw])
    p = np.vstack([ctr, ctr + rot.apply(np.eye(3))])  # (4, 3)
    q = bl.deskew(p, np.full(4, int(t_us), np.int64), ego, t0_us)
    return q[0], q[1:] - q[0]


def count_in(pts, ctr, axes, size, margin=0.2, floor=0.2):
    loc = (pts - ctr) @ axes.T  # (N, 3) in the cuboid frame
    h = np.asarray(size) / 2
    return int(((np.abs(loc[:, 0]) <= h[0] + margin) & (np.abs(loc[:, 1]) <= h[1] + margin)
                & (loc[:, 2] >= -h[2] + floor) & (loc[:, 2] <= h[2] + margin)).sum())


def visibility_sample(task):
    """One sample -> (boxes with LiDAR counts, info)."""
    clip_id, t0_us, set_name, args = task
    boxes, info = po.project_sample(task)
    if info.get("status") != "ok" or not boxes:
        return boxes, info
    avdi = po._avdi()
    extr = dc.with_retry(avdi.get_clip_feature, clip_id, "sensor_extrinsics", maybe_stream=True)
    ego = dc.with_retry(avdi.get_clip_feature, clip_id, "egomotion", maybe_stream=True)
    z = np.load(dc.path_for(clip_id, t0_us, set_name), allow_pickle=True)
    T = int(z["absolute_timestamps"][z["camera_indices"].tolist().index(1), -1])
    lz = bl.load_cached(bl.path_for(clip_id, t0_us, set_name))
    t_col = (t0_us - lz["dt_col"].astype(np.float64) * 1e3).astype(np.int64)  # (W,) capture times
    t_lidar = extr.sensor_poses[bl.LIDAR].apply(np.zeros((1, 3)))[0]
    s_col = bl.deskew(np.tile(t_lidar, (bl.W, 1)), t_col, ego, t0_us)  # (W, 3)
    pts = lidar_points(lz["range_img"], s_col)
    obs = pd.read_parquet(dc.OBSTACLES / f"{clip_id}.parquet")
    gap = args.max_gap_ms * 1000
    at_T = po.interpolate_tracks(obs, T, gap).set_index("track_id")
    for bx in boxes:
        tr = at_T.loc[bx["track_id"]]
        c_T, a_T = to_t0(tr, T, ego, t0_us)
        size = (tr.sx, tr.sy, tr.sz)
        col = int(np.clip((np.arctan2(c_T[1], c_T[0]) + np.pi) / (2 * np.pi) * bl.W, 0, bl.W - 1))
        t_cap = int(t_col[col])
        tc = po.interpolate_tracks(obs[obs.track_id == bx["track_id"]], t_cap, gap)
        if len(tc):
            c_c, a_c = to_t0(tc.iloc[0], t_cap, ego, t0_us)
        else:
            c_c, a_c = c_T, a_T
        bx.update(n_lidar=count_in(pts, c_c, a_c, size), n_lidar_notc=count_in(pts, c_T, a_T, size),
                  tc_ok=bool(len(tc)), lidar_dt_ms=(T - t_cap) / 1e3,
                  shift_m=float(np.linalg.norm(c_c - c_T)))
    info["n_lidar_pts"] = len(pts)
    return boxes, info


def answer_text(boxes, max_boxes):
    """Kept boxes of one sample -> A2-style JSON answer (nearest first, at most max_boxes)."""
    b = boxes.sort_values("dist_m").head(max_boxes)
    items = [{"bbox_2d": [round(float(v)) for v in (r.x1, r.y1, r.x2, r.y2)], "label": NAMES[r.label_class]}
             for r in b.itertuples()]
    return json.dumps(items) if items else "[ ]"  # "[]" is one token outside the student vocab


def overlay(set_name, clip_id, t0_us, boxes, k, path):
    data = dc.load_cached(dc.path_for(clip_id, t0_us, set_name))
    img = data["image_frames"][dc.FRONT_WIDE_SLOT, -1].permute(1, 2, 0).numpy()  # (320, 576, 3)
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.imshow(img)
    for b in boxes.itertuples():
        if not b.keep or b.coarse not in CLASSES:
            continue
        ok = b.n_lidar >= k
        x1, y1, x2, y2 = b.x1 * W / 1000, b.y1 * H / 1000, b.x2 * W / 1000, b.y2 * H / 1000
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.4,
                                   ls="-" if ok else "--", ec="#00c853" if ok else "#ff1744"))
        ax.text(x1, y1 - 2, f"{NAMES[b.label_class]} {b.dist_m:.0f}m n={b.n_lidar}", fontsize=6,
                color="#00c853" if ok else "#ff1744")
    ax.set_title(f"{clip_id[:8]} t0={t0_us / 1e6:.1f}s  solid green = kept (n_lidar >= {k}), "
                 f"dashed red = removed by the LiDAR test", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_labels(set_name, boxes, infos, k, max_boxes):
    import distill_data as dd

    tok = dd.SeqBuilder().tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    vmap = np.load(VOCAB_MAP)
    boxes = boxes.assign(keep_v2=boxes.keep & (boxes.n_lidar >= k) & boxes.coarse.isin(CLASSES))
    groups = {key: g for key, g in boxes[boxes.keep_v2].groupby(["clip_id", "t0_us"])}
    (LABELS / set_name / "det").mkdir(parents=True, exist_ok=True)
    n_box, n_tok, oov, bad = [], [], 0, 0
    for r in infos[infos.status == "ok"].itertuples():
        g = groups.get((r.clip_id, r.t0_us), boxes.iloc[:0])
        text = answer_text(g, max_boxes)
        ids = tok(text, add_special_tokens=False)["input_ids"] + [im_end]
        oov += int((vmap[ids] < 0).sum())
        back = [tuple(b[:4]) for b in parse_boxes(text)[0]]
        want = [tuple(float(v) for v in it["bbox_2d"]) for it in json.loads(text)]
        bad += back != want
        n_box.append(min(len(g), max_boxes) if len(g) else 0)
        n_tok.append(len(ids))
        np.savez(label_path(set_name, r.clip_id, r.t0_us), gen_ids=np.asarray(ids, np.int64),
                 text=json.dumps({"answer": text}), n_boxes_all=np.int32(len(g)))
    n_box, n_tok = np.asarray(n_box), np.asarray(n_tok)
    raw = boxes[boxes.keep_v2].groupby(["clip_id", "t0_us"]).size()
    return boxes, {"k": k, "max_boxes": max_boxes, "labels_written": len(n_box),
                   "boxes_per_sample": {q: float(np.percentile(n_box, q)) for q in (50, 90, 99, 100)},
                   "boxes_per_sample_mean": float(n_box.mean()), "empty_frac": float((n_box == 0).mean()),
                   "capped_frac": float((raw > max_boxes).sum() / max(len(n_box), 1)),
                   "tokens": {q: float(np.percentile(n_tok, q)) for q in (50, 90, 99, 100)},
                   "oov_tokens": oov, "roundtrip_mismatch": int(bad)}


def vis_stats(boxes):
    kb = boxes[boxes.keep & boxes.coarse.isin(CLASSES)]
    out = {"boxes_projected": len(boxes), "boxes_keep_3cls": len(kb), "tc_ok_frac": float(kb.tc_ok.mean())}
    for k in (1, 3, 5):
        s = {"kept": int((kb.n_lidar >= k).sum()), "removed_frac": float((kb.n_lidar < k).mean())}
        for lo, hi in DIST_BINS:
            d = kb[(kb.dist_m >= lo) & (kb.dist_m < hi)]
            s[f"removed_frac_{lo}-{hi if hi < 1e9 else 'inf'}m"] = [float((d.n_lidar < k).mean()), len(d)]
        for c in CLASSES:
            d = kb[kb.coarse == c]
            s[f"removed_frac_{c}"] = [float((d.n_lidar < k).mean()) if len(d) else None, len(d)]
        out[f"k{k}"] = s
    mv = kb[kb.shift_m > 1.0]
    out["moving_shift_gt_1m"] = {"n": len(mv), "removed_frac_k3_tc": float((mv.n_lidar < 3).mean()),
                                 "removed_frac_k3_no_tc": float((mv.n_lidar_notc < 3).mean())}
    out["n_lidar_median"] = float(kb.n_lidar.median())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-boxes", type=int, default=40)
    ap.add_argument("--labels-only", action="store_true", help="reuse <set>.parquet, rewrite labels")
    ap.add_argument("--overlays", type=int, default=0)
    ap.add_argument("--max-gap-ms", type=float, default=150.0)
    ap.add_argument("--min-vis", type=float, default=0.5)
    ap.add_argument("--min-h", type=float, default=12.0, help="px at 320x576")
    ap.add_argument("--max-dist", type=float, default=80.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.labels_only:
        man = pd.read_parquet(REPO / "outputs" / "distill_sets" / f"{args.set}.parquet")
        man = [(r.clip_id, int(r.t0_us)) for r in man.itertuples()
               if dc.path_for(r.clip_id, r.t0_us, args.set).exists()
               and bl.path_for(r.clip_id, r.t0_us, args.set).exists()]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(visibility_sample, [(c, t, args.set, args) for c, t in man], chunksize=4))
        pd.DataFrame([b for bs, _ in res for b in bs]).to_parquet(OUT / f"{args.set}.parquet")
        pd.DataFrame([i for _, i in res]).to_parquet(OUT / f"{args.set}_info.parquet")
    boxes = pd.read_parquet(OUT / f"{args.set}.parquet")
    infos = pd.read_parquet(OUT / f"{args.set}_info.parquet")

    boxes, lab = write_labels(args.set, boxes, infos, args.k, args.max_boxes)
    boxes.to_parquet(OUT / f"{args.set}.parquet")
    stats = {"set": args.set, "samples": len(infos), "status": infos.status.value_counts().to_dict(),
             "lidar_pts_per_sample": float(infos.n_lidar_pts.mean()) if "n_lidar_pts" in infos else None,
             "visibility": vis_stats(boxes), "labels": lab,
             "keep_v2_by_coarse": boxes[boxes.keep_v2].coarse.value_counts().to_dict(),
             "params": {k: getattr(args, k) for k in ("k", "max_boxes", "max_gap_ms", "min_vis", "min_h", "max_dist")}}
    (OUT / f"{args.set}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)

    if args.overlays:
        pdir = OUT / "overlays"
        pdir.mkdir(parents=True, exist_ok=True)
        ok = infos[infos.status == "ok"]
        # samples with at least one box removed by the LiDAR test first, so the plots show them
        rm = boxes[boxes.keep & boxes.coarse.isin(CLASSES) & (boxes.n_lidar < args.k)]
        keys = list(dict.fromkeys(list(zip(rm.clip_id, rm.t0_us)) + list(zip(ok.clip_id, ok.t0_us))))
        for c, t in keys[:args.overlays]:
            overlay(args.set, c, t, boxes[(boxes.clip_id == c) & (boxes.t0_us == t)], args.k,
                    pdir / f"{args.set}_{c[:8]}_{t}.png")
        print(f"overlays -> {pdir}", flush=True)


if __name__ == "__main__":
    main()
