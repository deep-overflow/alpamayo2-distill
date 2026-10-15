"""Reference 2D boxes on the front-wide t0 frame from the `obstacle.offline` 3D labels.

For every cached sample: take the front-wide frame closest to t0 (its own timestamp T from
the cache), interpolate each track's rig-frame cuboid to T (linear centre, nearest size and
orientation, only when the track has samples within `--max-gap-ms` on both sides or one
within half of it), move the 8 corners into the camera frame with the clip's sensor
extrinsics (`sensor_pose.inv()`, as A2's own overlays do) and project them with the f-theta
model. A box is the image-clipped hull of the projected corners in front of the camera and
inside the calibrated ray-angle domain.

Kept when: centre depth > 1 m, >= 4 corners projectable, >= `--min-vis` of the unclipped
box inside the image, clipped height >= `--min-h` px at the 320x576 model-input size, rig-
plane distance <= `--max-dist` m. No occlusion test (no LiDAR visibility): hidden objects
stay in, so recall against these boxes is a lower bound.

Boxes are written in A2/Qwen grounding units: x, y in [0, 1000] of the image width/height.

Usage (after `source experiments/oodval/env.sh`):
  python experiments/distill/project_obstacles.py --set calib_100 --overlays 20
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

import distill_cache as dc

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_refboxes"
CAM = "camera_front_wide_120fov"
COARSE = {"automobile": "vehicle", "heavy_truck": "vehicle", "bus": "vehicle",
          "other_vehicle": "vehicle", "trailer": "vehicle", "train_or_tram_car": "vehicle",
          "person": "pedestrian", "stroller": "pedestrian", "rider": "cyclist",
          "protruding_object": "other", "animal": "other"}
COLORS = {"vehicle": "#2a78d6", "pedestrian": "#e8452c", "cyclist": "#008300", "other": "#eda100"}
CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) / 2

_AVDI = None


def _avdi():
    global _AVDI
    if _AVDI is None:
        _AVDI = dc.make_avdi()
    return _AVDI


def interpolate_tracks(obs, T, max_gap_us):
    """-> DataFrame of one row per track at time T (rig frame), with `dt_us` to nearest sample."""
    rows = []
    for tid, g in obs.groupby("track_id"):
        ts = g.timestamp_us.to_numpy(np.int64)
        order = np.argsort(ts)
        g, ts = g.iloc[order], ts[order]
        i = np.searchsorted(ts, T)
        before = i - 1 if i > 0 else None
        after = i if i < len(ts) else None
        if before is not None and after is not None and ts[after] - ts[before] <= 2 * max_gap_us \
                and T - ts[before] <= max_gap_us and ts[after] - T <= max_gap_us:
            a, b = g.iloc[before], g.iloc[after]
            w = (T - ts[before]) / max(ts[after] - ts[before], 1)
            ctr = (1 - w) * a[["center_x", "center_y", "center_z"]].to_numpy(float) \
                + w * b[["center_x", "center_y", "center_z"]].to_numpy(float)
            near = a if w < 0.5 else b
            dt = min(T - ts[before], ts[after] - T)
        else:
            cand = [j for j in (before, after) if j is not None and abs(ts[j] - T) <= max_gap_us / 2]
            if not cand:
                continue
            j = min(cand, key=lambda j: abs(ts[j] - T))
            near = g.iloc[j]
            ctr = near[["center_x", "center_y", "center_z"]].to_numpy(float)
            dt = abs(ts[j] - T)
        rows.append({"track_id": tid, "label_class": near.label_class,
                     "cx": ctr[0], "cy": ctr[1], "cz": ctr[2],
                     "sx": near.size_x, "sy": near.size_y, "sz": near.size_z,
                     "qx": near.orientation_x, "qy": near.orientation_y,
                     "qz": near.orientation_z, "qw": near.orientation_w, "dt_us": int(dt)})
    return pd.DataFrame(rows)


def max_ray_angle(model):
    """Largest ray angle the calibration covers: the angle of the farthest image corner."""
    corners = np.array([[0, 0], [model.width - 1, 0], [0, model.height - 1],
                        [model.width - 1, model.height - 1]], dtype=float)
    rays = model.pixel2ray(corners)
    return float(np.arccos(np.clip(rays[:, 2], -1, 1)).max())


def project_sample(task):
    """One cached sample -> (list of box dicts, info dict)."""
    clip_id, t0_us, set_name, args = task
    info = {"clip_id": clip_id, "t0_us": t0_us}
    z = np.load(dc.path_for(clip_id, t0_us, set_name), allow_pickle=True)
    cams = z["camera_indices"].tolist()
    T = int(z["absolute_timestamps"][cams.index(1), -1])  # front-wide frame at t0
    info["frame_minus_t0_ms"] = (T - t0_us) / 1e3
    H = int(z["H"])
    p = dc.OBSTACLES / f"{clip_id}.parquet"
    if not p.exists():
        info["status"] = "no_obstacles"
        return [], info
    try:
        intr = dc.with_retry(_avdi().get_clip_feature, clip_id, "camera_intrinsics", maybe_stream=True)
        extr = dc.with_retry(_avdi().get_clip_feature, clip_id, "sensor_extrinsics", maybe_stream=True)
    except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
        info["status"] = f"calib_error: {type(e).__name__}: {str(e)[:100]}"
        return [], info
    model, pose = intr.camera_models[CAM], extr.sensor_poses[CAM]
    assert (model.width, model.height) == (int(z["src_W"]), int(z["src_H"])), \
        (model.width, model.height, int(z["src_W"]), int(z["src_H"]))
    th_max = max_ray_angle(model)

    tracks = interpolate_tracks(pd.read_parquet(p), T, args.max_gap_ms * 1000)
    info.update(status="ok", n_tracks=len(tracks))
    boxes = []
    for r in tracks.itertuples():
        ctr = np.array([r.cx, r.cy, r.cz])
        rot = spt.Rotation.from_quat([r.qx, r.qy, r.qz, r.qw])
        corners = ctr + rot.apply(CORNER_SIGNS * np.array([r.sx, r.sy, r.sz]))  # (8, 3) rig
        cam_c = pose.inv().apply(corners)  # (8, 3) camera: z forward, x right, y down
        cam_ctr = pose.inv().apply(ctr)
        if cam_ctr[2] <= 1.0:
            continue
        ang = np.arccos(np.clip(cam_c[:, 2] / np.linalg.norm(cam_c, axis=1), -1, 1))
        ok = (cam_c[:, 2] > 0.1) & (ang < th_max)
        if ok.sum() < 4:
            continue
        px = model.ray2pixel(cam_c[ok])  # (n, 2) native pixels
        x1, y1 = px.min(0)
        x2, y2 = px.max(0)
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        cx1, cy1 = np.clip(x1, 0, model.width), np.clip(y1, 0, model.height)
        cx2, cy2 = np.clip(x2, 0, model.width), np.clip(y2, 0, model.height)
        carea = max(cx2 - cx1, 0) * max(cy2 - cy1, 0)
        vis = carea / area if area > 0 else 0.0
        h_small = (cy2 - cy1) * H / model.height
        dist = float(np.hypot(r.cx, r.cy))
        boxes.append({
            "clip_id": clip_id, "t0_us": t0_us, "track_id": r.track_id,
            "label_class": r.label_class, "coarse": COARSE.get(r.label_class, "other"),
            "x1": cx1 / model.width * 1000, "y1": cy1 / model.height * 1000,
            "x2": cx2 / model.width * 1000, "y2": cy2 / model.height * 1000,
            "dist_m": dist, "depth_m": float(cam_ctr[2]), "vis_frac": float(vis),
            "h_px_small": float(h_small), "n_corners": int(ok.sum()), "dt_ms": r.dt_us / 1e3,
            "keep": bool(vis >= args.min_vis and h_small >= args.min_h and dist <= args.max_dist),
        })
    info["n_kept"] = sum(b["keep"] for b in boxes)
    return boxes, info


def overlay(set_name, clip_id, t0_us, boxes, path):
    data = dc.load_cached(dc.path_for(clip_id, t0_us, set_name))
    img = data["image_frames"][dc.FRONT_WIDE_SLOT, -1].permute(1, 2, 0).numpy()  # (320, 576, 3)
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.imshow(img)
    for b in boxes:
        x1, y1, x2, y2 = (b["x1"] * W / 1000, b["y1"] * H / 1000, b["x2"] * W / 1000,
                          b["y2"] * H / 1000)
        ls = "-" if b["keep"] else ":"
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.4, ls=ls,
                                   ec=COLORS[b["coarse"]]))
        if b["keep"]:
            ax.text(x1, y1 - 2, f"{b['label_class']} {b['dist_m']:.0f}m", fontsize=6,
                    color=COLORS[b["coarse"]])
    ax.set_title(f"{clip_id[:8]} t0={t0_us / 1e6:.1f}s  solid = kept, dotted = filtered", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-gap-ms", type=float, default=150.0)
    ap.add_argument("--min-vis", type=float, default=0.5)
    ap.add_argument("--min-h", type=float, default=12.0, help="px at 320x576")
    ap.add_argument("--max-dist", type=float, default=80.0)
    ap.add_argument("--overlays", type=int, default=0)
    args = ap.parse_args()

    man = pd.read_parquet(REPO / "outputs" / "distill_sets" / f"{args.set}.parquet")
    man = [(r.clip_id, int(r.t0_us)) for r in man.itertuples()
           if dc.path_for(r.clip_id, r.t0_us, args.set).exists()]
    tasks = [(c, t, args.set, args) for c, t in man]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(project_sample, tasks, chunksize=4))
    boxes = pd.DataFrame([b for bs, _ in res for b in bs])
    infos = pd.DataFrame([i for _, i in res])
    OUT.mkdir(parents=True, exist_ok=True)
    boxes.to_parquet(OUT / f"{args.set}.parquet")
    infos.to_parquet(OUT / f"{args.set}_info.parquet")

    kept = boxes[boxes.keep] if len(boxes) else boxes
    ok = infos[infos.status == "ok"]
    stats = {
        "set": args.set, "samples": len(infos), "samples_ok": len(ok),
        "status": infos.status.value_counts().to_dict(),
        "boxes_projected": len(boxes), "boxes_kept": len(kept),
        "kept_per_sample_mean": float(len(kept) / max(len(ok), 1)),
        "kept_by_coarse": kept.coarse.value_counts().to_dict() if len(kept) else {},
        "dt_ms_kept": {q: float(kept.dt_ms.quantile(q)) for q in (0.5, 0.9, 0.99)} if len(kept) else {},
        "frame_minus_t0_ms": {q: float(ok.frame_minus_t0_ms.quantile(q)) for q in (0.0, 0.5, 1.0)},
        "params": {k: getattr(args, k) for k in ("max_gap_ms", "min_vis", "min_h", "max_dist")},
    }
    (OUT / f"{args.set}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)

    if args.overlays:
        pdir = REPO / "outputs" / "distill_g0c" / "plots"
        pdir.mkdir(parents=True, exist_ok=True)
        for c, t in man[:args.overlays]:
            bs = boxes[(boxes.clip_id == c) & (boxes.t0_us == t)].to_dict("records") if len(boxes) else []
            overlay(args.set, c, t, bs, pdir / f"{args.set}_{c[:8]}_{t}.png")
        print(f"overlays -> {pdir}", flush=True)


if __name__ == "__main__":
    main()
