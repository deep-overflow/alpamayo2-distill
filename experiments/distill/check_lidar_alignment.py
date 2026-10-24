"""Gate L0a (ii): does the deskew put the LiDAR where the rest of the t0 sample is?

For clips with obstacle labels, the spin the cache uses is fetched again and kept two ways:
  raw     sensor -> rig only (each point left where the rig was at its own capture time)
  deskew  as in the cache: moved to the rig frame at t0 (build_lidar_distill.deskew)
Checks:
  in-box  points inside the obstacle cuboids interpolated to t0 (project_obstacles; 0.2 m
          margin), summed over clips -- the deskewed cloud should fill the boxes better
  overlay both clouds projected on the front-wide t0 frame of the distill cache, coloured by
          range, for visual inspection

Usage (after `source experiments/oodval/env.sh`):
  python experiments/distill/check_lidar_alignment.py --set dev_238 --n 12 --plots 4
"""

import argparse
import json
import sys
import zipfile
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

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_lidar_l0a"


def in_boxes(pts, tracks, margin=0.2):
    n = 0
    for r in tracks.itertuples():
        rot = spt.Rotation.from_quat([r.qx, r.qy, r.qz, r.qw])
        local = rot.inv().apply(pts - np.array([r.cx, r.cy, r.cz]))
        half = np.array([r.sx, r.sy, r.sz]) / 2 + margin
        n += int((np.abs(local) <= half).all(axis=1).sum())
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev_238")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--plots", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import huggingface_hub as hh
    import pyarrow.parquet as pq

    avdi = dc.make_avdi()
    man = pd.read_parquet(REPO / "outputs" / "distill_sets" / f"{args.set}.parquet")
    man = [r for r in man.itertuples() if (dc.OBSTACLES / f"{r.clip_id}.parquet").exists()
           and bl.path_for(r.clip_id, r.t0_us, args.set).exists()][:args.n]
    fs = hh.HfFileSystem()
    rows = []
    for j, r in enumerate(man):
        clip, t0 = r.clip_id, int(r.t0_us)
        chunk = int(avdi.clip_index.loc[clip, "chunk"])
        with (fs.open(bl.HUB + f"{bl.LIDAR}.chunk_{chunk:04d}.zip", "rb", block_size=4 * 2**20) as f,
              zipfile.ZipFile(f) as zf, zf.open(f"{clip}.{bl.LIDAR}.parquet") as member):
            pf = pq.ParquetFile(member)
            g, _, _ = bl.last_spin_before(pf, t0)
            blob = pf.read_row_group(g, columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
        pts, _, ts, _ = bl.decode(blob)
        extr = avdi.get_clip_feature(clip, "sensor_extrinsics", maybe_stream=True)
        intr = avdi.get_clip_feature(clip, "camera_intrinsics", maybe_stream=True)
        ego = avdi.get_clip_feature(clip, "egomotion", maybe_stream=True)
        raw = extr.sensor_poses[bl.LIDAR].apply(pts)
        desk = bl.deskew(raw, ts, ego, t0)
        tracks = po.interpolate_tracks(pd.read_parquet(dc.OBSTACLES / f"{clip}.parquet"), t0, 150_000)
        speed = float(np.linalg.norm(ego(np.array([t0])).velocity[0])) if hasattr(ego(np.array([t0])), "velocity") else float("nan")
        row = {"clip_id": clip, "n_tracks": len(tracks), "ego_speed": speed,
               "inbox_raw": in_boxes(raw, tracks), "inbox_deskew": in_boxes(desk, tracks),
               "shift_p50_m": float(np.median(np.linalg.norm(desk - raw, axis=1))),
               "shift_max_m": float(np.linalg.norm(desk - raw, axis=1).max())}
        rows.append(row)
        print(row, flush=True)
        if j < args.plots:
            data = dc.load_cached(dc.path_for(clip, t0, args.set))
            img = data["image_frames"][dc.FRONT_WIDE_SLOT, -1].permute(1, 2, 0).numpy()
            cam, pose = intr.camera_models[po.CAM], extr.sensor_poses[po.CAM]
            fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
            for ax, (name, cloud) in zip(axes, (("raw (no deskew)", raw), ("deskew to t0", desk))):
                c = pose.inv().apply(cloud)
                ok = c[:, 2] > 1.0
                px = cam.ray2pixel(c[ok])
                sx, sy = img.shape[1] / cam.width, img.shape[0] / cam.height
                inside = (px[:, 0] >= 0) & (px[:, 0] < cam.width) & (px[:, 1] >= 0) & (px[:, 1] < cam.height)
                ax.imshow(img)
                ax.scatter(px[inside, 0] * sx, px[inside, 1] * sy, s=0.3,
                           c=np.linalg.norm(c[ok][inside], axis=1), cmap="turbo", vmin=0, vmax=60)
                ax.set_title(f"{clip[:8]} {name}", fontsize=9)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(OUT / f"overlay_{clip[:8]}.png", dpi=110)
            plt.close(fig)
    df = pd.DataFrame(rows)
    res = {"n_clips": len(df), "inbox_raw": int(df.inbox_raw.sum()), "inbox_deskew": int(df.inbox_deskew.sum()),
           "inbox_gain": float(df.inbox_deskew.sum() / max(df.inbox_raw.sum(), 1)),
           "clips_deskew_ge_raw": int((df.inbox_deskew >= df.inbox_raw).sum()),
           "shift_p50_m_median": float(df.shift_p50_m.median()), "shift_max_m_max": float(df.shift_max_m.max())}
    res["pass"] = bool(res["inbox_deskew"] >= res["inbox_raw"])
    (OUT / "alignment_metrics.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
