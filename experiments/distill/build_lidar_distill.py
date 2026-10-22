"""Range-image cache of the top LiDAR for the distill sets (plan 2026-10-20, section 4.1).

Per sample (clip_id, t0_us):
  1. causality: take the last spin that ENDED at or before t0 (the spin nearest t0 runs up to
     +80 ms past it); every point timestamp is asserted <= t0.
  2. sensor -> rig with the clip's sensor_extrinsics, then deskew: each point is moved from the
     rig pose at its own timestamp to the rig frame at t0 (egomotion interpolation, evaluated
     once per distinct firing time) -- the frame the trajectories and the camera t0 frame use.
  3. range image (4, 128, 1024): rows = beam index (model_element[:, 0], 0 = top), columns =
     azimuth in rig@t0 (column c covers [-180 + c * 360 / 1024, ...) degrees, forward = 512);
     channels log1p(range) / log1p(200), intensity / 255, rig z clipped to [-3, 5] / 5, valid
     mask; the nearest return wins a cell.
  4. alongside: dt_col (1024,) = t0 - capture time per azimuth column in ms (median over the
     column's points, empty columns interpolated) for the encoder's temporal positional
     embedding, and beam_elev (128,) = mean elevation per beam in degrees.

The LiDAR is streamed from the Hub: only the zip directory, the parquet footer and the one row
group (one spin, ~1.4 MB) are read; samples of the same chunk share one opened zip.

Usage (after `source experiments/oodval/env.sh`):
  python experiments/distill/build_lidar_distill.py --set val_500 --workers 32
"""

import argparse
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import distill_cache as dc

REPO = Path(__file__).resolve().parents[2]
ROOT = dc.AV / "pre_processed" / "distill_a2_lidar"
HUB = "datasets/nvidia/PhysicalAI-Autonomous-Vehicles/lidar/lidar_top_360fov/"
LIDAR = "lidar_top_360fov"
H, W = 128, 1024
MAX_RANGE = 200.0


def path_for(clip_id, t0_us, set_name):
    return ROOT / set_name / "samples" / f"{clip_id}__t0_{int(t0_us)}.npz"


def load_cached(path):
    z = np.load(path)
    elev = z["beam_elev"].astype(np.float32)
    bad = np.isnan(elev)
    if bad.any():  # a beam without returns in this spin: interpolate over the beam index
        elev[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), elev[~bad])
    return {"range_img": z["range_img"].astype(np.float32), "dt_col": z["dt_col"].astype(np.float32),
            "beam_elev": elev}


def last_spin_before(pf, t0_us):
    md = pf.metadata
    end = np.array([md.row_group(g).column(2).statistics.min for g in range(md.num_row_groups)])
    start = np.array([md.row_group(g).column(1).statistics.min for g in range(md.num_row_groups)])
    ok = np.where(end <= t0_us)[0]
    if not len(ok):
        raise ValueError(f"no LiDAR spin ends before t0={t0_us}")
    g = int(ok.max())
    return g, int(start[g]), int(end[g])


def decode(blob):
    import DracoPy

    pc = DracoPy.decode(blob)
    attrs = {a.get("name"): np.asarray(a["data"]) for a in pc.attributes}
    pts = np.asarray(pc.points, dtype=np.float64)  # (N, 3) sensor frame
    return (pts, attrs["intensity"].reshape(-1).astype(np.float32),
            attrs["timestamp"].reshape(-1).astype(np.int64),
            attrs["model_element"][:, 0].astype(np.int64))


def deskew(rig_pts, ts, ego, t0_us):
    """rig-frame points measured at times ts -> rig frame at t0. (N, 3)"""
    import scipy.spatial.transform as spt

    uniq, inv = np.unique(ts, return_inverse=True)
    st = ego(uniq)  # poses of the rig in the world at each firing time
    ref = ego(np.array([t0_us], dtype=np.int64))
    rot = spt.Rotation.from_quat(st.pose.rotation.as_quat())[inv]
    world = rot.apply(rig_pts) + st.pose.translation[inv]
    r0 = spt.Rotation.from_quat(ref.pose.rotation.as_quat()[0])
    return r0.inv().apply(world - ref.pose.translation[0])


def range_image(pts_s, pts_t0, inten, ts, beam, t0_us):
    rng = np.linalg.norm(pts_s, axis=1)  # measured range from the sensor
    keep = (rng > 0.5) & (rng < MAX_RANGE) & (beam >= 0) & (beam < H)
    pts_s, pts_t0, inten, ts, beam, rng = (x[keep] for x in (pts_s, pts_t0, inten, ts, beam, rng))
    az = np.arctan2(pts_t0[:, 1], pts_t0[:, 0])  # rig@t0, forward = 0
    col = np.clip(((az + np.pi) / (2 * np.pi) * W).astype(np.int64), 0, W - 1)
    order = np.argsort(-rng)  # far first, so the nearest return is written last
    img = np.zeros((4, H, W), np.float32)
    b, c = beam[order], col[order]
    img[0, b, c] = np.log1p(rng[order]) / np.log1p(MAX_RANGE)
    img[1, b, c] = inten[order] / 255.0
    img[2, b, c] = np.clip(pts_t0[order, 2], -3.0, 5.0) / 5.0
    img[3, b, c] = 1.0
    dt = (t0_us - ts) / 1e3  # ms, >= 0
    dt_col = np.full(W, np.nan, np.float32)
    s = pd.Series(dt).groupby(col).median()
    dt_col[s.index.to_numpy()] = s.to_numpy()
    good = ~np.isnan(dt_col)
    dt_col = np.interp(np.arange(W), np.flatnonzero(good), dt_col[good]).astype(np.float32)
    elev = np.degrees(np.arcsin(pts_s[:, 2] / rng))
    e = pd.Series(elev).groupby(beam).mean()
    beam_elev = np.full(H, np.nan, np.float32)
    beam_elev[e.index.to_numpy()] = e.to_numpy()
    return img, dt_col, beam_elev, int(keep.sum())


_AVDI = None


def build_chunk(task):
    """All samples of one chunk with a single opened zip. -> list of status dicts."""
    chunk, items, set_name = task
    global _AVDI
    if _AVDI is None:
        _AVDI = dc.make_avdi()
    import huggingface_hub as hh
    import pyarrow.parquet as pq

    out = []
    todo = [(c, t) for c, t in items if not path_for(c, t, set_name).exists()]
    out += [{"clip_id": c, "t0_us": t, "status": "skip"} for c, t in items
            if path_for(c, t, set_name).exists()]
    if not todo:
        return out
    t_open = time.time()
    try:
        f = dc.with_retry(hh.HfFileSystem().open, HUB + f"{LIDAR}.chunk_{chunk:04d}.zip", "rb",
                          block_size=4 * 2**20)
        zf = zipfile.ZipFile(f)
    except Exception as e:  # noqa: BLE001  per-chunk failures are recorded, not fatal
        return out + [{"clip_id": c, "t0_us": t, "status": "error",
                       "error": f"open: {type(e).__name__}: {str(e)[:150]}"} for c, t in todo]
    sec_open = time.time() - t_open
    for clip_id, t0_us in todo:
        t = time.time()
        try:
            pf = pq.ParquetFile(zf.open(f"{clip_id}.{LIDAR}.parquet"))
            g, s0, s1 = last_spin_before(pf, t0_us)
            blob = pf.read_row_group(g, columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
            pts, inten, ts, beam = decode(blob)
            assert ts.max() <= t0_us, f"future point: {ts.max()} > t0 {t0_us}"
            extr = dc.with_retry(_AVDI.get_clip_feature, clip_id, "sensor_extrinsics", maybe_stream=True)
            ego = dc.with_retry(_AVDI.get_clip_feature, clip_id, "egomotion", maybe_stream=True)
            rig = extr.sensor_poses[LIDAR].apply(pts)
            pts_t0 = deskew(rig, ts, ego, t0_us)
            img, dt_col, beam_elev, n_kept = range_image(pts, pts_t0, inten, ts, beam, t0_us)
            p = path_for(clip_id, t0_us, set_name)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".npz.tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, range_img=img.astype(np.float16), dt_col=dt_col, beam_elev=beam_elev,
                         spin_index=np.int32(g), spin_start=np.int64(s0), spin_end=np.int64(s1),
                         n_points=np.int32(len(pts)), n_kept=np.int32(n_kept),
                         t0_us=np.int64(t0_us), clip_id=clip_id)
            tmp.rename(p)
            out.append({"clip_id": clip_id, "t0_us": t0_us, "status": "ok",
                        "sec": time.time() - t + sec_open / len(todo),
                        "end_minus_t0_ms": (s1 - t0_us) / 1e3, "dt_max_ms": float(dt_col.max()),
                        "valid_frac": float(img[3].mean()), "n_points": len(pts)})
        except Exception as e:  # noqa: BLE001  per-sample failures are recorded, not fatal
            out.append({"clip_id": clip_id, "t0_us": t0_us, "status": "error",
                        "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = pd.read_parquet(REPO / "outputs" / "distill_sets" / f"{args.set}.parquet")
    if args.limit:
        man = man.head(args.limit)
    ci = pd.read_parquet(dc.AV / "clip_index.parquet")
    man = man.assign(chunk=ci.loc[man.clip_id, "chunk"].to_numpy())
    tasks = [(int(ch), [(r.clip_id, int(r.t0_us)) for r in g.itertuples()], args.set)
             for ch, g in man.groupby("chunk")]
    root = ROOT / args.set
    (root / "samples").mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_chunk, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            recs += fut.result()
            if i % 20 == 0 or i == len(tasks):
                el = time.time() - t0
                n_err = sum(r["status"] == "error" for r in recs)
                print(f"[{i}/{len(tasks)} chunks] {len(recs)} samples, err={n_err}  "
                      f"elapsed {el / 60:.1f} min  eta {el / i * (len(tasks) - i) / 60:.1f} min",
                      flush=True)
    df = pd.DataFrame(recs)
    df.to_json(root / "build_log.json", orient="records", indent=1)
    ok = df[df.status == "ok"]
    n_disk = len(list((root / "samples").glob("*.npz")))
    gb = sum(p.stat().st_size for p in (root / "samples").glob("*.npz")) / 2**30
    stats = {"set": args.set, "manifest": len(man), "on_disk": n_disk,
             "errors": int((df.status == "error").sum()),
             "built_this_run": len(ok), "size_gb": round(gb, 2),
             **({"spin_end_minus_t0_ms": ok.end_minus_t0_ms.describe().round(1).to_dict(),
                 "dt_max_ms": ok.dt_max_ms.describe().round(1).to_dict(),
                 "valid_frac_mean": round(float(ok.valid_frac.mean()), 4),
                 "points_per_spin_mean": int(ok.n_points.mean()),
                 "sec_per_sample_worker": round(float(ok.sec.mean()), 2)} if len(ok) else {})}
    (root / "summary.json").write_text(json.dumps(stats, indent=2))
    (root / "index.json").write_text(json.dumps(sorted(p.stem for p in (root / "samples").glob("*.npz"))))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
