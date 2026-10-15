"""Build the 320x576 six-camera caches (and per-clip obstacle labels) for the distill sets.

Each manifest row (clip_id, t0_us) is loaded through the release loader with a local-first
dataset interface (distill_cache.local_first: files of the /mnt/nvme1n1 mirror are read in
place, the rest streamed, transient Hub errors retried),
the six driving-profile cameras are resized exactly as A2's processor would resize them
(see distill_cache.py), and one npz per sample is written. `obstacle.offline` for the clip is
streamed into labels/obstacle_perclip/ when not there yet (same raw parquet the compression
track's fetch_obstacles.py writes). Resumable: finished samples / clips are skipped.

Usage (after `source experiments/oodval/env.sh`):
  python experiments/distill/build_cache_distill.py --set train_10k --workers 24
  python experiments/distill/build_cache_distill.py --set dev_238
  python experiments/distill/build_cache_distill.py --set val_500
  python experiments/distill/build_cache_distill.py --set ood_262
"""

import argparse
import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import distill_cache as dc

REPO = Path(__file__).resolve().parents[2]
SETS = REPO / "outputs" / "distill_sets"
A2_SNAPSHOT = "/mnt/nvme1n1/ad_vla/cache/hub/models--nvidia--Alpamayo2-Super/snapshots/00554695*/"
MIN_PIXELS, MAX_PIXELS = 163_840, 196_608  # Alpamayo2SuperConfig defaults

_STATE = {}


def _init():
    """Per-process dataset interface and image processor."""
    if not _STATE:
        from transformers import AutoProcessor

        _STATE["avdi"] = dc.make_avdi()
        snap = glob.glob(A2_SNAPSHOT)[0]
        _STATE["ip"] = AutoProcessor.from_pretrained(
            snap, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS).image_processor
    return _STATE["avdi"], _STATE["ip"]


def fetch_obstacles(avdi, clip_id):
    """-> 'ok' | 'skip' | 'absent' | 'error: ...' for labels/obstacle_perclip/<clip>.parquet."""
    p = dc.OBSTACLES / f"{clip_id}.parquet"
    if p.exists():
        return "skip"
    try:
        d = dc.with_retry(avdi.get_clip_feature, clip_id, "obstacle.offline", maybe_stream=True)
    except KeyError:
        return "absent"
    except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
        return f"error: {type(e).__name__}: {str(e)[:120]}"
    if isinstance(d, dict):  # zip features come back keyed by member name
        d = next(iter(d.values()))
    tmp = p.with_suffix(".parquet.tmp")
    d.to_parquet(tmp)
    tmp.rename(p)
    return "ok"


def build_one(task):
    clip_id, t0_us, set_name, want_obstacles = task
    out = dc.path_for(clip_id, t0_us, set_name)
    rec = {"clip_id": clip_id, "t0_us": t0_us}
    avdi, ip = _init()
    if want_obstacles:
        rec["obstacles"] = fetch_obstacles(avdi, clip_id)
    if out.exists():
        rec.update(status="skip", bytes=out.stat().st_size)
        return rec

    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

    t = time.time()
    try:
        d = dc.with_retry(load_physical_aiavdataset, clip_id, t0_us=t0_us, avdi=avdi,
                          maybe_stream=True, camera_features=list(dc.CAMERA_FEATURES),
                          include_calibration=False)
    except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
        rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
        return rec
    t_load = time.time() - t

    frames = d["image_frames"]  # (6, 4, 3, H, W) uint8
    n_cam, n_frame, _, src_h, src_w = frames.shape
    small = dc.resize_like_processor(ip, frames.flatten(0, 1))  # (24, 3, 320, 576) uint8
    small = small.view(n_cam, n_frame, *small.shape[1:]).numpy()
    blobs = [dc.encode_frame(np.transpose(small[c, f], (1, 2, 0)))
             for c in range(n_cam) for f in range(n_frame)]

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            img_bytes=np.array(blobs, dtype=object), codec="webp_lossless",
            n_cam=np.int32(n_cam), n_frame=np.int32(n_frame),
            H=np.int32(small.shape[-2]), W=np.int32(small.shape[-1]),
            src_H=np.int32(src_h), src_W=np.int32(src_w),
            camera_indices=d["camera_indices"].numpy(),  # (6,)
            absolute_timestamps=d["absolute_timestamps"].numpy(),  # (6, 4)
            ego_history_xyz=d["ego_history_xyz"].numpy(),
            ego_history_rot=d["ego_history_rot"].numpy(),
            ego_future_xyz=d["ego_future_xyz"].numpy(),
            ego_future_rot=d["ego_future_rot"].numpy(),
            t0_us=np.int64(t0_us), clip_id=str(clip_id),
        )
    tmp.rename(out)
    rec.update(status="ok", sec=time.time() - t, sec_load=t_load, bytes=out.stat().st_size)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="outputs/distill_sets/<set>.parquet")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="build only the first N (smoke)")
    ap.add_argument("--no-obstacles", action="store_true")
    args = ap.parse_args()

    man = pd.read_parquet(SETS / f"{args.set}.parquet")
    if args.limit:
        man = man.head(args.limit)
    root = dc.ROOT / args.set
    (root / "samples").mkdir(parents=True, exist_ok=True)
    dc.OBSTACLES.mkdir(parents=True, exist_ok=True)
    lidar = man["lidar"] if "lidar" in man else pd.Series(True, index=man.index)
    tasks = [(r.clip_id, int(r.t0_us), args.set, bool(lid) and not args.no_obstacles)
             for r, lid in zip(man.itertuples(), lidar)]

    recs, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 50 == 0 or i == len(tasks):
                el = time.time() - t0
                n_err = sum(r["status"] == "error" for r in recs)
                gb = sum(r.get("bytes", 0) for r in recs) / 2**30
                print(f"[{i}/{len(tasks)}] err={n_err} {gb:.2f} GB  elapsed {el / 60:.1f} min  "
                      f"eta {el / i * (len(tasks) - i) / 60:.1f} min", flush=True)

    df = pd.DataFrame(recs)
    df.to_json(root / "build_log.json", orient="records", indent=1)
    index = sorted(p.stem for p in (root / "samples").glob("*.npz"))
    (root / "index.json").write_text(json.dumps(index))
    ok = df[df.status == "ok"]
    obst = df["obstacles"].value_counts().to_dict() if "obstacles" in df else {}
    total_gb = sum(p.stat().st_size for p in (root / "samples").glob("*.npz")) / 2**30
    lines = [f"distill cache {args.set}: six driving cameras x 4 frames at A2 model-input size",
             (f"manifest rows: {len(man)}   on disk: {len(index)}   "
             f"errors this run: {int((df.status == 'error').sum())}"),
             f"size: {total_gb:.2f} GB ({total_gb * 1024 / max(len(index), 1):.2f} MB/sample)",
             (f"built this run: {len(ok)}, mean {ok.sec.mean() if len(ok) else 0:.1f} s/sample/worker "
             f"(load {ok.sec_load.mean() if len(ok) else 0:.1f} s)"),
             f"obstacle labels: {obst}",
             (f"schema: 24 lossless WebP frames (cams {list(dc.CAMERA_FEATURES)} x frames "
             "t0-0.3..t0) at 320x576")]
    (root / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
