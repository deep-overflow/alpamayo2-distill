"""Build the 7-camera OOD-val cache Alpamayo 2 Super is evaluated from.

Streams each of the 262 OOD-val clips at its event timestamp through the release
loader (`load_physical_aiavdataset`, 7-camera ring, 4 frames) and writes one npz per
clip. Resumable: a sample already on disk is skipped. See `cache_a2.py` for the
schema and why the existing 4-camera cache cannot serve this model.

Usage:
  python experiments/oodval/build_cache_a2.py --workers 16
  python experiments/oodval/build_cache_a2.py --verify     # ego tensors vs the 4-cam cache
"""

import argparse
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import cache_a2 as ca

EGO_KEYS = ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")


def build_one(task):
    """Load one clip at its event timestamp and write the npz. Returns a status dict."""
    clip_id, t0_us = task
    out = ca.path_for(clip_id, t0_us)
    if out.exists():
        return {"clip_id": clip_id, "status": "skip", "bytes": out.stat().st_size}

    # imported here so each worker process sets up its own dataset interface
    from PIL import Image

    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

    t = time.time()
    try:
        d = load_physical_aiavdataset(clip_id, t0_us=t0_us, maybe_stream=True,
                                      include_calibration=False)
    except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
        return {"clip_id": clip_id, "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:200]}"}

    frames = d["image_frames"].numpy()  # (7, 4, 3, H, W) uint8
    n_cam, n_frame = frames.shape[0], frames.shape[1]
    jpegs = []
    for c in range(n_cam):
        for f in range(n_frame):
            img = np.transpose(frames[c, f], (1, 2, 0))  # (H, W, 3)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=95)
            jpegs.append(buf.getvalue())

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            jpeg_bytes=np.array(jpegs, dtype=object),
            n_cam=np.int32(n_cam), n_frame=np.int32(n_frame),
            H=np.int32(frames.shape[3]), W=np.int32(frames.shape[4]),
            camera_indices=d["camera_indices"].numpy(),  # (7,)
            absolute_timestamps=d["absolute_timestamps"].numpy(),  # (7, 4)
            ego_history_xyz=d["ego_history_xyz"].numpy(),
            ego_history_rot=d["ego_history_rot"].numpy(),
            ego_future_xyz=d["ego_future_xyz"].numpy(),
            ego_future_rot=d["ego_future_rot"].numpy(),
            t0_us=np.int64(t0_us), clip_id=str(clip_id),
        )
    tmp.rename(out)
    return {"clip_id": clip_id, "status": "ok", "sec": time.time() - t,
            "bytes": out.stat().st_size}


def verify(man):
    """Ego tensors must equal the 4-camera cache's: same clip, same t0, same loader math."""
    bad, missing = [], 0
    for r in man.itertuples():
        new, old = ca.path_for(r.clip_id, r.t0_us), ca.path_for(r.clip_id, r.t0_us, ca.OLD_CACHE)
        if not new.exists() or not old.exists():
            missing += 1
            continue
        zn, zo = np.load(new, allow_pickle=True), np.load(old, allow_pickle=True)
        diff = max(float(np.abs(zn[k] - zo[k]).max()) for k in EGO_KEYS)
        if diff != 0.0:
            bad.append((r.clip_id, diff))
    print(f"verify: {len(man) - missing} compared, {missing} missing, "
          f"{len(bad)} with a non-zero ego difference", flush=True)
    for clip_id, diff in bad[:10]:
        print(f"  {clip_id} max|diff|={diff:.3e}", flush=True)
    return not bad and not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="build only the first N (smoke)")
    ap.add_argument("--verify", action="store_true", help="only compare against the 4-cam cache")
    args = ap.parse_args()

    man = pd.read_parquet(ca.MANIFEST)
    if args.limit:
        man = man.head(args.limit)
    if args.verify:
        sys.exit(0 if verify(man) else 1)

    (ca.CACHE / "samples").mkdir(parents=True, exist_ok=True)
    tasks = [(r.clip_id, int(r.t0_us)) for r in man.itertuples()]
    done, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            (errors if r["status"] == "error" else done).append(r)
            if i % 10 == 0 or i == len(tasks):
                el = time.time() - t0
                gb = sum(x.get("bytes", 0) for x in done) / 2**30
                print(f"[{i}/{len(tasks)}] ok={len(done)} err={len(errors)} {gb:.2f} GB  "
                      f"elapsed {el / 60:.1f} min  eta {el / i * (len(tasks) - i) / 60:.1f} min",
                      flush=True)

    index = sorted(p.stem for p in (ca.CACHE / "samples").glob("*.npz"))
    (ca.CACHE / "index.json").write_text(json.dumps(index))
    (ca.CACHE / "errors.json").write_text(json.dumps(errors, indent=2))
    total_gb = sum(p.stat().st_size for p in (ca.CACHE / "samples").glob("*.npz")) / 2**30
    (ca.CACHE / "summary.txt").write_text(
        f"OOD-val sample cache for Alpamayo 2 Super\n"
        f"clips in manifest: {len(man)}\n"
        f"built: {len(index)}   failed: {len(errors)}\n"
        f"size: {total_gb:.2f} GB\n"
        f"schema: 28 jpegs (7 cam x 4 frame, quality 95) + ego + absolute_timestamps\n"
        f"t0_us: per-clip event_start_timestamp, from {ca.MANIFEST}\n")
    print(f"\ndone: {len(index)} samples, {total_gb:.2f} GB, {len(errors)} errors -> {ca.CACHE}",
          flush=True)


if __name__ == "__main__":
    main()
