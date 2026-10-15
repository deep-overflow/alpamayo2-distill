"""Draw the clip sets the pruned+distilled Alpamayo 2 student is trained and evaluated on.

Train: 10k distinct official-train clips with LiDAR (the same clips that carry
`obstacle.offline` 3D labels), one t0 each, bucket-balanced over {turn_left, turn_right,
decel_stop, accel, cruise}, at most `--chunk-cap` clips per chunk. Candidates are up to
`--cand-per-chunk` random clips from every eligible chunk; each candidate is bucketed at
every t0 of the 2..13 s grid (1 s, the grid the 1.5 recovery pool used) from one
egomotion read. The draw is the scarcity-first balanced draw of
alpamayo-model-compression/experiments/recovery/make_train_set.py plus the chunk cap;
`bucket5` / `ego_future_xy` are ported from there unchanged.

Dev (checkpoint selection): recovery_sets/val_official_238 (official val at t0 = 5.1 s,
disjoint from val_500 and the OOD manifest by construction).
Report: eval_sets/val_500 (t0 = 5.1 s) and eval_sets/ood_val (262 clips).
Also exported: eval_sets/calib_100 (detection probe G0d).
Excluded from train: calib_100 (pruning calibration + detection probe) and every clip of
the OOD manifest.

Writes outputs/distill_sets/{train_10k,dev_238,val_500,ood_262,calib_100}.parquet, bucket_pool.parquet
(resumable cache of the egomotion pass), config.json, summary.txt, plots/*.png.

Usage (after `source experiments/oodval/env.sh`):
  python experiments/distill/make_distill_sets.py --workers 32
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
SETS = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
OUT = REPO / "outputs" / "distill_sets"
BUCKETS = ("turn_left", "turn_right", "decel_stop", "accel", "cruise")
T0_GRID_US = [s * 1_000_000 for s in range(2, 14)]
DT = 0.1
SEED = 20261015
A2_CAMERAS = ("camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov",
              "camera_rear_left_70fov", "camera_rear_right_70fov", "camera_front_tele_30fov")
REQUIRED = ("lidar_top_360fov", "obstacle.offline", "egomotion", "camera_intrinsics",
            "sensor_extrinsics", *A2_CAMERAS)

C1, C2 = "#2a78d6", "#eda100"


def bucket5(xy):
    """eval_lib.bucket with the turn split by sign. xy (T, 2) in the ego frame at t0."""
    vel = np.diff(xy, axis=0) / DT  # (T-1, 2)
    speed = np.linalg.norm(vel, axis=1)  # (T-1,)
    moving = speed > 0.5
    if moving.sum() >= 2:
        head = np.arctan2(vel[moving, 1], vel[moving, 0])
        signed = float(np.rad2deg(np.angle(np.exp(1j * (head[-1] - head[0])))))
    else:
        signed = 0.0
    k = max(len(speed) // 8, 1)
    v0, v_end = float(speed[:k].mean()), float(speed[-k:].mean())
    if v_end < 0.5 or v_end < 0.5 * v0:
        b = "decel_stop"
    elif abs(signed) >= 30.0:
        b = "turn_left" if signed > 0 else "turn_right"
    elif v_end > 1.5 * max(v0, 0.5):
        b = "accel"
    else:
        b = "cruise"
    return b, signed, v0, v_end


_AVDI = None


def _avdi():
    """Local-first interface: files on /mnt/nvme1n1 are read in place, the rest streamed."""
    global _AVDI
    if _AVDI is None:
        import physical_ai_av
        _AVDI = physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=AV)
    return _AVDI


def ego_future_xy(egomotion, t0_us):
    """Future 6.4 s xy in the ego frame at t0, from the egomotion interpolator. (64, 2)"""
    import scipy.spatial.transform as spt
    fut_ts = t0_us + np.arange(1, 65, dtype=np.int64) * 100_000
    ref = egomotion(np.array([t0_us], dtype=np.int64))
    fut = egomotion(fut_ts)
    rot_inv = spt.Rotation.from_quat(ref.pose.rotation.as_quat()[0]).inv()
    local = rot_inv.apply(fut.pose.translation - ref.pose.translation[0])  # (64, 3)
    return local[:, :2]


def bucket_clip(clip_id):
    """Bucket every grid t0 of one clip from a single egomotion read."""
    try:
        ego = _avdi().get_clip_feature(clip_id, "egomotion", maybe_stream=True)
    except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
        return [{"clip_id": clip_id, "t0_us": -1, "error": f"{type(e).__name__}: {str(e)[:120]}"}]
    rows = []
    for t in T0_GRID_US:
        try:
            b, signed, v0, v_end = bucket5(ego_future_xy(ego, t))
            rows.append({"clip_id": clip_id, "t0_us": t, "bucket": b,
                         "turn_deg": signed, "v0": v0, "v_end": v_end})
        except Exception as e:  # noqa: BLE001  t0 outside the clip's egomotion span
            rows.append({"clip_id": clip_id, "t0_us": t, "error": f"{type(e).__name__}: {str(e)[:120]}"})
    return rows


def bucket_pool(clips, workers, path):
    """Bucket all candidate clips; rows are cached in `path` so reruns only add new clips."""
    pool = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["clip_id"])
    if "t0_us" in pool:  # a failed egomotion read (t0_us == -1) is retried, not kept
        pool = pool[pool.t0_us != -1]
    todo = sorted(set(clips) - set(pool.clip_id))
    if not todo:
        return pool
    print(f"bucketing {len(todo)} clips x {len(T0_GRID_US)} t0 ...", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(bucket_clip, c) for c in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rows += fut.result()
            if i % 500 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] elapsed {el / 60:.1f}m "
                      f"eta {el / i * (len(todo) - i) / 60:.1f}m", flush=True)
                if i % 5000 == 0:
                    pd.concat([pool, pd.DataFrame(rows)], ignore_index=True).to_parquet(path)
    pool = pd.concat([pool, pd.DataFrame(rows)], ignore_index=True)
    pool.to_parquet(path)
    return pool


def balanced_draw(pool, total, rng, chunk_of, chunk_cap):
    """Scarcity-first balanced draw, one sample per clip, at most `chunk_cap` clips per chunk."""
    order = sorted(BUCKETS, key=lambda b: pool[pool.bucket == b].clip_id.nunique())
    base, rem = divmod(total, len(BUCKETS))
    target = {b: base + (1 if i < rem else 0) for i, b in enumerate(order)}
    used, per_chunk, picks = set(), {}, []

    def draw_from(b, n):
        cand = pool[(pool.bucket == b) & ~pool.clip_id.isin(used)]
        clips = np.asarray(cand.clip_id.unique(), dtype=object)
        rng.shuffle(clips)
        got = 0
        for c in clips:
            if got == n:
                break
            ch = chunk_of[c]
            if per_chunk.get(ch, 0) >= chunk_cap:
                continue
            rows = cand[cand.clip_id == c]
            picks.append(rows.iloc[rng.integers(len(rows))])
            used.add(c)
            per_chunk[ch] = per_chunk.get(ch, 0) + 1
            got += 1
        return got

    for b in order:
        got = draw_from(b, target[b])
        if got < target[b]:
            print(f"bucket {b}: only {got}/{target[b]} clips available", flush=True)
    while len(picks) < total:
        spare = [(pool[(pool.bucket == b) & ~pool.clip_id.isin(used)].clip_id.nunique(), b)
                 for b in BUCKETS]
        spare = [(n, b) for n, b in sorted(spare, reverse=True) if n > 0]
        if not spare or not any(draw_from(b, 1) for _, b in spare):
            break
    return pd.DataFrame(picks).reset_index(drop=True)


def bar_plot(dfs, labels, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.8 / len(dfs)
    for i, (df, lab) in enumerate(zip(dfs, labels)):
        frac = [float((df.bucket == b).mean()) for b in BUCKETS]
        ax.bar(np.arange(len(BUCKETS)) + i * w, frac, width=w, color=[C1, C2][i % 2], label=lab)
    ax.set_xticks(np.arange(len(BUCKETS)) + w * (len(dfs) - 1) / 2)
    ax.set_xticklabels(BUCKETS)
    ax.set_ylabel("share of samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hist_plot(values, xlabel, path, bins):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.hist(values, bins=bins, color=C1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def flags(meta, clip_ids):
    m = meta.loc[list(clip_ids)]
    return {"lidar": int(m.lidar_top_360fov.sum()), "all_required": int(m[list(REQUIRED)].all(axis=1).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--cand-per-chunk", type=int, default=16)
    ap.add_argument("--chunk-cap", type=int, default=10)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    fp = pd.read_parquet(AV / "metadata" / "feature_presence.parquet")
    ci = pd.read_parquet(AV / "clip_index.parquet")
    dc = pd.read_parquet(AV / "metadata" / "data_collection.parquet")
    meta = fp.join(ci).join(dc)

    calib = set(pd.read_parquet(SETS / "eval_sets" / "calib_100.parquet").clip_id)
    ood_manifest = set(pd.read_parquet(SETS / "eval_sets" / "ood.parquet").clip_id)
    elig = meta[(meta.split == "train") & meta.clip_is_valid & meta[list(REQUIRED)].all(axis=1)]
    n_elig0 = len(elig)
    elig = elig[~elig.index.isin(calib) & ~elig.index.isin(ood_manifest)]
    print(f"eligible train clips: {n_elig0} -> {len(elig)} after calib/OOD exclusion "
          f"({elig.chunk.nunique()} chunks)", flush=True)

    # candidates: up to cand_per_chunk per chunk, fixed by the seed
    cand = []
    for _, g in elig.groupby("chunk"):
        ids = np.asarray(sorted(g.index), dtype=object)
        rng.shuffle(ids)
        cand += list(ids[:args.cand_per_chunk])
    print(f"candidates: {len(cand)} clips", flush=True)

    pool = bucket_pool(cand, args.workers, OUT / "bucket_pool.parquet")
    pool = pool[pool.clip_id.isin(cand)]
    errs = pool[pool["error"].notna()] if "error" in pool else pool.iloc[:0]
    # completion order of the egomotion pass is not reproducible; the draw must not see it
    ok = pool[pool.bucket.notna()].sort_values(["clip_id", "t0_us"]).reset_index(drop=True)
    print(f"pool: {len(ok)} samples over {ok.clip_id.nunique()} clips, {len(errs)} error rows",
          flush=True)

    chunk_of = elig.chunk.to_dict()
    train = balanced_draw(ok, args.n_train, rng, chunk_of, args.chunk_cap)
    train["t0_us"] = train.t0_us.astype(np.int64)
    train = train.drop(columns=[c for c in ("error",) if c in train])
    train = train.join(meta[["chunk", "country", "platform_class", "hour_of_day"]], on="clip_id")
    assert train.clip_id.is_unique
    train.to_parquet(OUT / f"train_{len(train) // 1000}k.parquet")

    dev = pd.read_parquet(SETS / "recovery_sets" / "val_official_238.parquet")
    val500 = pd.read_parquet(SETS / "eval_sets" / "val_500.parquet")
    ood = pd.read_parquet(SETS / "eval_sets" / "ood_val.parquet")
    calib_df = pd.read_parquet(SETS / "eval_sets" / "calib_100.parquet")
    for name, df in (("dev_238", dev), ("val_500", val500), ("ood_262", ood),
                     ("calib_100", calib_df)):
        df = df.copy()
        df["lidar"] = meta.loc[df.clip_id, "lidar_top_360fov"].to_numpy()
        df["chunk"] = meta.loc[df.clip_id, "chunk"].to_numpy()
        df.to_parquet(OUT / f"{name}.parquet")
    overlap = {
        "train&dev": len(set(train.clip_id) & set(dev.clip_id)),
        "train&val500": len(set(train.clip_id) & set(val500.clip_id)),
        "train&ood": len(set(train.clip_id) & set(ood.clip_id)),
        "train&calib": len(set(train.clip_id) & calib),
        "dev&val500": len(set(dev.clip_id) & set(val500.clip_id)),
        "dev&ood": len(set(dev.clip_id) & set(ood.clip_id)),
    }
    assert sum(overlap.values()) == 0, overlap

    bar_plot([ok, train], ["pool (all grid t0)", "selected"], OUT / "plots" / "train_buckets.png")
    per_chunk = train.groupby("chunk").size()
    hist_plot(per_chunk.to_numpy(), "selected clips per chunk", OUT / "plots" / "clips_per_chunk.png",
              bins=np.arange(0.5, args.chunk_cap + 1.5))
    hist_plot(train.t0_us.to_numpy() / 1e6, "t0 (s)", OUT / "plots" / "t0_hist.png",
              bins=np.arange(1.5, 14.5))

    cfg = {"seed": SEED, "n_train": args.n_train, "cand_per_chunk": args.cand_per_chunk,
           "chunk_cap": args.chunk_cap, "t0_grid_s": [t / 1e6 for t in T0_GRID_US],
           "required_features": list(REQUIRED),
           "eligible": {"train_all_required": n_elig0, "after_exclusion": len(elig),
                        "chunks": int(elig.chunk.nunique())},
           "excluded": {"calib_100": len(calib), "ood_manifest": len(ood_manifest)},
           "candidates": len(cand), "pool_samples": len(ok), "bucket_error_rows": len(errs),
           "overlap": overlap}
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))

    lines = [f"distill sets (seed {SEED})",
             (f"train: {len(train)} clips (one t0 each) from {train.chunk.nunique()} chunks, "
             f"max {per_chunk.max()} per chunk"),
             "  " + "  ".join(f"{b}={int((train.bucket == b).sum())}" for b in BUCKETS),
             "  countries: " + ", ".join(f"{k}={v}" for k, v in train.country.value_counts().items()),
             "  platforms: " + ", ".join(f"{k}={v}" for k, v in
                                          train.platform_class.value_counts().items()),
             f"  lidar/all-required: {flags(meta, train.clip_id)}",
             f"dev_238: {len(dev)} clips, {flags(meta, dev.clip_id)}",
             f"val_500: {len(val500)} clips, {flags(meta, val500.clip_id)}",
             f"ood_262: {len(ood)} clips, {flags(meta, ood.clip_id)}",
             (f"calib_100 (detection probe / pruning calib): {len(calib_df)} clips, "
             f"{flags(meta, calib_df.clip_id)}"),
             f"pool: {len(ok)} samples ({ok.clip_id.nunique()} clips), error rows {len(errs)}",
             f"overlaps: {overlap}"]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
