"""Per-clip cache of the 7-camera source ring Alpamayo 2 Super selects its inputs from.

The compression track's `pre_processed/ood` cache holds the 4 cameras Alpamayo-1.5
consumes (`[0, 1, 2, 6]`). Alpamayo 2's trajectory profile is 6 cameras
(`[0, 1, 2, 3, 5, 6]`), chosen by `select_task_input` out of the canonical 7-camera
ring, so the ring is what gets cached: 28 JPEGs (7 cameras x 4 frames) plus the ego
trajectory and the frame timestamps `select_task_input` validates.

JPEG quality 95 and the npz layout match `build_ood_cache.py`, so both models read
frames through the same round-trip. `clip_seed` is the same sha256 rule as
`sample_cache.clip_seed`.
"""

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from alpamayo2_super.common.constants import CAMERA_INDICES_TO_NAMES

AV = Path(os.environ.get("AD_VLA_DATA", "/mnt/nvme1n1/ad_vla/data")) / "physicalai_av"
CACHE = AV / "pre_processed" / "ood_a2"
OLD_CACHE = AV / "pre_processed" / "ood"  # the 4-camera Alpamayo-1.5 cache
MANIFEST = Path("/mnt/nvme1n1/ad_vla/outputs/chan/eval_sets/ood_val.parquet")


def clip_seed(global_seed, clip_id):
    """Stable per-clip seed. Independent of order, sharding and restarts."""
    h = hashlib.sha256(f"{global_seed}:{clip_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def path_for(clip_id, t0_us, root=CACHE):
    return Path(root) / "samples" / f"{clip_id}__t0_{t0_us}.npz"


def decode_frames(z):
    """jpeg_bytes -> (n_cam, n_frame, 3, H, W) uint8."""
    n_cam, n_frame = int(z["n_cam"]), int(z["n_frame"])
    H, W = int(z["H"]), int(z["W"])
    frames = np.empty((n_cam, n_frame, 3, H, W), dtype=np.uint8)
    for c in range(n_cam):
        for f in range(n_frame):
            img = np.asarray(Image.open(io.BytesIO(z["jpeg_bytes"][c * n_frame + f])))
            frames[c, f] = np.transpose(img, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
    return frames


def load_cached(path):
    """Rebuild the dict `load_physical_aiavdataset` returns, minus calibration."""
    z = np.load(path, allow_pickle=True)
    camera_indices = torch.from_numpy(z["camera_indices"])  # (7,)
    absolute_timestamps = torch.from_numpy(z["absolute_timestamps"])  # (7, 4)
    camera_tmin = int(absolute_timestamps.min().item())
    t0_us = int(z["t0_us"])
    n_frame = int(z["n_frame"])
    return {
        "image_frames": torch.from_numpy(decode_frames(z)),  # (7, 4, 3, H, W) uint8
        "camera_indices": camera_indices,
        "camera_names": [CAMERA_INDICES_TO_NAMES[i] for i in camera_indices.tolist()],
        "absolute_timestamps": absolute_timestamps,
        "relative_timestamps": (absolute_timestamps - camera_tmin).float() * 1e-6,  # (7, 4)
        "camera_tmin": camera_tmin,
        "ego_t0": torch.tensor([t0_us], dtype=torch.int64),
        "ego_t0_relative": torch.tensor([(t0_us - camera_tmin) * 1e-6], dtype=torch.float32),
        "ego_t0_frame_idx": torch.tensor([n_frame - 1], dtype=torch.int64),
        "ego_history_xyz": torch.from_numpy(z["ego_history_xyz"]),  # (1, 1, 16, 3)
        "ego_history_rot": torch.from_numpy(z["ego_history_rot"]),  # (1, 1, 16, 3, 3)
        "ego_future_xyz": torch.from_numpy(z["ego_future_xyz"]),  # (1, 1, 64, 3)
        "ego_future_rot": torch.from_numpy(z["ego_future_rot"]),  # (1, 1, 64, 3, 3)
        "t0_us": t0_us,
        "clip_id": str(z["clip_id"]),
    }
