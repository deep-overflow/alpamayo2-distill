"""Per-sample cache of the six driving-profile cameras at Alpamayo 2's model-input size.

A2's processor shrinks every 1080x1920 frame to 320x576 (smart_resize, factor 32,
min/max pixels 163,840 / 196,608) before the ViT sees it, so frames are stored already
at that size: the processor's own bicubic resize on [0, 1] floats, clamped, rounded to
uint8, then stored losslessly (WebP lossless, ~3.5 MB per sample). Gate G0a chose lossless:
JPEG q95 at this size moved the teacher 3.4x as much as the full-resolution JPEG cache did
(criterion 3x), while the lossless uint8 frames moved it less than that cache.

Cameras are the trajectory / meta-action / grounding profile `[0, 1, 2, 3, 5, 6]` in that
order; frames are t0-0.3 s .. t0. The npz layout otherwise matches oodval/cache_a2.py.
"""

import contextlib
import io
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from alpamayo2_super.common.constants import CAMERA_INDICES_TO_NAMES

AV = Path(os.environ.get("AD_VLA_DATA", "/mnt/nvme1n1/ad_vla/data")) / "physicalai_av"
ROOT = AV / "pre_processed" / "distill_a2"
OBSTACLES = AV / "labels" / "obstacle_perclip"
CAMERA_FEATURES = ("camera_cross_left_120fov", "camera_front_wide_120fov",
                   "camera_cross_right_120fov", "camera_rear_left_70fov",
                   "camera_rear_right_70fov", "camera_front_tele_30fov")
FRONT_WIDE_SLOT = 1  # index of camera 1 (front wide) in CAMERA_FEATURES


def local_first(avdi):
    """Make `avdi.open_file` read the local mirror under AV before streaming from the Hub.

    `PhysicalAIAVDatasetInterface(local_dir=...)` only consults the HF cache in `open_file`,
    so without this every feature is streamed even when its chunk sits on /mnt/nvme1n1.
    """
    stream = avdi.open_file

    @contextlib.contextmanager
    def open_file(filename, mode="rb", maybe_stream=False):
        p = AV / filename
        if p.exists():
            with open(p, mode) as f:
                yield f
        else:
            with stream(filename, mode=mode, maybe_stream=maybe_stream) as f:
                yield f

    avdi.open_file = open_file
    return avdi


def make_avdi():
    import physical_ai_av

    return local_first(physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=AV))


def with_retry(fn, *args, tries=3, wait=20, **kwargs):
    """Hub streaming fails transiently under load (FileNotFoundError / timeouts); retry those."""
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except (FileNotFoundError, OSError, ConnectionError):
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def path_for(clip_id, t0_us, set_name):
    return ROOT / set_name / "samples" / f"{clip_id}__t0_{int(t0_us)}.npz"


def encode_frame(img_hwc):
    """(H, W, 3) uint8 -> lossless WebP bytes."""
    buf = io.BytesIO()
    Image.fromarray(img_hwc).save(buf, format="WEBP", lossless=True, quality=50, method=2)
    return buf.getvalue()


def decode_frames(z):
    """img_bytes (lossless WebP; jpeg_bytes in the first q95 build) -> (n_cam, n_frame, 3, H, W)."""
    n_cam, n_frame = int(z["n_cam"]), int(z["n_frame"])
    H, W = int(z["H"]), int(z["W"])
    blobs = z["img_bytes"] if "img_bytes" in z else z["jpeg_bytes"]
    frames = np.empty((n_cam, n_frame, 3, H, W), dtype=np.uint8)
    for c in range(n_cam):
        for f in range(n_frame):
            img = np.asarray(Image.open(io.BytesIO(blobs[c * n_frame + f])).convert("RGB"))
            frames[c, f] = np.transpose(img, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
    return frames


def load_cached(path):
    """Rebuild the dict `load_physical_aiavdataset` returns for the six-camera profile."""
    z = np.load(path, allow_pickle=True)
    camera_indices = torch.from_numpy(z["camera_indices"])  # (6,)
    absolute_timestamps = torch.from_numpy(z["absolute_timestamps"])  # (6, 4)
    camera_tmin = int(absolute_timestamps.min().item())
    t0_us = int(z["t0_us"])
    n_frame = int(z["n_frame"])
    return {
        "image_frames": torch.from_numpy(decode_frames(z)),  # (6, 4, 3, 320, 576) uint8
        "camera_indices": camera_indices,
        "camera_names": [CAMERA_INDICES_TO_NAMES[i] for i in camera_indices.tolist()],
        "absolute_timestamps": absolute_timestamps,
        "relative_timestamps": (absolute_timestamps - camera_tmin).float() * 1e-6,  # (6, 4)
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


def target_size(image_processor, height, width):
    """(h, w) the processor resizes a (height, width) frame to.

    At call time `_further_process_kwargs` replaces `size` with min/max_pixels whenever both
    are set (get_processor sets them), so those win over `image_processor.size`.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    ip = image_processor
    if ip.min_pixels is not None and ip.max_pixels is not None:
        lo, hi = ip.min_pixels, ip.max_pixels
    else:
        lo, hi = ip.size["shortest_edge"], ip.size["longest_edge"]
    return smart_resize(height, width, factor=ip.patch_size * ip.merge_size,
                        min_pixels=lo, max_pixels=hi)


def resize_like_processor(image_processor, frames_u8):
    """(N, 3, H, W) uint8 -> (N, 3, h, w) uint8 through the processor's own resize call."""
    from transformers.image_processing_utils_fast import SizeDict
    from transformers.image_utils import pil_torch_interpolation_mapping

    h, w = target_size(image_processor, frames_u8.shape[-2], frames_u8.shape[-1])
    x = frames_u8.float() / 255.0  # same scaling as helper.prepare_model_inputs
    x = image_processor.resize(image=x, size=SizeDict(height=h, width=w),
                               interpolation=pil_torch_interpolation_mapping[
                                   image_processor.resample])
    return (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
