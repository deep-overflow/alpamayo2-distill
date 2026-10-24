"""Slide figure: one cached LiDAR range image (the student's LiDAR input) under the front-wide
camera frame of the same sample.

The camera frame sits above the azimuth span it covers (about +-60 deg around forward), then
three of the four input channels (range, intensity, height; the fourth, the valid mask, shows as
the grey background) and the per-column time offset from t0 that the temporal position
embedding receives.

Usage:
  python experiments/distill/plot_range_image.py --set dev_238 \
      --clip 0c20d1e4-68ec-46f0-ae33-093908efe6af --t0 5100000
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import build_lidar_distill as bl
import distill_cache as dc

REPO = Path(__file__).resolve().parents[2]
INK, MUTED, EMPTY, LINE = "#1b1d21", "#5b626d", "#e6e8eb", "#2a78d6"
CAM_HALF_FOV = 60.0  # deg, front-wide camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev_238")
    ap.add_argument("--clip", default="0c20d1e4-68ec-46f0-ae33-093908efe6af")
    ap.add_argument("--t0", type=int, default=5100000)
    ap.add_argument("--repo", default=str(REPO), help="checkout whose outputs/ receives the figure")
    args = ap.parse_args()

    lid = bl.load_cached(bl.path_for(args.clip, args.t0, args.set))
    img = lid["range_img"]  # (4, 128, 1024): log range, intensity, rig z, valid
    valid = img[3] > 0.5  # (128, 1024)
    rng = np.where(valid, np.expm1(img[0] * np.log1p(bl.MAX_RANGE)), np.nan)  # m
    inten = np.where(valid, img[1], np.nan)
    height = np.where(valid, img[2] * 5.0, np.nan)  # m, rig frame (ground ~ 0)
    elev = lid["beam_elev"]  # (128,) deg, top -> bottom
    cam = dc.load_cached(dc.path_for(args.clip, args.t0, args.set))["image_frames"]
    cam = cam[dc.FRONT_WIDE_SLOT, -1].permute(1, 2, 0).numpy()  # (320, 576, 3)

    # the cache stores columns from -180 deg (rear, via the right) to +180; shown mirrored so the
    # vehicle's left is on the left, as in the camera frame above it
    rng, inten, height, valid = (a[:, ::-1] for a in (rng, inten, height, valid))
    dt = lid["dt_col"][::-1]  # (1024,) ms
    W = valid.shape[1]
    col = lambda deg: (180.0 - deg) / 360.0 * W  # azimuth (deg, left = +) -> displayed column
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(13.33, 7.5), dpi=200)
    L, Wd, CB = 0.075, 0.83, 0.012  # left edge, panel width, colorbar width (figure fractions)
    x0 = lambda deg: L + Wd * col(deg) / W

    a_cam = fig.add_axes([x0(CAM_HALF_FOV), 0.665, x0(-CAM_HALF_FOV) - x0(CAM_HALF_FOV), 0.27])
    a_cam.imshow(cam)
    a_cam.set_xticks([])
    a_cam.set_yticks([])
    fig.text(x0(CAM_HALF_FOV) - 0.01, 0.80, "Front-wide camera\nat t0", ha="right", va="center",
             fontsize=10, color=MUTED)

    rows = [("Range (m)", rng, "viridis", (0, 60)), ("Intensity", inten, "magma", (0, 0.5)),
            ("Height (m)", height, "cividis", (-2, 4))]
    ys = [0.455, 0.295, 0.135]
    cmap_bg = matplotlib.colors.ListedColormap([EMPTY])
    axes = []
    for (name, arr, cmap, (lo, hi)), y in zip(rows, ys):
        ax = fig.add_axes([L, y, Wd, 0.145])
        ax.imshow(np.zeros_like(valid, float), cmap=cmap_bg, aspect="auto", interpolation="nearest")
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, aspect="auto", interpolation="nearest")
        ticks = [12, 64, 116]
        ax.set_yticks(ticks, [f"{elev[t]:+.0f}°" for t in ticks])
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylabel(name, fontsize=10)
        ax.set_xticks([])
        for d in (-CAM_HALF_FOV, CAM_HALF_FOV):
            ax.axvline(col(d), color="white", lw=0.8, ls="--", alpha=0.9)
        cb = fig.add_axes([L + Wd + 0.008, y, CB, 0.145])
        fig.colorbar(im, cax=cb)
        cb.tick_params(labelsize=8)
        axes.append(ax)
    axes[0].text(col(0), -4, "front camera field of view", ha="center", va="bottom", fontsize=9,
                 color=MUTED)
    for d, xc in ((CAM_HALF_FOV, 0.0), (-CAM_HALF_FOV, 1.0)):  # camera corners -> FOV edges
        fig.add_artist(matplotlib.patches.ConnectionPatch(
            xyA=(xc, 0.0), coordsA=a_cam.transAxes, xyB=(col(d), 0), coordsB=axes[0].transData,
            color=MUTED, lw=0.8, ls=":"))

    a_dt = fig.add_axes([L, 0.075, Wd, 0.05])
    a_dt.plot(np.arange(W), dt, color=LINE, lw=2)
    a_dt.set_xlim(-0.5, W - 0.5)
    a_dt.set_ylabel("Age (ms)", fontsize=10)
    a_dt.set_yticks([100, 150])
    a_dt.tick_params(axis="y", labelsize=8)
    a_dt.grid(axis="y", color=EMPTY, lw=0.8)
    for s in ("top", "right"):
        a_dt.spines[s].set_visible(False)
    a_dt.set_xticks([col(d) for d in (180, 90, 0, -90, -180)],
                    ["rear", "left (90°)", "front (0°)", "right (90°)", "rear"])
    a_dt.tick_params(axis="x", labelsize=9)
    fig.text(L, 0.985, "LiDAR range image: 128 beams × 1024 azimuth bins, one sweep ending before t0, "
             "motion-compensated to t0", va="top", fontsize=11, color=INK)
    fig.text(L, 0.004, "Grey = no return (valid-mask channel). Age = t0 minus capture time per column, "
             "fed to the time position embedding.", fontsize=8.5, color=MUTED)

    out = Path(args.repo) / "outputs" / "distill_lidar_l0a" / "range_image_slide.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
