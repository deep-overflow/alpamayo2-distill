"""Per-clip qualitative view: Alpamayo 1.5 vs Alpamayo 2 Super on one OOD-val clip.

Same idea as the compression track's `viz_openloop_ood.py` (front camera at t0, the
rolled-out trajectories against ground truth in bird's-eye view, the generated CoC
against the curated reference), with the two models as the two rows and the camera
block extended to the six views Alpamayo 2 consumes. Alpamayo 1.5 only sees four of
them, which the thumbnails mark, because a rear view is the one input difference that
can explain a disagreement between the two.

Predictions and ground truth share the ego rig frame at t0 (x forward, y left), so the
BEV panels invert the horizontal axis: left in the plot is left in the world. Both
panels of a clip share one axis box so an overshoot is drawn at its true size.

Highlighted sample and the numbers in the panel titles are best-of-6 (`[:6]`), the
reduction the tables use; samples 7-8 are drawn fainter.

Usage:
  python experiments/oodval/viz_oodval.py --exp-id a2_oodval              # all 262
  python experiments/oodval/viz_oodval.py --clip 002b195e-3e0d-4a20-92dd-0e9920e74461
"""

import argparse
import io
import json
import sys
import textwrap
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

import cache_a2 as ca
from analyze_oodval import A15_DIR, C2, C15, GRID, INK, INK2, NAME2, NAME15, SURFACE, load_rows

K_EVAL = 6
FRONT = 1
# (camera id, label, seen by Alpamayo 1.5)
THUMBS = [(0, "front left", True), (6, "front tele", True), (2, "front right", True),
          (3, "rear left", False), (5, "rear right", False)]


def load_clip(clip_id, t0_us):
    z = np.load(ca.path_for(clip_id, t0_us), allow_pickle=True)
    n_frame = int(z["n_frame"])
    cams = z["camera_indices"].tolist()

    def frame(cam_id):  # the frame at t0 is the last of the 4
        return Image.open(io.BytesIO(z["jpeg_bytes"][cams.index(cam_id) * n_frame + n_frame - 1]))

    return {"front": frame(FRONT), "thumbs": [frame(c).reduce(4) for c, _, _ in THUMBS],
            "hist": z["ego_history_xyz"][0, 0][:, :2],  # (16, 2)
            "gt": z["ego_future_xyz"][0, 0][:, :2]}  # (64, 2)


def best_of_k(row, k=K_EVAL):
    ade = np.asarray(row["ade_rollout_k"][:k])
    return int(np.argmin(ade)), float(ade.min()), float(min(row["fde_rollout_k"][:k]))


def bev_limits(clip, rows, pad=2.0):
    xy = np.concatenate([clip["hist"], clip["gt"]]
                        + [np.asarray(r["pred_xy_k"]).reshape(-1, 2) for r in rows])
    lo, hi = xy[:, 1].min() - pad, xy[:, 1].max() + pad
    mid, half = (lo + hi) / 2, max((hi - lo) / 2, 4.0)
    return (mid + half, mid - half), (xy[:, 0].min() - pad, xy[:, 0].max() + pad)


def draw_bev(ax, clip, row, name, color, xlim, ylim):
    pred = np.asarray(row["pred_xy_k"])  # (8, 64, 2)
    i_best, ade, fde = best_of_k(row)
    ax.set_facecolor(SURFACE)
    ax.plot(clip["hist"][:, 1], clip["hist"][:, 0], color=INK2, lw=1.4, ls=":")
    for j in range(len(pred)):
        if j != i_best:
            ax.plot(pred[j, :, 1], pred[j, :, 0], color=color, lw=0.9,
                    alpha=0.40 if j < K_EVAL else 0.15)
    ax.plot(clip["gt"][:, 1], clip["gt"][:, 0], color=INK, lw=2.4)
    ax.plot(pred[i_best, :, 1], pred[i_best, :, 0], color=color, lw=2.0)
    ax.plot(0, 0, marker="s", ms=5, color=INK)
    ax.set_xlim(*xlim)  # reversed: +y is left
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_title(f"{name}\nminADE@{K_EVAL} {ade:.2f} m\nminFDE@{K_EVAL} {fde:.2f} m",
                 fontsize=8.5, loc="left", color=INK)
    ax.set_xlabel("lateral y (m)   ← left", fontsize=8, color=INK2)
    ax.set_ylabel("forward x (m)", fontsize=8, color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_color(GRID)


def render(task):
    clip_id, t0_us, r15, r2, out_png = task
    clip = load_clip(clip_id, t0_us)
    # the camera block is sized to the 16:9 frames so it leaves no dead bands; the two
    # BEV panels sit side by side at full height because driving paths are long and thin
    fig = plt.figure(figsize=(14.0, 7.7), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.62, 0.5, 0.5], left=0.025, right=0.985,
                          top=0.86, bottom=0.21, wspace=0.16)

    cam = gs[0, 0].subgridspec(2, len(THUMBS), height_ratios=[5.0, 1.0], hspace=0.10,
                               wspace=0.03)
    ax = fig.add_subplot(cam[0, :])
    ax.imshow(clip["front"])
    ax.set_axis_off()
    ax.set_anchor("N")
    ax.set_title("front wide camera @ t0", fontsize=9, loc="left", color=INK)
    for i, ((_, label, in15), img) in enumerate(zip(THUMBS, clip["thumbs"])):
        ax = fig.add_subplot(cam[1, i])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_anchor("N")
        ax.set_title(label + ("" if in15 else "  (A2 only)"), fontsize=7.5, color=INK2, pad=2)

    xlim, ylim = bev_limits(clip, [r15, r2])
    draw_bev(fig.add_subplot(gs[0, 1]), clip, r15, NAME15, C15, xlim, ylim)
    draw_bev(fig.add_subplot(gs[0, 2]), clip, r2, NAME2, C2, xlim, ylim)
    fig.legend(handles=[Line2D([], [], color=INK, lw=2.4, label="ground truth"),
                        Line2D([], [], color=INK2, lw=2.0, label=f"best of {K_EVAL} (model color)"),
                        Line2D([], [], color=INK2, lw=0.9, alpha=0.5, label="other samples"),
                        Line2D([], [], color=INK2, lw=1.4, ls=":", label="ego history")],
               fontsize=7.5, loc="lower right", bbox_to_anchor=(0.985, 0.03), frameon=False,
               labelcolor=INK, ncol=2)

    fig.suptitle(f"{clip_id}    bucket={r2['bucket']}    cluster={r2['cluster']}    t0_us={t0_us}",
                 fontsize=10, x=0.025, ha="left", y=0.965, color=INK)
    y = 0.125
    for label, text, color in (("reference CoC", r2["gt_coc"], INK),
                               (NAME15, r15["gen_coc"] or "<empty>", C15),
                               (NAME2, r2["gen_coc"] or "<empty>", C2)):
        body = textwrap.shorten(str(text).replace("\n", " "), width=120, placeholder=" ...")
        fig.add_artist(Line2D([0.025, 0.04], [y + 0.007, y + 0.007], color=color, lw=4,
                              transform=fig.transFigure))
        fig.text(0.047, y, label, fontsize=8.5, fontweight="bold", color=INK)
        fig.text(0.185, y, body, fontsize=8.5, color=INK)
        y -= 0.04
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return clip_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="a2_oodval")
    ap.add_argument("--clip", nargs="*", default=None, help="explicit clip ids (default: all)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    viz = out_dir / "viz"
    r15, r2 = load_rows(A15_DIR), load_rows(out_dir)
    t0 = dict(zip(*[pd.read_parquet(ca.MANIFEST)[c] for c in ("clip_id", "t0_us")]))
    clips = args.clip or sorted(set(r15) & set(r2))
    tasks = [(c, int(t0[c]), r15[c], r2[c], viz / "all" / f"{c}.png") for c in clips]
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(render, tasks), 1):
            if i % 25 == 0 or i == len(tasks):
                print(f"rendered {i}/{len(tasks)}", flush=True)

    # reading lists: where to look first
    a15 = {c: best_of_k(r15[c])[1] for c in clips}
    a2 = {c: best_of_k(r2[c])[1] for c in clips}
    lists = {"worst10_alpamayo2": sorted(clips, key=lambda c: -a2[c])[:10],
             "worst10_alpamayo1.5": sorted(clips, key=lambda c: -a15[c])[:10],
             "largest_gain_a2_over_a1.5": sorted(clips, key=lambda c: a2[c] - a15[c])[:10],
             "largest_loss_a2_vs_a1.5": sorted(clips, key=lambda c: a15[c] - a2[c])[:10]}
    md = [f"# OOD-val per-clip figures ({len(clips)} clips, minADE@{K_EVAL})", ""]
    for name, ids in lists.items():
        md += [f"## {name}", "", "| clip | bucket | A1.5 | A2 | reference CoC |", "|---|---|---:|---:|---|"]
        md += [f"| [{c[:8]}](all/{c}.png) | {r2[c]['bucket']} | {a15[c]:.2f} | {a2[c]:.2f} | "
               f"{r2[c]['gt_coc']} |" for c in ids] + [""]
    (viz / "index.md").write_text("\n".join(md))
    (viz / "lists.json").write_text(json.dumps(lists, indent=2))
    print(f"-> {viz}", flush=True)


if __name__ == "__main__":
    main()
