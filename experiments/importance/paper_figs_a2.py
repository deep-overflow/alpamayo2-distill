"""The seven paper figures of alpamayo-model-compression/paper-analysis/figures/1_main, each
rendered for Alpamayo 1.5 and Alpamayo 2 Super side by side (1.5 left, 2 right) so the two
models can be read against each other.

Rendering rules are the originals' (paper-analysis/scripts/{paper_same_position,
paper_token_share, head_map_combined}.py and experiments/paper/fig_criterion.py figure2), with
the layer axis, the late-band shading and the ticks adapted to each model's depth:

  1.5  36 layers, late band 22-34 (shaded as in the paper), step at layer 23
  2    64 layers, late band 39-62 = the paper band moved by relative depth, and a dotted
       line at the fitted step (layer 51 for Q heads, 50 for MLP channels;
       plans/2026-09-27_a2-dual-importance.md). The 1.5 panels carry the same dotted line
       at their step (23) so the two panels have the same elements.

Data: 1.5 from gradanat_v1 (typed per-clip gradients) and stepimp_fm_perstep_v2 (expert
per-step gradients); 2 from outputs/a2_importance_v1 (one run gives both).

Usage:
  python experiments/importance/paper_figs_a2.py            # all seven
  python experiments/importance/paper_figs_a2.py --only fig2_mass_curve fig2_steps_q_head
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
COMPRESSION = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(COMPRESSION / "paper-analysis" / "scripts"))
import paper_style as ps  # applies fig_criterion's rcParams on import

OUT = REPO / "figures" / "a2_vs_a15"
OUT15 = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
RUN2 = REPO / "outputs" / "a2_importance_v1"
VIS, HIST, PT, COC, SINK = range(5)
PURPLE = "#8E6BB3"
TRAJ2, COC2 = "#0072B2", "#D55E00"  # fig_criterion's default palette (fig2_* in 1_main use it)

MODELS = {
    "a15": {"name": "Alpamayo 1.5", "n": 36, "band": (21.5, 35.5), "tau": {"q": 23, "mlp": 23},
            "ticks": [0, 7, 14, 21, 28, 35], "kv_group": 4,
            "mlp_ticklabels": ["0", "3k", "6k", "9k", "12k"],
            "anat_q": OUT15 / "gradanat_v1" / "anatomy_perclip_q.npz",
            "anat_mlp": OUT15 / "gradanat_v1" / "anatomy_perclip_mlp.npz",
            "steps": OUT15 / "stepimp_fm_perstep_v2" / "step_importance.npz"},
    "a2": {"name": "Alpamayo 2 Super", "n": 64, "band": (38.5, 63.5), "tau": {"q": 51, "mlp": 50},
           "ticks": [0, 9, 18, 27, 36, 45, 54, 63], "kv_group": 8,
           "mlp_ticklabels": ["0", "6.4k", "12.8k", "19.2k", "25.6k"],
           "anat_q": RUN2 / "anatomy_perclip_q.npz", "anat_mlp": RUN2 / "anatomy_perclip_mlp.npz",
           "steps": RUN2 / "step_importance.npz"},
}
KEYS = {"q": ("fm_full", "ce"), "mlp": ("fm_type", "ce_type")}


def rho_layers(a, b):
    out = np.full(a.shape[0], np.nan)
    for li in range(a.shape[0]):
        if np.ptp(a[li]) > 0 and np.ptp(b[li]) > 0:
            out[li] = spearmanr(a[li], b[li])[0]
    return out


def load_typed(m, axis):
    z = np.load(m[f"anat_{axis}"])
    kt, kc = KEYS[axis]
    return z[kt], z[kc]  # (N, 5, L, U) signed per-clip typed gradients


def depth_marks(ax, m, axis):
    """Late-band shading, the step line and the layer ticks of one model."""
    ax.axvspan(*m["band"], color=ps.GREY, alpha=0.13, lw=0)
    ax.axvline(m["tau"][axis] - 0.5, color="black", lw=0.8, ls=(0, (1.5, 2)), gid="step")
    ax.set_xlim(-0.5, m["n"] - 0.5)
    ax.set_xticks(m["ticks"])
    ax.set_xlabel("VLM layer index")


def pair(figsize=(7.2, 2.8)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, key in zip(axes, ("a15", "a2")):
        ax.set_title(MODELS[key]["name"], fontsize=ps.TEXT, pad=4)
    return fig, axes


def export_pair(fig, axes, name, ncol):
    """paper_style.export for a two-panel figure: one legend above both panels, and the step
    marker keeps its dotted style (export() would make every line solid)."""
    for ax in axes:
        for t in (ax.xaxis.label, ax.yaxis.label):
            t.set_fontsize(ps.LABEL)
        ax.tick_params(axis="both", which="both", labelsize=ps.TEXT)
        ax.title.set_fontsize(ps.TEXT)
        for line in ax.lines:
            if line.get_gid() != "step":
                line.set_linestyle("-")
        ax.grid(axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
               fontsize=ps.TEXT, handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
               frameon=False)
    fig.tight_layout(pad=0.8, rect=(0, 0, 1, 0.9))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext_ in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext_}")
    plt.close(fig)
    print(f"  {name}.png / .pdf / .svg")


def same_position():
    """fig4_same_position_q_head: raw within-layer agreement at one kind of position."""
    fig, axes = pair()
    colors = {"All": "black", "Vision": ps.BLUE, "Prompt text": ps.YELLOW, "CoC": PURPLE,
              "Ego history": ps.BRICK}
    for ax, key in zip(axes, ("a15", "a2")):
        m = MODELS[key]
        fm, ce = load_typed(m, "q")
        fm32, ce32 = fm.astype(np.float32), ce.astype(np.float32)
        last = m["n"] - 1
        series = {"All": rho_layers(np.abs(fm32.sum(1)).mean(0), np.abs(ce32.sum(1)).mean(0))}
        for label, ty in (("Vision", VIS), ("Prompt text", PT), ("CoC", COC), ("Ego history", HIST)):
            series[label] = rho_layers(np.abs(fm32[:, ty]).mean(0), np.abs(ce32[:, ty]).mean(0))
        L = np.arange(last)
        depth_marks(ax, m, "q")
        for label, y in series.items():
            ax.plot(L, y[:last], color=colors[label], lw=1.8 if label == "All" else 1.3, label=label)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylim(-0.25, 1.05)
        ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
        late = slice(round(22 * m["n"] / 36), last)
        print(f"  {m['name']} same-position late means: "
              + ", ".join(f"{k} {np.nanmean(v[late]):.2f}" for k, v in series.items()))
    axes[0].set_ylabel(r"Rank agreement $\rho$")
    export_pair(fig, axes, "fig4_same_position_q_head", 5)


def token_share():
    """fig4_token_share_{traj,coc}_q_head_split: additive share of each layer's score by position."""
    shares = {}
    for key in ("a15", "a2"):
        fm, ce = load_typed(MODELS[key], "q")
        for obj, arr in (("traj", fm), ("coc", ce)):
            part = np.zeros(arr.shape[1:3])  # (5, L)
            for i in range(len(arr)):
                g = arr[i].astype(np.float64)
                part += (g * np.sign(g.sum(0, keepdims=True))).sum(-1)
            part /= len(arr)
            tot = part.sum(0)
            shares[(key, obj)] = np.nan_to_num(part / np.where(tot > 0, tot, np.nan))
    for obj in ("traj", "coc"):
        fig, axes = pair()
        for ax, key in zip(axes, ("a15", "a2")):
            m = MODELS[key]
            s = shares[(key, obj)]
            last = m["n"] - 1 if obj == "traj" else m["n"]  # I_traj is identically 0 at the last layer
            L = np.arange(last)
            depth_marks(ax, m, "q")
            ax.plot(L, s[VIS][:last], color=ps.BLUE, lw=1.5, label="Vision")
            ax.plot(L, s[PT][:last], color=ps.YELLOW, lw=1.5, label="Prompt text")
            ax.plot(L, s[COC][:last], color=PURPLE, lw=1.5, label="CoC")
            ax.plot(L, s[HIST][:last], color=ps.BRICK, lw=1.5, label="Ego history")
            ax.set_ylim(0, 1.05)
        axes[0].set_ylabel("Share of layer importance")
        export_pair(fig, axes, f"fig4_token_share_{obj}_q_head_split", 4)


def analyse(arr, last):
    """head_map_combined.analyse: dominant position type, its share, rank in layer, I_FM order."""
    g = arr.astype(np.float32)
    C = (g * np.sign(g.sum(1, keepdims=True))).mean(0).astype(np.float64)  # (5, L, U)
    I = C.sum(0)
    types = ((VIS, ps.BLUE), (PT, ps.YELLOW), (COC, PURPLE), (HIST, ps.BRICK))
    C4 = np.stack([C[t] for t, _ in types])
    dom, share = C4.argmax(0), np.clip(C4.max(0) / np.where(I > 0, I, np.nan), 0, 1)
    rank = np.full(I.shape, np.nan)
    for l in range(last):
        rank[l] = np.argsort(np.argsort(I[l])) / (I.shape[1] - 1)
    order = np.stack([np.argsort(-I[l]) for l in range(I.shape[0])])
    return {"dom": dom, "share": share, "rank": rank, "order": order, "last": last,
            "colors": [c for _, c in types]}


def type_img(d):
    L, U = d["dom"].shape
    img = np.zeros((L, U, 4))
    for k, col in enumerate(d["colors"]):
        img[d["dom"] == k, :3] = to_rgb(col)
    img[..., 3] = 0.12 + 0.88 * np.nan_to_num(d["share"], nan=0.0)
    img[d["last"]:] = (0.85, 0.85, 0.85, 1.0)
    return img


def head_map():
    """fig6e_head_map_combined_sortq, the two models as two 2x4 blocks."""
    grey = (0.85, 0.85, 0.85)
    LABEL, TICK, LEGEND, TITLE = 15, 12, 15, 15
    fig, axes = plt.subplots(2, 8, figsize=(ps.SINGLE[0] * 8.4, ps.SINGLE[1] * 2.9))
    titles = (r"$I_{\mathrm{FM}}$: position type", r"$I_{\mathrm{CE}}$: position type",
              r"$I_{\mathrm{FM}}$: rank in layer", r"$I_{\mathrm{CE}}$: rank in layer")
    im_rank = None
    for b, key in enumerate(("a15", "a2")):
        m = MODELS[key]
        n = m["n"]
        for r, axis in enumerate(("q", "mlp")):
            fm, ce = load_typed(m, axis)
            d_fm, d_ce = analyse(fm, n - 1), analyse(ce, n)
            L, U = d_fm["dom"].shape
            ext = (-0.5, U - 0.5, -0.5, L - 0.5)
            def ordered(M, o=d_fm["order"], n_l=L):
                return np.stack([M[l, o[l]] for l in range(n_l)])
            panels = [type_img(d_fm), type_img(d_ce),
                      np.ma.masked_invalid(ordered(d_fm["rank"])), np.ma.masked_invalid(ordered(d_ce["rank"]))]
            for c, P in enumerate(panels):
                ax = axes[r, 4 * b + c]
                if c < 2:
                    ax.imshow(P, origin="lower", extent=ext, interpolation="nearest", aspect="auto")
                else:
                    im_rank = ax.imshow(P, origin="lower", extent=ext, interpolation="nearest",
                                        aspect="auto", cmap="viridis", vmin=0, vmax=1)
                    im_rank.cmap.set_bad(grey)
                ax.axhline(m["band"][0], color="black", lw=0.8, ls=(0, (4, 2)))
                ax.axhline(m["tau"][axis] - 0.5, color="black", lw=0.8, ls=(0, (1.5, 2)))
                ax.set_yticks(m["ticks"])
                if axis == "q":
                    if c < 2:
                        for x in np.arange(m["kv_group"] - 0.5, U - 1, m["kv_group"]):
                            ax.axvline(x, color="white", lw=0.5)
                    ax.set_xticks([0, U // 4 - 1, U // 2 - 1, 3 * U // 4 - 1, U - 1])
                    ax.set_xlabel(r"Q head ($I_{\mathrm{FM}}$ rank)" if c >= 2 else "Q head index")
                    ax.set_title(titles[c], fontsize=TITLE)
                else:
                    step = U // 4
                    ax.set_xticks([0, step, 2 * step, 3 * step, U - 1])
                    ax.set_xticklabels(m["mlp_ticklabels"])
                    ax.set_xlabel("MLP ch. index" if c < 2 else r"MLP ch. ($I_{\mathrm{FM}}$ rank)")
                if c == 0:
                    ax.set_ylabel("Layer (" + ("Q heads" if axis == "q" else "MLP") + ")")
        fig.text((0.235, 0.705)[b], 0.985, m["name"], ha="center", va="top", fontsize=TITLE + 2,
                 fontweight="semibold")
    for ax in axes.flat:
        ax.xaxis.label.set_size(LABEL)
        ax.yaxis.label.set_size(LABEL)
        ax.tick_params(axis="both", labelsize=TICK)
    handles = [Patch(color=c, label=n) for c, n in zip(analyse(np.zeros((1, 5, 2, 2)), 1)["colors"],
                                                       ("Vision", "Prompt text", "Generated CoC", "Ego history"))]
    fig.tight_layout(pad=0.8, rect=(0, 0.08, 0.945, 0.955))
    cax = fig.add_axes([0.952, 0.20, 0.006, 0.62])
    cb = fig.colorbar(im_rank, cax=cax)
    cb.set_label("Rank in layer", fontsize=LABEL)
    cb.ax.tick_params(labelsize=TICK)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=LEGEND, frameon=False,
               bbox_to_anchor=(0.47, 0.0), handlelength=1.4, columnspacing=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext_ in ("png", "pdf"):
        fig.savefig(OUT / f"fig6e_head_map_combined_sortq.{ext_}", dpi=300)
    plt.close(fig)
    print("  fig6e_head_map_combined_sortq.png / .pdf")


def step_matrix(a):
    n_step = a.shape[0]
    M = np.ones((n_step, n_step))
    for i in range(n_step):
        for j in range(i + 1, n_step):
            M[i, j] = M[j, i] = np.nanmean([spearmanr(a[i, l], a[j, l]).statistic for l in range(a.shape[1])])
    return M


def steps(axis):
    """fig2_steps_{q_head,mlp}: rank agreement between denoising steps."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    im = None
    for ax, key in zip(axes, ("a15", "a2")):
        m = MODELS[key]
        a = np.load(m["steps"])[f"{axis}_abs_step"]  # (S, L, U)
        M = step_matrix(a)
        im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
        n_step = a.shape[0]
        ax.set_xticks(range(n_step))
        ax.set_yticks(range(n_step))
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("Denoising step")
        ax.set_title(m["name"], fontsize=9, pad=4)
        for sp in ax.spines.values():
            sp.set_visible(True)
        off = ~np.eye(n_step, dtype=bool)
        print(f"  {m['name']} {axis}: step-to-step rho {M[off].mean():.3f} (min {M[off].min():.3f})")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cb.set_label(r"rank agreement $\rho$", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    name = {"q": "fig2_steps_q_head", "mlp": "fig2_steps_mlp"}[axis]
    OUT.mkdir(parents=True, exist_ok=True)
    for ext_ in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext_}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf / .svg")


def mass_curve():
    """fig2_mass_curve: cumulative importance when expert units are removed lowest-score-first."""
    def curve(a):
        cur = [np.cumsum(np.sort(a[l])) / max(a[l].sum(), 1e-12) for l in range(a.shape[0])]
        return np.mean(np.stack(cur), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, key in zip(axes, ("a15", "a2")):
        m = MODELS[key]
        z = np.load(m["steps"])
        cq, cm = curve(z["q_abs_step"].sum(0)), curve(z["mlp_abs_step"].sum(0))
        ax.plot([0, 1], [0, 1], color=ps.GREY, lw=0.8, ls="--", label="uniform")
        ax.plot(np.arange(len(cq) + 1) / len(cq), np.r_[0, cq], color=TRAJ2, lw=1.5, label="Q head")
        ax.plot(np.arange(len(cm) + 1) / len(cm), np.r_[0, cm], color=COC2, lw=1.5, ls=(0, (4, 2)),
                label="MLP channel")
        ax.axvline(0.9, color="black", lw=0.7, ls=":")
        q90, m90 = cq[int(0.9 * len(cq)) - 1], cm[int(0.9 * len(cm)) - 1]
        xq, xm = int(0.9 * len(cq)) / len(cq), int(0.9 * len(cm)) / len(cm)
        ax.scatter([xq], [q90], color=TRAJ2, s=18, zorder=4)
        ax.scatter([xm], [m90], color=COC2, marker="s", s=18, zorder=4)
        ax.annotate(f"{100 * q90:.0f}%", (xq, q90), textcoords="offset points", xytext=(-30, 2),
                    color=TRAJ2, fontsize=7.5)
        ax.annotate(f"{100 * m90:.1f}%", (xm, m90), textcoords="offset points", xytext=(-32, 6),
                    color=COC2, fontsize=7.5)
        ax.set_xlabel("Fraction of units removed")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(m["name"], fontsize=9, pad=4)
        ax.grid(axis="y")
        print(f"  {m['name']}: bottom 90% carries Q {100 * q90:.1f}%  MLP {100 * m90:.1f}%")
    axes[0].set_ylabel("Importance removed")
    axes[0].legend(loc="upper left")
    fig.tight_layout(pad=0.8)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext_ in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig2_mass_curve.{ext_}")
    plt.close(fig)
    print("  fig2_mass_curve.png / .pdf / .svg")


FIGS = {"fig4_same_position_q_head": same_position, "fig4_token_share": token_share,
        "fig6e_head_map_combined_sortq": head_map, "fig2_mass_curve": mass_curve,
        "fig2_steps_q_head": lambda: steps("q"), "fig2_steps_mlp": lambda: steps("mlp")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(FIGS), default=None)
    args = ap.parse_args()
    for name in args.only or FIGS:
        print(name)
        FIGS[name]()
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
