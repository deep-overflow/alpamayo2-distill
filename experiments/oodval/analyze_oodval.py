"""Alpamayo 2 Super vs Alpamayo 1.5 on OOD-val 262: paired open-loop comparison.

Joins the rows written by `run_oodval.py` with the Alpamayo-1.5 rows of
`baseline_pred_oodval` on `clip_id`. Both were produced under the same estimator
(1 CoC rollout, K=8 seeded denoisings), so every reduction below is taken the same
way on both sides: minADE@K is the minimum over the first K samples.

delta = A2 - A1.5 throughout, so a negative delta means Alpamayo 2 is closer to the
ground-truth path. minADE deltas are heavy-tailed, so the median and Wilcoxon are read
first and the mean (with a paired bootstrap CI) alongside, as in the compression track.

Usage:
  python experiments/oodval/analyze_oodval.py --exp-id a2_oodval
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
COMPRESSION = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(COMPRESSION / "experiments" / "head_analysis"))

import cache_a2 as ca
import eval_lib as el

A15_DIR = Path("/mnt/nvme1n1/ad_vla/outputs/chan/baseline_pred_oodval")
K_MAIN = 6  # the reduction the published tables use
HORIZONS = {"1.6 s": "_h16", "3.2 s": "_h32", "6.4 s": ""}

# dataviz reference palette, slots 1-2, validated together on the light surface
C15, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
NAME15, NAME2 = "Alpamayo 1.5 (10B)", "Alpamayo 2 Super (34B)"


def load_rows(d, skip_prefix=("config", "metrics")):
    rows = {}
    for p in sorted(Path(d).glob("*_s*of*.json")):
        if p.name.startswith(skip_prefix):
            continue
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r
    return rows


def reduce_k(row, k, metric="ade", suffix=""):
    return float(min(row[f"{metric}_rollout_k{suffix}"][:k]))


def avg_ade(row):
    """Expected single-sample ADE: the mean over the 8 samples, not the best of them."""
    return float(np.mean(row["ade_rollout_k"]))


def endpoint_spread(row):
    """RMS distance (m) of the 8 sampled 6.4 s endpoints from their centroid.

    best-of-K rewards diversity, so a tighter sampler gains less from K. Read this
    beside minADE@K before attributing a gap to accuracy.
    """
    end = np.asarray(row["pred_xy_k"])[:, -1]  # (K, 2)
    return float(np.sqrt(((end - end.mean(0)) ** 2).sum(1).mean()))


def arc_len(xy):
    """Distance travelled along a path (..., T, 2), counted from the ego origin at t0."""
    xy = np.concatenate([np.zeros_like(xy[..., :1, :]), xy], axis=-2)
    return np.linalg.norm(np.diff(xy, axis=-2), axis=-1).sum(-1)


def progress(row, gt_xy):
    """Longitudinal reading: distance travelled in 6.4 s, model minus human (m).

    `bias` is signed and averaged over the 8 samples (positive: drives further than the
    human did); `abs_best` is the magnitude for the best-of-6 sample, i.e. how much of
    that sample's error is speed rather than path shape.
    """
    d = arc_len(np.asarray(row["pred_xy_k"])) - arc_len(gt_xy)  # (K,)
    i_best = int(np.argmin(row["ade_rollout_k"][:K_MAIN]))
    return float(d.mean()), float(abs(d[i_best])), float(np.std(d))


def paired(a2, a15):
    """Paired summary of delta = a2 - a15 over clips."""
    a2, a15 = np.asarray(a2, float), np.asarray(a15, float)
    d = a2 - a15
    mean, lo, hi = el.paired_bootstrap_ci(d)
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(d), size=(10000, len(d)))
    med_lo, med_hi = np.percentile(np.median(d[idx], axis=1), [2.5, 97.5])
    out = {"n": len(d),
           "a15_mean": float(a15.mean()), "a2_mean": float(a2.mean()),
           "a15_median": float(np.median(a15)), "a2_median": float(np.median(a2)),
           "delta_mean": mean, "delta_mean_ci": [lo, hi],
           "delta_median": float(np.median(d)), "delta_median_ci": [float(med_lo), float(med_hi)],
           "rel_mean_pct": float(100 * (a2.mean() / a15.mean() - 1)),
           "rel_median_pct": float(100 * (np.median(a2) / np.median(a15) - 1)),
           "a2_win_rate": float((d < 0).mean()),
           "delta_trim10": float(stats.trim_mean(d, 0.1))}
    if len(d) >= 10 and np.any(d != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(a2, a15).pvalue)
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(color=GRID, lw=1.0)
    ax.set_axisbelow(True)


def plot_scatter(a15, a2, out):
    fig, ax = plt.subplots(figsize=(6.2, 6.0), facecolor=SURFACE)
    style(ax)
    lo, hi = 0.03, max(a15.max(), a2.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.0)
    better = a2 < a15
    for mask, c, lab in ((better, C2, f"{NAME2} closer (n={better.sum()})"),
                         (~better, C15, f"{NAME15} closer (n={(~better).sum()})")):
        ax.scatter(a15[mask], a2[mask], s=26, color=c, edgecolor=SURFACE, linewidth=0.8,
                   label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{NAME15}  minADE@{K_MAIN} (m)", color=INK2, fontsize=9)
    ax.set_ylabel(f"{NAME2}  minADE@{K_MAIN} (m)", color=INK2, fontsize=9)
    ax.set_title(f"Per-clip minADE@{K_MAIN}, OOD-val (n={len(a2)})", loc="left", fontsize=11,
                 color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_ecdf(a15, a2, out):
    fig, ax = plt.subplots(figsize=(7.0, 4.2), facecolor=SURFACE)
    style(ax)
    for v, c, lab in ((a15, C15, NAME15), (a2, C2, NAME2)):
        x = np.sort(v)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, lw=2.0, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(f"minADE@{K_MAIN} (m)", color=INK2, fontsize=9)
    ax.set_ylabel("share of clips at or below", color=INK2, fontsize=9)
    ax.set_title(f"Distribution of per-clip minADE@{K_MAIN}, OOD-val (n={len(a2)})", loc="left",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_groups(groups, title, out):
    """Dumbbell of the two medians per group, and the paired mean delta with its CI."""
    names = list(groups)
    y = np.arange(len(names))[::-1]
    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(11.0, 0.5 * len(names) + 1.8), sharey=True,
                                   facecolor=SURFACE, gridspec_kw={"wspace": 0.06})
    style(ax)
    style(ax_d)
    for yi, n in zip(y, names):
        g = groups[n]
        ax.plot([g["a15_median"], g["a2_median"]], [yi, yi], color=GRID, lw=2.0, zorder=2)
    ax.scatter([groups[n]["a15_median"] for n in names], y, s=64, color=C15, edgecolor=SURFACE,
               linewidth=2, zorder=3, label=NAME15)
    ax.scatter([groups[n]["a2_median"] for n in names], y, s=64, color=C2, edgecolor=SURFACE,
               linewidth=2, zorder=3, label=NAME2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  (n={groups[n]['n']})" for n in names], fontsize=8, color=INK)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"median minADE@{K_MAIN} (m)", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    # above the axes: inside, it lands on whichever row has the largest medians
    fig.legend(*ax.get_legend_handles_labels(), fontsize=8, frameon=False, labelcolor=INK,
               loc="upper right", bbox_to_anchor=(0.9, 1.0), ncol=2)

    ax_d.axvline(0, color=INK2, lw=1.0)
    for yi, n in zip(y, names):
        g = groups[n]
        ax_d.plot(g["delta_mean_ci"], [yi, yi], color=INK2, lw=2.0, zorder=2)
        ax_d.scatter([g["delta_mean"]], [yi], s=64, color=INK, edgecolor=SURFACE, linewidth=2,
                     zorder=3)
    ax_d.set_xlabel("paired mean delta A2 - A1.5 (m), 95% CI.  Left of 0: A2 closer",
                    color=INK2, fontsize=9)
    ax_d.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.02, y=1.0, ha="left", va="top", fontsize=11, color=INK)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_curves(by_k, by_h, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), facecolor=SURFACE)
    ks = sorted(by_k)
    hs = list(by_h)
    for ax, xs, pos, src, xlabel, title in (
            (axes[0], ks, ks, by_k, "K (best of the first K samples)", "mean minADE@K vs K, 6.4 s"),
            (axes[1], hs, [float(h.split()[0]) for h in hs], by_h, "horizon (s)",
             f"mean minADE@{K_MAIN} vs horizon")):
        style(ax)
        for key, c, lab in (("a15_mean", C15, NAME15), ("a2_mean", C2, NAME2)):
            ax.plot(pos, [src[x][key] for x in xs], color=c, lw=2.0, marker="o", ms=6,
                    mec=SURFACE, mew=1.5, label=lab)
        # the two lines end within a few cm of each other, so one shared end label
        # instead of two that would sit on top of one another
        last = src[xs[-1]]
        ax.annotate(f"A2 {last['a2_mean']:.2f}\nA1.5 {last['a15_mean']:.2f}",
                    (pos[-1], (last["a2_mean"] + last["a15_mean"]) / 2), xytext=(9, 0),
                    textcoords="offset points", fontsize=8, color=INK, va="center")
        ax.set_xticks(pos)
        ax.set_xticklabels([str(p) for p in pos])
        span = pos[-1] - pos[0]
        ax.set_xlim(pos[0] - 0.06 * span, pos[-1] + 0.22 * span)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
        ax.set_ylabel("m", color=INK2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def coc_stats(rows):
    first = [(r["gen_coc"].split() or [""])[0].lower() == (r["gt_coc"].split() or ["?"])[0].lower()
             for r in rows]
    return {"degenerate_rate": float(np.mean([r["coc_degenerate"] for r in rows])),
            "degenerate_strict_rate": float(np.mean([r["coc_degenerate_strict"] for r in rows])),
            "mean_tokens": float(np.mean([r["gen_len"] for r in rows])),
            "mean_chars": float(np.mean([r["coc_len"] for r in rows])),
            "first_word_matches_reference": float(np.mean(first))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="a2_oodval")
    ap.add_argument("--a2-dir", default=None, help="override the A2 rows dir (testing)")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    r15 = load_rows(A15_DIR)
    r2 = load_rows(args.a2_dir or out_dir)
    clips = sorted(set(r15) & set(r2))
    if not args.allow_partial and not (len(clips) == len(r15) == len(r2) == 262):
        raise SystemExit(f"expected 262 joined clips, got A1.5 {len(r15)}, A2 {len(r2)}, "
                         f"joined {len(clips)} (pass --allow-partial for an interim read)")
    for c in clips:
        assert r15[c]["bucket"] == r2[c]["bucket"], f"bucket differs on {c}: GT path mismatch"
    L15, L2 = [r15[c] for c in clips], [r2[c] for c in clips]

    m = {"n_clips": len(clips), "k_main": K_MAIN, "delta": "A2 - A1.5 (negative: A2 closer)"}
    m["overall"] = {
        f"min{met.upper()}@{k}": paired([reduce_k(r, k, met) for r in L2],
                                        [reduce_k(r, k, met) for r in L15])
        for met in ("ade", "fde") for k in (1, K_MAIN, 8)}
    m["overall"]["avgADE_over_8"] = paired([avg_ade(r) for r in L2], [avg_ade(r) for r in L15])
    m["overall"]["endpoint_spread_m"] = paired([endpoint_spread(r) for r in L2],
                                               [endpoint_spread(r) for r in L15])
    # GT path from the clip cache (rows carry predictions only); np.load is lazy, so this
    # touches the ego arrays and not the 28 JPEGs
    t0 = dict(zip(*[pd.read_parquet(ca.MANIFEST)[c] for c in ("clip_id", "t0_us")]))
    gts = [np.load(ca.path_for(c, int(t0[c])), allow_pickle=True)["ego_future_xyz"][0, 0, :, :2]
           for c in clips]
    p15 = np.array([progress(r, g) for r, g in zip(L15, gts)])  # (n, 3)
    p2 = np.array([progress(r, g) for r, g in zip(L2, gts)])
    for j, name in enumerate(("progress_bias_avg8_m", "progress_abs_err_best6_m",
                              "progress_std_over_8_m")):
        m["overall"][name] = paired(p2[:, j], p15[:, j])
    by_k = {k: paired([reduce_k(r, k) for r in L2], [reduce_k(r, k) for r in L15])
            for k in range(1, 9)}
    by_h = {h: paired([reduce_k(r, K_MAIN, "ade", s) for r in L2],
                      [reduce_k(r, K_MAIN, "ade", s) for r in L15]) for h, s in HORIZONS.items()}
    m["by_k"] = {str(k): v for k, v in by_k.items()}
    m["by_horizon"] = by_h

    a15 = np.array([reduce_k(r, K_MAIN) for r in L15])
    a2 = np.array([reduce_k(r, K_MAIN) for r in L2])
    for field in ("bucket", "cluster"):
        keys = sorted({r[field] for r in L15}, key=lambda g: -sum(r[field] == g for r in L15))
        m[f"by_{field}"] = {}
        for g in keys:
            sel = np.array([r[field] == g for r in L15])
            m[f"by_{field}"][g] = paired(a2[sel], a15[sel])
    m["tail"] = {f"share_minADE@{K_MAIN}_gt_{t}m": {"a15": float((a15 > t).mean()),
                                                     "a2": float((a2 > t).mean())}
                 for t in (1.0, 2.0, 4.0)}
    m["coc"] = {"a15": coc_stats(L15), "a2": coc_stats(L2),
                "identical_text_rate": float(np.mean([x["gen_coc"].strip() == y["gen_coc"].strip()
                                                      for x, y in zip(L15, L2)]))}
    if "sec_total" in L2[0]:
        m["a2_timing_s_per_clip"] = {k: float(np.mean([r[k] for r in L2]))
                                     for k in ("sec_gen", "sec_denoise", "sec_total")}
        m["a2_prompt_len_mean"] = float(np.mean([r["prompt_len"] for r in L2]))
    d = a2 - a15
    order = np.argsort(d)
    m["largest_a2_gains"] = [{"clip_id": clips[i], "a15": float(a15[i]), "a2": float(a2[i]),
                              "bucket": L15[i]["bucket"]} for i in order[:10]]
    m["largest_a2_losses"] = [{"clip_id": clips[i], "a15": float(a15[i]), "a2": float(a2[i]),
                               "bucket": L15[i]["bucket"]} for i in order[::-1][:10]]
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    # the replicas each wrote config_s{i}of{n}.json so they would not overwrite one
    # another; config.json is the run-level record the outputs/ convention expects
    shard_cfgs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("config_s*of*.json"))]
    if shard_cfgs:
        cfg = {k: v for k, v in shard_cfgs[0].items() if k not in ("shard", "gpus", "n_clips")}
        cfg.update({"n_clips": len(clips), "n_shards": len(shard_cfgs),
                    "gpus_per_shard": [c["gpus"] for c in shard_cfgs],
                    "compared_against": str(A15_DIR), "k_main": K_MAIN})
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    plot_scatter(a15, a2, out_dir / "plots" / "scatter_minade6.png")
    plot_ecdf(a15, a2, out_dir / "plots" / "ecdf_minade6.png")
    plot_groups(m["by_bucket"], f"minADE@{K_MAIN} by scenario bucket", out_dir / "plots" /
                "by_bucket.png")
    plot_groups(m["by_cluster"], f"minADE@{K_MAIN} by OOD event cluster", out_dir / "plots" /
                "by_cluster.png")
    plot_curves(by_k, by_h, out_dir / "plots" / "by_k_and_horizon.png")

    def line(name, p):
        w = f"  Wilcoxon p={p['wilcoxon_p']:.2e}" if "wilcoxon_p" in p else ""
        return (f"{name:34s} n={p['n']:3d}  A1.5 {p['a15_mean']:.4f}/{p['a15_median']:.4f}  "
                f"A2 {p['a2_mean']:.4f}/{p['a2_median']:.4f}  d_mean {p['delta_mean']:+.4f} "
                f"[{p['delta_mean_ci'][0]:+.4f}, {p['delta_mean_ci'][1]:+.4f}]  d_med "
                f"{p['delta_median']:+.4f}  A2 wins {100 * p['a2_win_rate']:.1f}%{w}")

    txt = [f"Alpamayo 2 Super vs Alpamayo 1.5, OOD-val, {len(clips)} clips joined",
           "columns: mean/median (m); delta = A2 - A1.5, negative means A2 closer to GT", ""]
    txt += [line(k, v) for k, v in m["overall"].items()] + ["", "by horizon (minADE@6):"]
    txt += [line(k, v) for k, v in by_h.items()] + ["", "by bucket (minADE@6):"]
    txt += [line(k, v) for k, v in m["by_bucket"].items()] + ["", "by cluster (minADE@6):"]
    txt += [line(k, v) for k, v in m["by_cluster"].items()] + ["", "tail:"]
    txt += [f"  {k}: A1.5 {v['a15']:.3f}  A2 {v['a2']:.3f}" for k, v in m["tail"].items()]
    txt += ["", f"CoC: {json.dumps(m['coc'])}"]
    if "a2_timing_s_per_clip" in m:
        txt += [f"A2 timing s/clip: {json.dumps(m['a2_timing_s_per_clip'])}"]
    (out_dir / "summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
