"""Finding 1 on Alpamayo 1.5 and Alpamayo 2 Super, computed by the same functions.

plans/2026-09-27_a2-dual-importance.md. Every quantity is computed for both models from
their own run files with one code path, so the comparison cannot hide a rule change:

  1.5  importance_v2 (Fig. 1 source), gradanat_v1 (token anatomy), stepimp_fm_perstep_v2
  2    a2_importance_v1 (one run gives all three; see run_importance_a2.py)

The rules are copied from alpamayo-model-compression: rho_per_layer (fig_criterion.py),
step_fit / ratio / noise ceiling / hand-off (experiments/paper/fig_why_differs.py), the
additive token share and same-position agreement (paper-analysis/scripts/finding1_numbers.py),
the expert mass curve (analyze_expert_axis.py). Depth bands are Alpamayo 1.5's, moved to
Alpamayo 2 by relative depth; the last layer is left out of every band because I_traj is
identically zero there.

G0c: on the 1.5 files this script must reproduce the published numbers (REF below);
the run stops with a nonzero exit if it does not. G0a (last layer exactly zero) is checked
on both models.

Usage:
  python experiments/importance/analyze_importance_a2.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
OUT15 = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
VIS, HIST, PT, COC, SINK = range(5)  # a2_grad_lib.TYPES / run_gradient_anatomy.TYPES order

# dataviz reference palette, slots 1-2 (validated together); color = model everywhere
C15, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
NAME = {"a15": "Alpamayo 1.5 (36 layers)", "a2": "Alpamayo 2 Super (64 layers)"}
COLOR = {"a15": C15, "a2": C2}

# published Alpamayo 1.5 values the 1.5 half of this script must reproduce (G0c)
REF = {
    "rho_q_trunk": (0.782, 0.001), "rho_q_late": (0.335, 0.001),
    "rho_mlp_trunk": (0.753, 0.001), "rho_mlp_late": (0.339, 0.001),
    "tau_q": (23, 0), "tau_mlp": (23, 0), "step_q": (4.640, 0.001), "step_mlp": (4.558, 0.001),
    "r2_q": (0.858, 0.001), "r2_mlp": (0.933, 0.001),
    "disatt_late_q": (0.375, 0.001), "disatt_late_mlp": (0.539, 0.001),
    "ceiling_share_q": (0.027, 0.001), "ceiling_share_mlp": (0.441, 0.001),
    "peak_traj_q": (19, 0), "peak_coc_q": (24, 0), "peak_traj_mlp": (18, 0), "peak_coc_mlp": (27, 0),
    "exp_bottom50_q": (0.2496, 0.0001), "exp_bottom50_mlp": (0.0044, 0.0001),
    "handoff_traj_q": (19, 0), "handoff_coc_q": (17, 0),
    "handoff_traj_mlp": (21, 0), "handoff_coc_mlp": (18, 0),
    "vis_lo_traj_mlp": (0.901, 0.001), "vis_hi_traj_mlp": (0.264, 0.001),
    "vis_lo_coc_mlp": (0.749, 0.001), "vis_hi_coc_mlp": (0.018, 0.001),
    "vis_lo_traj_q": (0.721, 0.001), "vis_hi_traj_q": (0.353, 0.001),
    "vis_lo_coc_q": (0.626, 0.001), "vis_hi_coc_q": (0.016, 0.001),
    "same_vision_late_corr_q": (0.79, 0.015), "same_pooled_late_corr_q": (0.38, 0.015),
}


def spec(n_layers, n_q, n_mlp):
    """Depth bands for a model with n_layers, moved from Alpamayo 1.5's by relative depth."""
    s = n_layers / 36
    b = round(22 * s)  # 22 -> 22 / 39
    return {
        "N": n_layers, "LAST": n_layers - 1,
        "trunk": slice(0, b), "late": slice(b, n_layers - 1),
        "tok_lo": slice(round(6 * s), round(17 * s) + 1),  # 6-17 -> 11-30
        "tok_hi": slice(round(27 * s), round(34 * s) + 1),  # 27-34 -> 48-60
        "handoff": (round(6 * s), round(33 * s)),  # range(6, 33) -> range(11, 59)
        "keep_q": round(19 / 32 * n_q), "keep_mlp": round(7390 / 12288 * n_mlp),
        "tau_window": (round((0.657 - 0.08) * (n_layers - 1)), round((0.657 + 0.08) * (n_layers - 1))),
    }


def rho_per_layer(a, b):
    return np.array([spearmanr(a[l], b[l]).statistic if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0
                     else np.nan for l in range(a.shape[0])])


def step_fit(y):
    """Two-level piecewise-constant least-squares fit. y: (n,) -> tau, level0, level1, R2."""
    best = None
    for tau in range(3, len(y) - 3):
        a, b = y[:tau].mean(), y[tau:].mean()
        sse = ((y[:tau] - a) ** 2).sum() + ((y[tau:] - b) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, tau, a, b)
    sse, tau, a, b = best
    return tau, a, b, 1 - sse / ((y - y.mean()) ** 2).sum()


def ratio_block(z, pc, sp, rng):
    """fig_why_differs.ratio_panels: log(sum I_traj / sum I_CoC) over depth, step fit, bootstrap."""
    L = sp["LAST"]
    out = {}
    for ax in ("q", "mlp"):
        t, c = z[f"traj_vlm_{ax}"].sum(1)[:L], z[f"coc_vlm_{ax}"].sum(1)[:L]  # (L,)
        y = np.log(t / c)
        tau, a, b, r2 = step_fit(y)
        T, C = pc[f"traj_vlm_{ax}"].sum(2)[:, :L], pc[f"coc_vlm_{ax}"].sum(2)[:, :L]  # (N, L)
        boot = np.array([step_fit(np.log(T[i].mean(0) / C[i].mean(0)))[:3]
                         for i in rng.integers(0, len(T), (2000, len(T)))])  # (2000, 3)
        steps = np.exp(boot[:, 1] - boot[:, 2])
        per_clip = np.array([step_fit(np.log(T[i] / C[i]))[:3] for i in range(len(T))])
        vals, counts = np.unique(boot[:, 0], return_counts=True)
        out[ax] = {"tau": int(tau), "early": float(np.exp(a)), "late": float(np.exp(b)),
                   "step": float(np.exp(a - b)), "r2": float(r2),
                   "boot_tau_share": float(np.mean(boot[:, 0] == tau)),
                   "boot_tau_mode": int(vals[counts.argmax()]),
                   "boot_tau_mode_share": float(counts.max() / len(boot)),
                   "boot_step_ci": [float(np.percentile(steps, 2.5)), float(np.percentile(steps, 97.5))],
                   "clips_with_step": float(np.mean(per_clip[:, 1] > per_clip[:, 2])),
                   "ratio_by_layer": np.exp(y).tolist()}
    return out


def ceiling_block(pc, sp, rng, n_split=50):
    """fig_why_differs.ceiling_panels: split-half self- and cross-agreement per layer."""
    out = {}
    for ax in ("q", "mlp"):
        T, C = pc[f"traj_vlm_{ax}"], pc[f"coc_vlm_{ax}"]  # (N, L, U)
        n = len(T)
        same, cross = [], []
        for _ in range(n_split):
            p = rng.permutation(n)
            A, B = p[:n // 2], p[n // 2:]
            tA, tB, cA, cB = T[A].mean(0), T[B].mean(0), C[A].mean(0), C[B].mean(0)
            same.append([rho_per_layer(tA, tB), rho_per_layer(cA, cB)])
            cross.append(0.5 * (rho_per_layer(tA, cB) + rho_per_layer(tB, cA)))
        same, cross = np.mean(same, 0), np.mean(cross, 0)  # (2, L), (L,)
        dis = cross / np.sqrt(same[0] * same[1])
        band = {k: [float(np.nanmean(v[sp["trunk"]])), float(np.nanmean(v[sp["late"]]))]
                for k, v in (("traj_self", same[0]), ("coc_self", same[1]), ("cross", cross),
                             ("disattenuated", dis))}
        ceil = [np.sqrt(band["traj_self"][i] * band["coc_self"][i]) for i in (0, 1)]
        band["ceiling_share_of_log_fall"] = float(
            np.log(ceil[1] / ceil[0]) / np.log(band["cross"][1] / band["cross"][0]))
        band["by_layer"] = {"traj_self": same[0].tolist(), "coc_self": same[1].tolist(),
                            "cross": cross.tolist()}
        out[ax] = band
    return out


def kept_overlap(a, b, k):
    return np.array([len(set(np.argsort(-a[l])[:k]) & set(np.argsort(-b[l])[:k])) / k
                     for l in range(a.shape[0])])


def expert_block(z, step):
    out = {}
    for ax in ("q", "mlp"):
        v = np.sort(z[f"traj_exp_{ax}"], axis=1)  # (L, U) ascending within layer
        cum = np.cumsum(v, axis=1) / v.sum(1, keepdims=True)
        x = (np.arange(v.shape[1]) + 1) / v.shape[1]
        m = cum.mean(0)
        out[f"bottom50_{ax}"] = float(np.interp(0.5, x, m))
        out[f"curve_{ax}"] = [x[:: max(1, len(x) // 200)].tolist(), m[:: max(1, len(x) // 200)].tolist()]
        mass = step[f"{ax}_abs_step"].sum((1, 2))
        mass = mass / mass.sum()
        i = int(mass.argmin())
        out[f"steps_{ax}"] = {"mass": mass.tolist(), "argmin": i,
                              "first_over_min": float(mass[0] / mass[i]),
                              "last_over_min": float(mass[-1] / mass[i])}
    return out


def token_block(pq, pm, sp):
    """Additive token shares (rule 1 of finding1_numbers.py) and the hand-off layer."""
    out = {}
    lo, hi = sp["handoff"]
    for ax, arrs in (("q", (pq["ce"], pq["fm_full"])), ("mlp", (pm["ce_type"], pm["fm_type"]))):
        for obj, arr in zip(("coc", "traj"), arrs):
            part = np.zeros(arr.shape[1:3])  # (5, L)
            for i in range(len(arr)):
                g = arr[i].astype(np.float64)  # (5, L, U)
                part += (g * np.sign(g.sum(0, keepdims=True))).sum(-1)
            part /= len(arr)
            tot = part.sum(0)
            sh = np.nan_to_num(part / np.where(tot > 0, tot, np.nan))  # (5, L)
            vis = sh[VIS]
            out[f"{obj}_{ax}"] = {
                "share_by_type_and_layer": sh.tolist(),
                "vis_lo": float(np.mean(vis[sp["tok_lo"]])), "vis_hi": float(np.mean(vis[sp["tok_hi"]])),
                "text_lo": float(np.mean((sh[PT] + sh[COC])[sp["tok_lo"]])),
                "text_hi": float(np.mean((sh[PT] + sh[COC])[sp["tok_hi"]])),
                "bands": {b: {t: float(sh[k][sp[b]].mean()) for t, k in
                              (("vision", VIS), ("prompt", PT), ("coc", COC), ("hist", HIST))}
                          for b in ("trunk", "late")},
                "handoff": next((l for l in range(lo, hi) if (vis[l:hi] < 0.5).all()), None),
            }
    return out


def split_half(t, c, n_split, rng):
    n = len(t)
    st, sc_, cr = [], [], []
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = p[:n // 2], p[n // 2:]
        ta, tb, ca, cb = t[a].mean(0), t[b].mean(0), c[a].mean(0), c[b].mean(0)
        st.append(rho_per_layer(ta, tb))
        sc_.append(rho_per_layer(ca, cb))
        cr.append(0.5 * (rho_per_layer(ta, cb) + rho_per_layer(tb, ca)))
    return np.nanmean(st, 0), np.nanmean(sc_, 0), np.nanmean(cr, 0)


def same_token_block(pq, pm, sp):
    """finding1_numbers.py table D: raw and split-half-corrected agreement at the same positions."""
    out = {}
    L = sp["LAST"]
    for ax, fm, ce, n_split in (("q", pq["fm_full"], pq["ce"], 50), ("mlp", pm["fm_type"], pm["ce_type"], 10)):
        rng = np.random.default_rng(0)
        series = {"pooled": (np.abs(fm.sum(1)), np.abs(ce.sum(1)))}
        for name, ti in (("vision", VIS), ("prompt", PT), ("CoC", COC), ("hist", HIST)):
            series[name] = (np.abs(fm[:, ti]), np.abs(ce[:, ti]))
        res = {}
        for name, (t, c) in series.items():
            raw = rho_per_layer(t.mean(0), c.mean(0))
            st, sc_, cr = split_half(t[:, :, :], c[:, :, :], n_split, rng)
            st, sc_, cr = st[:L], sc_[:L], cr[:L]
            late = sp["late"]
            ok = (st[late] > 0.2) & (sc_[late] > 0.2)
            res[name] = {"raw_trunk": float(np.nanmean(raw[sp["trunk"]])),
                         "raw_late": float(np.nanmean(raw[late])),
                         "corrected_late": float(np.nanmean(cr[late][ok] / np.sqrt(st[late][ok] * sc_[late][ok])))
                         if ok.any() else None,
                         "late_layers_used": int(ok.sum()), "late_layers": int(late.stop - late.start)}
        out[ax] = res
    return out


def analyze(key, z, pc, pq, pm, step, n_layers, n_q, n_mlp):
    sp = spec(n_layers, n_q, n_mlp)
    m = {"spec": {k: (str(v) if isinstance(v, slice) else v) for k, v in sp.items()}}
    # G0a: I_traj identically zero at the last layer (its output reaches no cache)
    m["G0a_last_layer_traj_max"] = {ax: float(np.abs(pc[f"traj_vlm_{ax}"][:, -1]).max()) for ax in ("q", "mlp")}
    for ax in ("q", "mlp"):
        r = rho_per_layer(z[f"traj_vlm_{ax}"], z[f"coc_vlm_{ax}"])
        m[f"rho_{ax}"] = {"by_layer": r.tolist(), "trunk": float(np.nanmean(r[sp["trunk"]])),
                          "late": float(np.nanmean(r[sp["late"]]))}
        t, c = z[f"traj_vlm_{ax}"].sum(1), z[f"coc_vlm_{ax}"].sum(1)
        tn, cn = t / t.max(), c / c.max()
        Lr = np.arange(len(t))
        m[f"depth_{ax}"] = {"peak_traj": int(t.argmax()), "peak_coc": int(c.argmax()),
                            "com_traj": float((Lr * tn).sum() / tn.sum()),
                            "com_coc": float((Lr * cn).sum() / cn.sum()),
                            "traj_norm": tn.tolist(), "coc_norm": cn.tolist()}
        k = sp[f"keep_{ax}"]
        ov = kept_overlap(z[f"traj_vlm_{ax}"], z[f"coc_vlm_{ax}"], k)
        m[f"kept_overlap_{ax}"] = {"k": k, "chance": k / z[f"traj_vlm_{ax}"].shape[1],
                                   "trunk": float(np.mean(ov[sp["trunk"]])),
                                   "late": float(np.mean(ov[sp["late"]])), "by_layer": ov.tolist()}
        kv = {f"{o}_{kk}": (z[f"{o}_kv_{kk}"].sum(1) / z[f"{o}_kv_{kk}"].sum()).tolist()
              for o in ("traj", "coc") for kk in ("k", "v")}
        m["kv_profile"] = kv
    rng = np.random.default_rng(0)
    m["ratio"] = ratio_block(z, pc, sp, rng)
    m["ceiling"] = ceiling_block(pc, sp, np.random.default_rng(0))
    m["expert"] = expert_block(z, step)
    m["tokens"] = token_block(pq, pm, sp)
    m["same_token"] = same_token_block(pq, pm, sp)
    return m, sp


def flat(m):
    """The quantities REF and the verdicts read, under one set of names."""
    f = {}
    for ax in ("q", "mlp"):
        f[f"rho_{ax}_trunk"], f[f"rho_{ax}_late"] = m[f"rho_{ax}"]["trunk"], m[f"rho_{ax}"]["late"]
        r = m["ratio"][ax]
        f[f"tau_{ax}"], f[f"step_{ax}"], f[f"r2_{ax}"] = r["tau"], r["step"], r["r2"]
        f[f"disatt_late_{ax}"] = m["ceiling"][ax]["disattenuated"][1]
        f[f"ceiling_share_{ax}"] = m["ceiling"][ax]["ceiling_share_of_log_fall"]
        f[f"peak_traj_{ax}"], f[f"peak_coc_{ax}"] = m[f"depth_{ax}"]["peak_traj"], m[f"depth_{ax}"]["peak_coc"]
        f[f"exp_bottom50_{ax}"] = m["expert"][f"bottom50_{ax}"]
        for obj in ("traj", "coc"):
            tk = m["tokens"][f"{obj}_{ax}"]
            f[f"handoff_{obj}_{ax}"] = tk["handoff"]
            f[f"vis_lo_{obj}_{ax}"], f[f"vis_hi_{obj}_{ax}"] = tk["vis_lo"], tk["vis_hi"]
    f["same_vision_late_corr_q"] = m["same_token"]["q"]["vision"]["corrected_late"]
    f["same_pooled_late_corr_q"] = m["same_token"]["q"]["pooled"]["corrected_late"]
    return f


def verdicts(m, sp):
    """R1-R9 of the plan, each as (pass, the numbers it read)."""
    f = flat(m)
    v = {}
    v["R1"] = (all(f[f"rho_{a}_trunk"] >= 0.6 and f[f"rho_{a}_late"] <= f[f"rho_{a}_trunk"] - 0.2
                   for a in ("q", "mlp")),
               {a: [round(f[f"rho_{a}_trunk"], 3), round(f[f"rho_{a}_late"], 3)] for a in ("q", "mlp")})
    r = m["ratio"]
    v["R2"] = (all(r[a]["r2"] >= 0.7 and r[a]["boot_step_ci"][0] > 1.5 and r[a]["boot_tau_mode_share"] >= 0.5
                   and r[a]["clips_with_step"] >= 0.9 for a in ("q", "mlp")),
               {a: {"r2": round(r[a]["r2"], 3), "step_ci_lo": round(r[a]["boot_step_ci"][0], 2),
                    "tau_mode_share": round(r[a]["boot_tau_mode_share"], 3),
                    "clips_with_step": round(r[a]["clips_with_step"], 3)} for a in ("q", "mlp")})
    lo, hi = sp["tau_window"]
    v["R3"] = (all(lo <= r[a]["tau"] <= hi for a in ("q", "mlp")),
               {"window": [lo, hi], **{a: r[a]["tau"] for a in ("q", "mlp")}})
    c = m["ceiling"]["q"]
    v["R4"] = (c["disattenuated"][1] <= 0.6 and c["ceiling_share_of_log_fall"] <= 0.25,
               {"q_disatt_late": round(c["disattenuated"][1], 3),
                "q_ceiling_share": round(c["ceiling_share_of_log_fall"], 3),
                "mlp_disatt_late (report only)": round(m["ceiling"]["mlp"]["disattenuated"][1], 3)})
    e = m["expert"]
    v["R5"] = (e["bottom50_mlp"] < 0.05 and e["bottom50_q"] > 0.15,
               {"q": round(e["bottom50_q"], 4), "mlp": round(e["bottom50_mlp"], 4)})
    v["R6"] = (all(2 <= e[f"steps_{a}"]["argmin"] <= 7 and e[f"steps_{a}"]["first_over_min"] >= 1.3
                   and e[f"steps_{a}"]["last_over_min"] >= 1.3 for a in ("q", "mlp")),
               {a: {k: (round(x, 2) if isinstance(x, float) else x) for k, x in e[f"steps_{a}"].items()
                    if k != "mass"} for a in ("q", "mlp")})
    v["R7"] = (all(f[f"peak_traj_{a}"] < f[f"peak_coc_{a}"] for a in ("q", "mlp")),
               {a: [f[f"peak_traj_{a}"], f[f"peak_coc_{a}"]] for a in ("q", "mlp")})
    ok8, d8 = True, {}
    for a in ("q", "mlp"):
        for obj in ("traj", "coc"):
            b = m["tokens"][f"{obj}_{a}"]["bands"]
            h = m["tokens"][f"{obj}_{a}"]["handoff"]
            ok8 &= b["trunk"]["vision"] > 0.5 and b["late"]["vision"] < 0.5 and h is not None \
                and h < r[a]["tau"]
            d8[f"{obj}_{a}"] = {"trunk_vision": round(b["trunk"]["vision"], 3),
                                "late_vision": round(b["late"]["vision"], 3), "handoff": h,
                                "tau": r[a]["tau"]}
    v["R8"] = (bool(ok8), d8)
    st = m["same_token"]["q"]
    vis, pool = st["vision"]["corrected_late"], st["pooled"]["corrected_late"]
    v["R9"] = (vis is not None and pool is not None and vis >= 0.70 and vis >= pool + 0.2,
               {"q_vision_late_corr": vis, "q_pooled_late_corr": pool})
    return {k: {"pass": bool(p), "values": d} for k, (p, d) in v.items()}


def fmt(x):
    """A table cell: ints as is, floats to 4 places, None (e.g. no hand-off layer) as 'none'."""
    return "none" if x is None else str(x) if isinstance(x, int) else f"{x:.4f}"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(color=GRID, lw=1.0)
    ax.set_axisbelow(True)


def rel(n):
    return np.arange(n) / (n - 1)


def plots(res, out):
    out.mkdir(parents=True, exist_ok=True)
    axes_names = (("q", "Q heads"), ("mlp", "MLP channels"))

    def two(title, fname, draw, ylabel, logy=False, ylim=None):
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURFACE, sharey=True)
        for axp, (ax, label) in zip(axs, axes_names):
            style(axp)
            for key in ("a15", "a2"):
                draw(axp, key, ax)
            axp.set_title(label, loc="left", fontsize=10, color=INK)
            axp.set_xlabel("relative depth (layer / last layer)", fontsize=9, color=INK2)
            if logy:
                axp.set_yscale("log")
            if ylim:
                axp.set_ylim(*ylim)
        axs[0].set_ylabel(ylabel, fontsize=9, color=INK2)
        handles, labels = axs[0].get_legend_handles_labels()
        # below the panels: up to four entries do not fit beside the title
        fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=2,
                   labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0.12 if len(labels) > 2 else 0.07, 1, 0.94))
        fig.savefig(out / fname, dpi=150, facecolor=SURFACE)
        plt.close(fig)

    def late_band(axp, key):
        sp = res[key]["sp"]
        n = sp["N"]
        axp.axvspan(sp["late"].start / (n - 1), (sp["late"].stop - 1) / (n - 1),
                    color=COLOR[key], alpha=0.06, lw=0)

    def d_rho(axp, key, ax):
        r = np.array(res[key]["m"][f"rho_{ax}"]["by_layer"], dtype=float)
        late_band(axp, key)
        axp.plot(rel(len(r)), r, color=COLOR[key], lw=2, label=NAME[key])
    two("Within-layer rank agreement between I_traj and I_CoC (Spearman)", "rank_agreement.png",
        d_rho, "Spearman rho", ylim=(-0.25, 1.0))

    def d_ratio(axp, key, ax):
        rr = res[key]["m"]["ratio"][ax]
        y = np.array(rr["ratio_by_layer"])
        x = np.arange(len(y)) / res[key]["sp"]["LAST"]
        axp.plot(x, y, color=COLOR[key], lw=2, label=NAME[key])
        tau = rr["tau"]
        axp.hlines([rr["early"], rr["late"]], [x[0], x[tau]], [x[tau - 1], x[-1]], color=COLOR[key],
                   lw=1, ls=":")
        axp.annotate(f"tau={tau}, step {rr['step']:.1f}x", (x[tau], rr["late"]), xytext=(4, -12),
                     textcoords="offset points", fontsize=8, color=INK)
    two("Layer importance ratio I_traj / I_CoC with the two-level step fit", "ratio_step.png",
        d_ratio, "sum I_traj / sum I_CoC (log)", logy=True)

    for obj, lab in (("traj", "I_traj"), ("coc", "I_CoC")):
        def d_depth(axp, key, ax, obj=obj):
            y = np.array(res[key]["m"][f"depth_{ax}"][f"{obj}_norm"])
            axp.plot(rel(len(y)), y, color=COLOR[key], lw=2, label=NAME[key])
        two(f"Layer-summed {lab}, each normalised to its maximum", f"depth_{obj}.png", d_depth,
            "normalised importance", ylim=(0, 1.05))

    def d_ceiling(axp, key, ax):
        b = res[key]["m"]["ceiling"][ax]["by_layer"]
        x = rel(len(b["cross"]))
        axp.plot(x, b["cross"], color=COLOR[key], lw=2, label=f"{NAME[key]}: I_traj vs I_CoC")
        axp.plot(x, np.sqrt(np.array(b["traj_self"]) * np.array(b["coc_self"])), color=COLOR[key],
                 lw=1, ls="--", label=f"{NAME[key]}: ceiling (split-half)")
    two("Cross-objective agreement against its split-half ceiling", "ceiling.png", d_ceiling,
        "Spearman rho across halves", ylim=(-0.2, 1.02))

    for obj, lab in (("traj", "I_traj"), ("coc", "I_CoC")):
        def d_tok(axp, key, ax, obj=obj):
            sh = np.array(res[key]["m"]["tokens"][f"{obj}_{ax}"]["share_by_type_and_layer"])
            n = sh.shape[1] - (1 if obj == "traj" else 0)  # I_traj's last layer is 0/0
            axp.plot(rel(sh.shape[1])[:n], sh[VIS][:n], color=COLOR[key], lw=2, label=f"{NAME[key]}: vision")
            axp.plot(rel(sh.shape[1])[:n], (sh[PT] + sh[COC])[:n], color=COLOR[key], lw=1.2, ls="--",
                     label=f"{NAME[key]}: text (prompt + CoC)")
            axp.axhline(0.5, color=INK2, lw=0.8)
        two(f"Additive token-type share of {lab} per layer", f"token_share_{obj}.png", d_tok,
            "share of the layer's score", ylim=(-0.05, 1.05))

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURFACE)
    names = ("pooled", "vision", "prompt", "CoC", "hist")
    for axp, (ax, label) in zip(axs, axes_names):
        style(axp)
        w = 0.36
        for j, key in enumerate(("a15", "a2")):
            st = res[key]["m"]["same_token"][ax]
            vals = [st[n]["corrected_late"] if st[n]["corrected_late"] is not None else np.nan for n in names]
            xs = np.arange(len(names)) + (j - 0.5) * w
            axp.bar(xs, vals, width=w - 0.04, color=COLOR[key], label=NAME[key])
            for xv, yv in zip(xs, vals):
                if np.isfinite(yv):
                    axp.annotate(f"{yv:.2f}", (xv, yv), xytext=(0, 3), textcoords="offset points",
                                 ha="center", fontsize=7, color=INK)
        axp.set_xticks(np.arange(len(names)))
        axp.set_xticklabels(names, fontsize=8, color=INK)
        axp.set_ylim(0, 1.05)
        axp.set_title(f"{label}: late band, split-half corrected", loc="left", fontsize=10, color=INK)
    axs[0].set_ylabel("I_traj vs I_CoC agreement", fontsize=9, color=INK2)
    axs[0].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "same_token.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE, sharey=True)
    for axp, (ax, label) in zip(axs, axes_names):
        style(axp)
        for key in ("a15", "a2"):
            mass = res[key]["m"]["expert"][f"steps_{ax}"]["mass"]
            axp.plot(np.arange(len(mass)), mass, color=COLOR[key], lw=2, marker="o", ms=5,
                     mec=SURFACE, mew=1.2, label=NAME[key])
        axp.set_xticks(range(10))
        axp.set_xticklabels([f"{(s + 0.5) / 10:.2f}" for s in range(10)], fontsize=7)
        axp.set_xlabel("denoising time t (0 = noise, 1 = action)", fontsize=9, color=INK2)
        axp.set_title(f"expert {label}", loc="left", fontsize=10, color=INK)
    axs[0].set_ylabel("share of first-order mass", fontsize=9, color=INK2)
    axs[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "expert_steps.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    style(ax)
    for key in ("a15", "a2"):
        for a, ls in (("q", "--"), ("mlp", "-")):
            x, y = res[key]["m"]["expert"][f"curve_{a}"]
            ax.plot(x, y, color=COLOR[key], lw=2 if a == "mlp" else 1.2, ls=ls,
                    label=f"{NAME[key]}: {'MLP' if a == 'mlp' else 'Q heads'}")
    ax.axvline(0.5, color=INK2, lw=0.8)
    ax.set_xlabel("lowest-scoring fraction of the layer's expert units", fontsize=9, color=INK2)
    ax.set_ylabel("share of the layer's I_traj mass", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "expert_concentration.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def load_a15():
    z = dict(np.load(OUT15 / "importance_v2" / "importance.npz"))
    pc = dict(np.load(OUT15 / "importance_v2" / "importance_perclip.npz"))
    pq = dict(np.load(OUT15 / "gradanat_v1" / "anatomy_perclip_q.npz"))
    pm = dict(np.load(OUT15 / "gradanat_v1" / "anatomy_perclip_mlp.npz"))
    step = dict(np.load(OUT15 / "stepimp_fm_perstep_v2" / "step_importance.npz"))
    return z, pc, pq, pm, step


def load_a2(d):
    return tuple(dict(np.load(d / f)) for f in ("importance.npz", "importance_perclip.npz",
                                                  "anatomy_perclip_q.npz", "anatomy_perclip_mlp.npz",
                                                  "step_importance.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="a2_importance_v1")
    ap.add_argument("--exp-id", default="a2_importance_analysis")
    ap.add_argument("--a15-only", action="store_true", help="G0c only: reproduce the 1.5 numbers")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)

    res = {}
    z, pc, pq, pm, step = load_a15()
    m15, sp15 = analyze("a15", z, pc, pq, pm, step, 36, 32, 12288)
    res["a15"] = {"m": m15, "sp": sp15}
    del z, pc, pq, pm
    f = flat(m15)
    g0c = {k: {"got": f[k], "ref": ref, "ok": abs(f[k] - ref) <= tol} for k, (ref, tol) in REF.items()}
    bad = [k for k, v in g0c.items() if not v["ok"]]
    print(f"G0c (1.5 reproduces the published numbers): {len(REF) - len(bad)}/{len(REF)} "
          f"{'PASS' if not bad else 'FAIL ' + str({k: (round(g0c[k]['got'], 4), g0c[k]['ref']) for k in bad})}")
    if args.a15_only:
        (out / "g0c.json").write_text(json.dumps(g0c, indent=2))
        sys.exit(1 if bad else 0)
    if bad:
        sys.exit("G0c failed: the analysis code does not reproduce Alpamayo 1.5; not reading Alpamayo 2")

    run = REPO / "outputs" / args.run
    z, pc, pq, pm, step = load_a2(run)
    n, nq, nm = z["traj_vlm_q"].shape[0], z["traj_vlm_q"].shape[1], z["traj_vlm_mlp"].shape[1]
    m2, sp2 = analyze("a2", z, pc, pq, pm, step, n, nq, nm)
    res["a2"] = {"m": m2, "sp": sp2}
    g0a = {k: v for key in ("a15", "a2") for k, v in
           ((f"{key}_{ax}", res[key]["m"]["G0a_last_layer_traj_max"][ax]) for ax in ("q", "mlp"))}
    v15, v2 = verdicts(m15, sp15), verdicts(m2, sp2)
    metrics = {"G0a_last_layer_traj_max": g0a, "G0c": g0c, "verdicts": {"a15": v15, "a2": v2},
               "a15": m15, "a2": m2, "n_clips": {"a2": int(pc["traj_vlm_q"].shape[0])}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "config.json").write_text(json.dumps({
        "plan": "plans/2026-09-27_a2-dual-importance.md", "run_a2": str(run),
        "runs_a15": ["importance_v2", "gradanat_v1", "stepimp_fm_perstep_v2"],
        "spec_a15": m15["spec"], "spec_a2": m2["spec"]}, indent=2))
    plots(res, out / "plots")

    f2 = flat(m2)
    lines = [f"Finding 1: Alpamayo 1.5 vs Alpamayo 2 Super (A2 n={metrics['n_clips']['a2']} clips of calib_100)",
             f"G0a last-layer I_traj max: {g0a}", f"G0c: {len(REF) - len(bad)}/{len(REF)} reproduced", "",
             f"{'quantity':28s} {'A1.5':>10s} {'A2':>10s}"]
    for k in REF:
        a, b = f[k], f2[k]
        lines.append(f"{k:28s} {fmt(a):>10s} {fmt(b):>10s}")
    lines += ["", f"{'verdict':6s} {'A1.5':>6s} {'A2':>6s}  A2 values"]
    for k in v2:
        lines.append(f"{k:6s} {v15[k]['pass']!s:>6s} {v2[k]['pass']!s:>6s}  {json.dumps(v2[k]['values'])}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
