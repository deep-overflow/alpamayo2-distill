# Alpamayo 1.5 vs Alpamayo 2 Super — the seven `1_main` figures, side by side

The seven analysis figures of `alpamayo-model-compression/paper-analysis/figures/1_main`, each
rendered twice with the same code and placed next to each other: **Alpamayo 1.5 on the left,
Alpamayo 2 Super on the right**. Rendered by `experiments/importance/paper_figs_a2.py`
(rules copied from the originals' scripts; see its docstring).

| file | what it shows | original |
|---|---|---|
| `fig4_same_position_q_head` | within-layer rank agreement of `I_traj` vs `I_CoC`, Q heads, pooled ("All") and with both scores restricted to one position type (raw, 100-clip means) | `paper_same_position.py` |
| `fig4_token_share_traj_q_head_split` | additive share of each layer's `I_traj` earned at vision / prompt-text / generated-CoC / ego-history positions, Q heads | `paper_token_share.py … q split` |
| `fig4_token_share_coc_q_head_split` | the same for `I_CoC` | 〃 |
| `fig6e_head_map_combined_sortq` | per-unit maps: dominant position type (alpha = its share) and rank in layer for `I_FM` / `I_CE`, Q heads (top) and MLP channels (bottom); rank panels ordered by the `I_FM` rank | `head_map_combined.py … sortq` |
| `fig2_mass_curve` | expert: cumulative first-order mass removed when units are removed lowest-score-first, Q heads vs MLP channels; marker at 90% of units | `fig_criterion.py figure2` |
| `fig2_steps_q_head`, `fig2_steps_mlp` | expert: rank agreement of per-step importance between the ten denoising steps | 〃 |

## Conventions that differ between the two panels

- **Layer axis** is each model's own index: 0–35 (1.5) and 0–63 (2 Super).
- **Grey band** = the paper's late band 22–34 for 1.5; for 2 Super the same band moved by
  relative depth, 39–62 (`plans/2026-09-27_a2-dual-importance.md`). The last layer is left out
  because `I_traj` is identically zero there.
- **Dotted vertical line** = the fitted two-level step of `log(ΣI_traj / ΣI_CoC)`: layer 23 for
  1.5 (both axes); layer 51 (Q heads) / 50 (MLP channels) for 2 Super. In the head map it is the
  dotted horizontal line; the dashed one is the band boundary.
- Q-head panels of the head map separate the KV groups with thin white lines: 4 heads per group
  in 1.5, 8 in 2 Super (both have 8 KV heads).

## Data

- 1.5: `outputs/chan/gradanat_v1/anatomy_perclip_{q,mlp}.npz` (typed per-clip gradients, the
  source of the paper's fig4/fig6) and `outputs/chan/stepimp_fm_perstep_v2/step_importance.npz`
  (per-step expert gradients, the source of fig2). The 1.5 panels reproduce the originals
  (e.g. bottom-90% mass Q 68.5% / MLP 1.8%, step-to-step ρ 0.668 / 0.874).
- 2 Super: `outputs/a2_importance_v1/` (`run_importance_a2.py`, calib_100, 2026-09-27).

## Numbers printed while rendering

| | Alpamayo 1.5 | Alpamayo 2 Super |
|---|---|---|
| same-position agreement, late band, All → Vision / Prompt / CoC / Ego | 0.33 → 0.72 / 0.75 / 0.36 / 0.86 | 0.54 → 0.66 / 0.69 / 0.50 / 0.83 |
| expert bottom-90% units carry (Q / MLP) | 68.5% / 1.8% | 66.0% / 1.7% |
| expert step-to-step ρ, mean (min), Q heads | 0.668 (0.138) | 0.673 (0.209) |
| expert step-to-step ρ, mean (min), MLP | 0.874 (0.739) | 0.828 (0.661) |
