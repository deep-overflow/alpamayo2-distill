# Alpamayo 2 Super on OOD-val 262: open-loop eval + visualization vs Alpamayo 1.5

## Purpose

Measure Alpamayo 2 Super (34B = Qwen3-VL-32B + 2B expert) open-loop on the same
OOD-val 262 clips the compression track uses as its primary long-tail set, and
compare it clip-by-clip with the Alpamayo-1.5-10B baseline already on disk.

Question: how much better (or not) is A2 than A1.5 on long-tail driving, where
does the gap come from (bucket / event cluster / horizon), and what do the
failures look like. This is a **reference measurement for the distillation
track** (teacher = A2), not a method comparison.

## What is already on disk (verified 2026-09-20)

| item | location | note |
|---|---|---|
| OOD-val manifest | `/mnt/nvme1n1/ad_vla/outputs/chan/eval_sets/ood_val.parquet` | 262 rows: `clip_id, t0_us (event time), gt_coc, cluster, split, chunk` |
| A1.5 rows with waypoints | `.../outputs/chan/baseline_pred_oodval/baseline_s{0..3}of4.json` | 262/262, `pred_xy_k` (8, 64, 2), `ade_rollout_k`, `gen_coc`, TF condition too |
| A1.5 frame cache | `.../pre_processed/ood/samples/*.npz` | **4 cameras** `[0,1,2,6]` x 4 frames, JPEG |
| A1.5 viz script | compression repo branch `worktree-ood-val-viz`, `viz_openloop_ood.py` | front cam + BEV + CoC, best-of-6 |

A1.5 reference numbers recomputed from those rows (rollout condition, n=262):

| | mean | median |
|---|---:|---:|
| minADE@6 (published reduction) | 0.9995 | 0.5745 |
| minFDE@6 | 2.4959 | 1.1184 |
| minADE@8 | 0.9214 | 0.5272 |
| ADE@1 | 2.0897 | 1.4440 |

Buckets: cruise 140 / decel_stop 51 / accel 42 / turn 29.

## What is missing

1. **Weights**: `nvidia/Alpamayo2-Super` is not in the HF cache. Repo is reachable
   (not gated for this account), 71.6 GB / 15 shards, sha `00554695`.
   `/mnt/nvme1n1` has 345 GB free (95% used) -> fits, but it is a shared disk.
2. **Frames**: A2's trajectory profile is **6 cameras `[0,1,2,3,5,6]` x 4 frames**
   (adds rear-left/rear-right 70fov). The existing OOD cache has 4 cameras, so the
   262 clips must be re-streamed from HF at their event `t0_us`.
3. **Environment**: no venv for this repo yet.
4. **Single-GPU fit**: 71.6 GB bf16 does not fit a 48 GB card -> multi-GPU placement.

## Protocol (matched to `run_baseline.py`, the A1.5 runner)

Per clip:

```
seed_all(clip_seed(42, clip_id))                     # sha256 rule, same as A1.5
1 CoC rollout   (do_sample, temperature 0.6, top_p 0.98, max_new_tokens 256)
K=8 denoisings  on that single CoC's KV cache, seed = base + k, batch 1 each
ADE/FDE per sample vs ego_future_xyz[..., :2]   (64 steps, 6.4 s @ 10 Hz)
```

- Reductions reported: **minADE@6 / minFDE@6 primary** (prefix of the 8, the
  published reduction), @8 and @1 alongside; horizons 1.6 / 3.2 / 6.4 s.
- A2's public API (`sample_trajectories_from_data(num_traj_samples=8)`) draws 8
  *independent CoCs*, which is a different estimator (CoC diversity lowers minADE
  for free). So the runner re-implements the sampling loop **outside the model
  source**: same prefill/generate/expert calls as `sample_trajectories_from_data`,
  but one generation and a loop of 8 seeded `diffusion.sample(batch_size=1)` over
  the same cache (the native `step_fn` already crops the cache back each step).
- Diffusion steps: A2 native default (10, euler). A1.5 rows used its own default.
- Row schema identical to `run_baseline.py --save-pred` (`minADE_rollout`,
  `ade_rollout_k`, `fde_rollout_k`, `*_h16/_h32`, `pred_xy_k`, `gen_coc`, `gen_len`,
  `coc_*`, `bucket`, `cluster`, `gt_coc`, `seed`) so both models join on `clip_id`.
  `bucket`, `ade_fde`, `coc_degenerate`, `paired_bootstrap_ci` are imported from the
  compression repo's `eval_lib.py` (numpy-only) so definitions cannot drift.
- Optional GT-CoC teacher-forced condition (`minADE_tf`, `nll_gtcoc`): A1.5 has it.
  **Open decision 2.**

### Confounds / caveats (stated, not removable)

- A2 sees 6 cameras, A1.5 sees 4. Inherent to each model's input contract.
- Seeds pair nothing across two different models; pairing is on clips only. Each
  model gets one CoC draw per clip, so per-clip differences are noisy; the
  aggregate paired statistics are the reading, not single clips.
- A2's training set is not public at clip level. The 262 are official `val`, but
  leakage cannot be excluded from here. Report as a caveat.
- Frames go through the same JPEG round-trip as the A1.5 cache (same encoder
  settings as `build_ood_cache.py`), so input degradation is matched.
- A1.5 rows were produced on RTX 5880 Ada; irrelevant across models, recorded anyway.

## Implementation

New code only under `experiments/oodval/` (no edits to `src/`):

| file | role |
|---|---|
| `build_cache_a2.py` | stream 262 clips via the official `load_physical_aiavdataset` (7-cam ring, `t0_us` = event time, calibration best-effort), write one npz per clip: 28 JPEGs + ego hist/future + timestamps (+ pickled calibration). Resumable, 16 workers. Cache at `/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/ood_a2/` (~1.5 GB). Gate: ego tensors bitwise equal to the existing 4-cam cache for the same clip. |
| `run_oodval.py` | the runner above. `--shard/--n-shards`, `--limit`, `--gpus a,b[,c]`, resumable rows file, `config.json` (model sha, GPUs, placement, seeds, determinism flags), per-shard summary. |
| `analyze_oodval.py` | merge shards, join A1.5 rows, paired delta (mean, median, 10k bootstrap CI, Wilcoxon, win rate) overall / per bucket / per cluster / per horizon / K in {1,6,8}; CoC degeneracy + length; `metrics.json`, `summary.txt`, plots (paired scatter log-log, delta ECDF, per-bucket and per-cluster bars with CI, horizon curve). |
| `viz_oodval.py` | per-clip figure in the style of the existing open-loop viz: front camera at t0, BEV rows `A1.5` / `A2` on a shared axis box (y inverted so left is left), GT path, 6 samples with best highlighted, three CoC lines (reference / A1.5 / A2). Renders all 262 + `worst10_a2`, `biggest_gain`, `biggest_loss` index lists. |

Outputs: `outputs/a2_oodval/{config.json,metrics.json,summary.txt,plots/,viz/}` (gitignored).
Report: `reports/2026-09-20_alpamayo2-vs-alpamayo1.5-oodval.md` (facts / analysis split).

### Multi-GPU placement

VLM and expert both have 64 layers and expert layer *i* attends to VLM KV-cache
layer *i*, so a naive `device_map="auto"` breaks (README warns the same). Plan:
custom `device_map` that splits by layer index and **co-locates
`expert.expert.layers[i]` with `vlm...layers[i]`**; visual tower, embeddings and the
first (deepstack) layers on the first GPU; CUDA graphs off. With K handled as
batch-1 loops the KV cache is ~1.3 GB, so 2 x 48 GB per replica should fit
(~37 GB weights per card). Fallback if it does not: 3 GPUs per replica, or the
demo's layout (VLM split over 2 cards, expert + moved cache on a third).

### Environment

`uv sync --locked` into `/mnt/nvme1n1/ad_vla/venvs/alpamayo2`, symlinked as `.venv`
— the same convention as `alpamayo-model-compression/.venv -> venvs/alpamayo-mc`.
The pins (torch 2.8.0, transformers 4.57.1, flash-attn 2.8.3, physical-ai-av 0.2.0)
are identical to `alpamayo-mc`, so the flash-attn wheel should come from the uv
cache rather than a rebuild. Nothing is installed into any existing venv. Note:
`HF_HUB_ENABLE_HF_TRANSFER=1` is exported on this box but `hf_transfer` is not in
every venv; the download step sets it explicitly.
(Global rule says Docker; no Docker workflow exists for these sibling projects on
this host, so this follows the established venv convention. Say so if you want a
container instead.)

## Steps and gates

1. venv + `hf download nvidia/Alpamayo2-Super` (72 GB) — in parallel with 2.
2. Build the 6-cam OOD-val cache (est. 20-30 min, 16 workers). Gate: 262/262, ego
   tensors match the 4-cam cache bitwise.
3. Smoke: README default clip (`030c760c`, t0 5.1 s) through the multi-GPU
   placement -> finite trajectory, sensible CoC, minADE in a plausible range; then
   5 OOD-val clips: shapes, NaN, CoC length, peak memory per card, s/clip.
4. Determinism check: same 5 clips twice -> identical rows (same GPU arch).
5. Full run, sharded over the replicas; monitor first iterations, GPU util/mem, logs.
6. `analyze_oodval.py` -> `viz_oodval.py` -> report. `ruff check` on the new files.

Cost estimate: eval ~15-25 s/clip -> 262 clips in ~45-60 min on 2 replicas; with
the TF condition roughly 1.6x.

## Deviations from this plan (recorded after the run)

- Step 3's README-clip smoke was replaced by stronger gates on OOD-val clips: bitwise
  parity of the re-implemented sampling loop against the release
  `sample_trajectories_from_data` (`check_native_parity.py`), identical output under a
  2-card vs 3-card split, and identical output on a rerun on other cards.
- The runner takes its cards from `CUDA_VISIBLE_DEVICES` in the shell instead of a
  `--gpus` argument: the imports initialise CUDA, so setting it inside `main()` is
  ignored and the process lands on card 0.
- HF auth comes from `experiments/oodval/env.sh` (same exports as the compression
  repo's `run_retry_host.sh`); the ambient `HF_HOME` has no access to the gated dataset.
- Camera calibration is not cached; the per-clip figures are BEV-based and need none.
- Added to the analysis after seeing the first results: average-over-8 ADE, endpoint
  spread, and distance-travelled (longitudinal) error.

## Decisions (approved 2026-09-20)

1. **Sampling protocol** — matched: 1 CoC x 8 traj. A2-native (8 independent
   CoC+traj) is not run.
2. **GT-CoC teacher-forced condition** — skipped in this pass; rollout only. Rows are
   resumable per clip, so it can be added later without re-running the rollout.
3. **GPUs** — 4-7 (RTX 5880 Ada), 2 replicas x 2 cards. 0-3 stay free.
4. Branch `oodval-eval`; no commit / PR until asked.
