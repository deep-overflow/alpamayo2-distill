"""P2 depth importance: task-loss increase when one decoder layer is removed from both towers.

Block Influence (importance_distill.py) ranks layers by how much they rotate the hidden
state, and A2's few huge residual channels make most middle layers look like identities by
that measure. This is the direct check the plan calls for: for each layer i, the teacher runs
with VLM layer i and expert layer i bypassed (their forward returns the input; expert layer i
reads VLM cache i, so the pair goes together) and the teacher-forced losses on its own labels
are compared with the intact model on the same clips:

  driving CE (CoC + meta-action tokens), det CE (detection JSON), expert flow-matching loss
  (n draws per clip with fixed seeds, identical across layers)

Layers are split over `--n-shards` replicas; `--merge` combines the shard files.

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=0,1 python experiments/distill/depth_importance.py --shard 0 --n-shards 2 &
  CUDA_VISIBLE_DEVICES=2,3 python experiments/distill/depth_importance.py --shard 1 --n-shards 2
  python experiments/distill/depth_importance.py --merge
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_depth_importance"  # --exp-id overrides
LOSSES = ("driving_ce", "det_ce", "fm")


def skip_hook(module, args, kwargs, out):
    h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
    o = out[0] if isinstance(out, tuple) else out
    h = h.to(o.device)
    return (h, *out[1:]) if isinstance(out, tuple) else h


def clip_losses(model, builder, item, n_fm, seed):
    import distill_cache as dc
    import distill_data as dd
    import torch
    from run_oodval import seed_all

    clip_id, t0_us = item
    data = dc.load_cached(dc.path_for(clip_id, t0_us, "train_10k"))
    res = {}
    for task in ("driving", "det"):
        lab = dd.load_label("train_10k", task, clip_id, t0_us)
        seq = (builder.driving(data, lab["gen_ids"]) if task == "driving"
               else builder.det(data, lab["gen_ids"]))
        ids = seq["input_ids"].to("cuda:0")
        n = seq["n_target"]
        out = model.vlm(input_ids=ids, attention_mask=seq["attention_mask"].to("cuda:0"),
                        use_cache=(task == "driving"), logits_to_keep=n + 1,
                        **{k: seq[k].to("cuda:0") for k in dd.VISION_KEYS})
        logits = out.logits[0, :-1].float()
        res[f"{task}_ce"] = float(torch.nn.functional.cross_entropy(
            logits, ids[0, -n:].to(logits.device)))
        if task == "driving":
            traj = {k: v.to("cuda:0") for k, v in seq["traj"].items()}
            fms = []
            for k in range(n_fm):
                seed_all(seed + k)
                fms.append(float(dd.expert_fm(model, traj, out)))
            res["fm"] = float(np.mean(fms))
        del out
    return res


def run(args):
    import distill_data as dd
    import torch
    from importance_distill import calib_items, load_model
    from run_oodval import set_determinism

    set_determinism()
    items = calib_items(args.n, args.seed)
    model, vmap, per_gpu = load_model(args.student)
    builder = dd.SeqBuilder(vocab_map=vmap)
    vlm_layers = model.vlm.model.language_model.layers
    exp_layers = model.expert.expert.layers
    layers = [i for i in range(len(vlm_layers)) if i % args.n_shards == args.shard]
    rows, t0 = [], time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for drop in [None, *layers]:
            handles = []
            if drop is not None:
                handles = [vlm_layers[drop].register_forward_hook(skip_hook, with_kwargs=True),
                           exp_layers[drop].register_forward_hook(skip_hook, with_kwargs=True)]
            per = [clip_losses(model, builder, it, args.n_fm, 1000 * j) for j, it in enumerate(items)]
            for h in handles:
                h.remove()
            rows.append({"drop": drop, **{k: float(np.mean([p[k] for p in per])) for k in LOSSES},
                         "per_clip": per})
            print(f"drop {drop}: " + ", ".join(f"{k} {rows[-1][k]:.4f}" for k in LOSSES)
                  + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"shard{args.shard}of{args.n_shards}.json").write_text(json.dumps({
                "items": items, "layers_per_card": per_gpu, "rows": rows}, indent=1))


def merge():
    shards = [json.loads(p.read_text()) for p in sorted(OUT.glob("shard*of*.json"))]
    base = {k: np.mean([s["rows"][0][k] for s in shards]) for k in LOSSES}
    drops = sorted((r for s in shards for r in s["rows"][1:]), key=lambda r: r["drop"])
    L = len(drops)
    delta = {k: np.array([r[k] - base[k] for r in drops]) for k in LOSSES}  # (L,)
    rel = {k: delta[k] / base[k] for k in LOSSES}
    score = np.mean([rel[k] for k in LOSSES], axis=0)  # mean relative loss increase
    np.savez(OUT / "depth.npz", layers=np.array([r["drop"] for r in drops]), score=score,
             **{f"delta_{k}": v for k, v in delta.items()}, **{f"base_{k}": base[k] for k in LOSSES})
    metrics = {"base": base, "n_layers": L, "n_clips": len(shards[0]["items"]),
               "score_rank_low_to_high": [int(drops[i]["drop"]) for i in np.argsort(score)],
               **{f"delta_{k}": delta[k].round(5).tolist() for k in LOSSES}}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (OUT / "config.json").write_text(json.dumps({
        "what": "drop-one-layer (both towers) teacher-forced loss increase",
        "calib": "importance_distill.calib_items, train_10k", "n_clips": len(shards[0]["items"]),
        "score": "mean over driving CE, det CE, FM of (loss_without_layer - base) / base"}, indent=2))
    lines = [f"drop-one depth importance, {L} layers, {metrics['n_clips']} clips",
             "base: " + ", ".join(f"{k} {v:.4f}" for k, v in base.items()),
             "layer  d_drvCE   d_detCE   d_FM     score"]
    lines += [f"{int(r['drop']):5d} {delta['driving_ce'][i]:+8.4f} {delta['det_ce'][i]:+8.4f} "
              f"{delta['fm'][i]:+8.4f} {score[i]:+8.4f}" for i, r in enumerate(drops)]
    lines.append("least harmful to drop (low score first): "
                 + str(metrics["score_rank_low_to_high"][:32]))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _plot(drops, delta, score)


def _plot(drops, delta, score):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["drop"] for r in drops]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax, (name, v) in zip(axes, [*delta.items(), ("score (mean rel.)", score)]):
        ax.bar(x, v, color="#2a78d6")
        ax.set_ylabel(name)
        ax.set_yscale("symlog", linthresh=1e-3)
    axes[-1].set_xlabel("dropped layer (both towers)")
    fig.tight_layout()
    fig.savefig(OUT / "depth_drop_one.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--n-fm", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--student", default=None, help="student checkpoint dir (default: teacher)")
    ap.add_argument("--exp-id", default="distill_depth_importance")
    args = ap.parse_args()
    global OUT
    OUT = REPO / "outputs" / args.exp_id
    if args.merge:
        merge()
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("set CUDA_VISIBLE_DEVICES")
    run(args)


if __name__ == "__main__":
    main()
