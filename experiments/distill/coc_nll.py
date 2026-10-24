"""CoC NLL: how likely a model finds the human-refined Chain-of-Causation of the OOD clips.

The ood_262 manifest carries `gt_coc` (PhysicalAI-AV reasoning/ood_reasoning.parquet,
human-verified). Each sentence is tokenized with the teacher tokenizer and appended to the
driving prompt (same images, history and question as training), and the teacher-forced
negative log-likelihood of its tokens is recorded per clip (nats per token, natural log).
A student cannot emit ids it pruned from its vocabulary, so a clip whose CoC contains one is
flagged `oov` and skipped for that student. Students normalise over their pruned vocabulary.
A student with a LiDAR encoder gets the range-image tokens as in training; clips without a LiDAR
cache are skipped for it. --compare scores the clips both runs scored.

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=0,1 python experiments/distill/coc_nll.py --tag a2
  CUDA_VISIBLE_DEVICES=2 python experiments/distill/coc_nll.py --tag s1 \
      --student /mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/s1/distill_s1/best
  CUDA_VISIBLE_DEVICES=0 python experiments/distill/coc_nll.py --tag lidar_s2 \
      --student /mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/s2_wide/distill_s2_wide_lidar/best
  python experiments/distill/coc_nll.py --compare a2 s1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parents[2]


def run(args, out_dir):
    import build_lidar_distill as bl
    import distill_cache as dc
    import distill_data as dd
    import lidar_lib as ll
    import torch
    from importance_distill import load_model
    from run_oodval import set_determinism

    set_determinism()
    enc = None
    if args.student is None:
        model, vmap, where = load_model(None)
    else:
        from eval_student import load_student

        model, vmap, _, enc = load_student(args.student, "cuda:0")
        where = f"student {args.student}" + (" + LiDAR" if enc is not None else "")
    builder = dd.SeqBuilder(vocab_map=vmap)
    man = pd.read_parquet(Path(args.repo) / "outputs" / "distill_sets" / f"{args.set}.parquet")
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for j, r in enumerate(man.itertuples()):
            ids = builder.tokenizer(r.gt_coc, add_special_tokens=False)["input_ids"]
            lpath = bl.path_for(r.clip_id, r.t0_us, args.set)
            row = {"clip_id": r.clip_id, "t0_us": int(r.t0_us), "n_tok": len(ids),
                   "oov": bool(vmap is not None and (vmap[ids] < 0).any()),
                   "no_lidar": enc is not None and not lpath.exists()}
            if not (row["oov"] or row["no_lidar"]):
                data = dc.load_cached(dc.path_for(r.clip_id, r.t0_us, args.set))
                seq = builder.driving(data, ids, lidar=enc is not None)
                x = seq["input_ids"].to("cuda:0")
                n = seq["n_target"]
                vis = {k: seq[k].to("cuda:0") for k in dd.VISION_KEYS}
                if enc is None:
                    out = model.vlm(input_ids=x, attention_mask=seq["attention_mask"].to("cuda:0"),
                                    logits_to_keep=n + 1, **vis)
                else:
                    out = ll.lidar_vlm_forward(model, enc, x, seq["attention_mask"].to("cuda:0"),
                                               vis["pixel_values"], vis["image_grid_thw"],
                                               ll.lidar_tensors(bl.load_cached(lpath), "cuda:0"),
                                               logits_to_keep=n + 1)
                logits = out.logits[0, :-1].float()  # (n, V)
                nll = torch.nn.functional.cross_entropy(logits, x[0, -n:].to(logits.device),
                                                        reduction="none")  # (n,)
                row.update(nll_sum=float(nll.sum()), nll_mean=float(nll.mean()))
                del out
            rows.append(row)
            if (j + 1) % 25 == 0:
                print(f"{args.tag} [{j + 1}/{len(man)}]", flush=True)
    df = pd.DataFrame(rows)
    ok = df[~(df.oov | df.no_lidar)]
    summary = {"tag": args.tag, "model": where, "set": args.set, "n_clips": len(df),
               "n_scored": len(ok), "n_oov": int(df.oov.sum()), "n_no_lidar": int(df.no_lidar.sum()),
               "nll_per_token_clip_mean": float(ok.nll_mean.mean()),
               "nll_per_token_token_mean": float(ok.nll_sum.sum() / ok.n_tok.sum())}
    (out_dir / f"{args.tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=2))


def compare(a, b, out_dir):
    from analyze_p4 import boot_ci

    da, db = (pd.DataFrame(json.loads((out_dir / f"{t}.json").read_text())["rows"]) for t in (a, b))
    m = da.merge(db, on=["clip_id", "t0_us"], suffixes=(f"_{a}", f"_{b}"))
    m = m[m[f"nll_mean_{a}"].notna() & m[f"nll_mean_{b}"].notna()]
    diff = (m[f"nll_mean_{b}"] - m[f"nll_mean_{a}"]).to_numpy()
    res = {"n_common": len(m), "n_excluded": int(len(da) - len(m)),
           a: {"nll_per_token": float(m[f"nll_mean_{a}"].mean()),
               "ppl": float(np.exp(m[f"nll_sum_{a}"].sum() / m[f"n_tok_{a}"].sum()))},
           b: {"nll_per_token": float(m[f"nll_mean_{b}"].mean()),
               "ppl": float(np.exp(m[f"nll_sum_{b}"].sum() / m[f"n_tok_{b}"].sum()))},
           f"{b}-{a} nll_per_token": {"mean": float(diff.mean()), "ci95": boot_ci(diff)}}
    (out_dir / f"compare_{a}_{b}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--student", default=None, help="student checkpoint dir (default: teacher)")
    ap.add_argument("--set", default="ood_262")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--repo", default=str(REPO), help="checkout whose outputs/ holds the results")
    args = ap.parse_args()
    out_dir = Path(args.repo) / "outputs" / "distill_coc_nll"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.compare:
        compare(*args.compare, out_dir)
    else:
        run(args, out_dir)


if __name__ == "__main__":
    main()
