"""Turn importance statistics into a PruneSpec, slice the source checkpoint, save the student.

Scores (importance_distill.py stats, each vector divided by its mean so tasks weigh alike):
  VLM tower    average of driving and det, each the mean of the all-position and the
               generated-position statistics
  expert tower the expert task
Selection, per kept layer unless noted:
  layers   0, 1, 2 always (deepstack), then the highest drop-one task-loss increase
           (--depth, depth_importance.py) or, without it, the highest Block Influence
  kv       highest summed head score of the group's query heads (VLM + expert)
  q, q_e   highest head scores inside each kept group
  mlp      highest MLP channel scores
  d, d_e   highest residual-channel scores summed over the kept layers (+ final norm)
  vocab    required ids + every prompt id + every teacher-generated id + the teacher's top-8
           candidates on --vocab-sets (only when slicing from the teacher)

Presets follow the plan (section 4.3); S1 uses 16 Q / 4 KV (4 query heads per group) so
that every S2 candidate can be cut from it (a group can only lose heads).

Usage:
  python experiments/distill/make_student.py --arch S1 --name s1
  python experiments/distill/make_student.py --arch C-balanced --name s2_balanced \
      --source /mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/s1/final \
      --stats outputs/distill_importance_s1/stats.npz
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import slice_lib as pl

REPO = Path(__file__).resolve().parents[2]
CKPT = Path("/mnt/nvme1n1/ad_vla/outputs/chan/a2_distill")
ARCHS = {  # L, d, nQ, nKV, m, d_e, nQ_e, m_e
    "S1": (32, 2048, 16, 4, 8192, 768, 8, 3072),
    "C-deep": (28, 896, 8, 4, 3584, 384, 4, 1536),
    "C-balanced": (20, 1024, 16, 4, 4096, 384, 4, 1536),
    "C-wide": (14, 1280, 16, 4, 5120, 512, 4, 2048),
    # the plan's sizes assumed V' ~ 24k; at V' = 4,885 the MLPs grow to the same ~466M each
    "C-deep-v2": (28, 896, 8, 4, 4096, 384, 4, 1536),
    "C-balanced-v2": (20, 1024, 16, 4, 4864, 384, 4, 1536),
    "C-wide-v2": (14, 1280, 16, 4, 5632, 512, 4, 2048),
}


def nrm(x):
    return x / (np.mean(x) + 1e-12)


def vlm_score(st, key):
    """(layers, ...) VLM score: driving and det, all and target positions, equally weighted."""
    parts = [nrm(st[f"{t}/vlm/{s}/{key}"]) for t in ("driving", "det") for s in ("all", "tgt")]
    return np.mean(parts, axis=0)


def build_spec(st, src_cfg, arch, depth=None):
    L, d, nq, nkv, m, d_e, nq_e, m_e = arch
    t, e = src_cfg["vlm_config"]["text_config"], src_cfg["expert_config"]["llm_config"]
    rep, rep_e = t["num_attention_heads"] // t["num_key_value_heads"], \
        e["num_attention_heads"] // e["num_key_value_heads"]
    per, per_e = nq // nkv, nq_e // nkv
    assert per * nkv == nq and per_e * nkv == nq_e and per <= rep and per_e <= rep_e, \
        "target heads per KV group must divide evenly and not exceed the source's"

    bi = nrm(vlm_score(st, "bi")) + nrm(st["expert/exp/all/bi"])  # (L_src,)
    if depth is None:  # Block Influence only
        rank = np.argsort(-bi)
    else:  # drop-one task-loss increase (depth_importance.py), BI breaks ties
        score = np.full(len(bi), np.inf)
        score[depth["layers"]] = depth["score"]
        rank = np.lexsort((-bi, -score))
    rest = [i for i in rank if i not in (0, 1, 2)]
    layers = sorted([0, 1, 2] + [int(i) for i in rest[:L - 3]])

    head, head_e = vlm_score(st, "head"), nrm(st["expert/exp/all/head"])  # (L_src, nQ)
    mlp, mlp_e = vlm_score(st, "mlp"), nrm(st["expert/exp/all/mlp"])
    res, res_e = vlm_score(st, "res"), nrm(st["expert/exp/all/res"])  # (L_src, d)
    kv, q, mlps, q_e, mlps_e = [], [], [], [], []
    for li in layers:
        g_score = head[li].reshape(-1, rep).sum(1) + head_e[li].reshape(-1, rep_e).sum(1)
        groups = np.sort(np.argsort(-g_score)[:nkv])
        kv.append(groups)
        q.append(np.concatenate([g * rep + np.sort(np.argsort(-head[li, g * rep:(g + 1) * rep])[:per])
                                 for g in groups]))
        q_e.append(np.concatenate([g * rep_e + np.sort(np.argsort(-head_e[li, g * rep_e:(g + 1) * rep_e])[:per_e])
                                   for g in groups]))
        mlps.append(np.sort(np.argsort(-mlp[li])[:m]))
        mlps_e.append(np.sort(np.argsort(-mlp_e[li])[:m_e]))
    final = np.mean([nrm(st[f"{t}/vlm/all/final"][0]) for t in ("driving", "det")], axis=0)
    d_score = res[layers].sum(0) + final
    d_e_score = res_e[layers].sum(0) + nrm(st["expert/exp/all/final"][0])
    return pl.PruneSpec(layers=layers, d=np.sort(np.argsort(-d_score)[:d]), kv=kv, q=q, mlp=mlps,
                        d_e=np.sort(np.argsort(-d_e_score)[:d_e]), q_e=q_e, mlp_e=mlps_e)


def build_vocab(tcfg, sets, top=8):
    """required ids + prompt ids + teacher-generated ids + teacher top-`top` candidates."""
    import distill_data as dd

    ids = [pl.required_ids(tcfg)]
    b = dd.SeqBuilder()
    for set_name in sets:
        items = dd.labelled_items(set_name)
        for j, (c, t) in enumerate(items):
            for task in ("driving", "det"):
                lab = dd.load_label(set_name, task, c, t)
                ids += [lab["gen_ids"], lab["topk_ids"][:, :top].ravel()]
            if j < 4:  # prompts are fixed templates; a few clips cover every prompt id
                import distill_cache as dc
                data = dc.load_cached(dc.path_for(c, t, set_name))
                ids += [b.driving(data)["input_ids"][0].numpy(), b.det(data)["input_ids"][0].numpy()]
    return np.unique(np.concatenate(ids)).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", default=None, help="checkpoint dir to slice (default: teacher)")
    ap.add_argument("--stats", default=str(REPO / "outputs" / "distill_importance_teacher" / "stats.npz"))
    ap.add_argument("--depth", default=None,
                    help="depth.npz from depth_importance.py; layers then follow its drop-one score")
    ap.add_argument("--vocab-sets", nargs="*", default=["train_10k", "dev_238"])
    ap.add_argument("--vocab-top", type=int, default=8)
    args = ap.parse_args()

    src = args.source or pl.TEACHER_SNAPSHOT
    src_cfg = json.loads((Path(src) / "config.json").read_text())
    st = dict(np.load(args.stats))
    depth = dict(np.load(args.depth)) if args.depth else None
    spec = build_spec(st, src_cfg, ARCHS[args.arch], depth)
    if args.source is None:  # vocabulary pruning happens once, when leaving the teacher
        spec.vocab = build_vocab(src_cfg, args.vocab_sets, args.vocab_top)
    spec.meta = {"arch": args.arch, "source": src, "stats": args.stats, "depth": args.depth,
                 "vocab_sets": args.vocab_sets if args.source is None else None}
    spec.validate(src_cfg)

    sd = pl.slice_weights(pl.TeacherWeights(src), spec)
    out = CKPT / args.name / "init"
    cfg = pl.save_student(out, src_cfg, spec, sd)
    cfg["cotrain_expert_vlm"] = True  # the release default freezes the VLM
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if spec.vocab is None and (Path(src) / "vocab_map.npy").exists():  # inherit the source's ids
        shutil.copy(Path(src) / "vocab_map.npy", out / "vocab_map.npy")
    counts = pl.count_params(sd)
    formula = pl.formula_params(spec, V=len(spec.vocab) if spec.vocab is not None else
                                src_cfg["vlm_config"]["text_config"]["vocab_size"])
    assert formula["budget_excl_vit"] == counts["budget_excl_vit"], (formula, counts)
    summary = {"arch": args.arch, "counts": spec.counts(), "params": counts,
               "layers": spec.layers, "out": str(out)}
    (out / "student_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
