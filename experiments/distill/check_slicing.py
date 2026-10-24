"""Gate G0b: is slice_lib's slicing exactly the sub-network it claims to keep?

static  (CPU) keep-all slicing of layers {0, 31, 63} + every non-layer tensor reproduces
        the teacher tensors bit for bit; keep-all student_config equals the teacher config.
masked  (GPU) a partial spec -- 9 of 64 layers (0..2 kept for deepstack), 4 of 8 KV groups
        and 4 of 8 query heads per kept group, half the MLP channels, half the expert MLP,
        1 of 2 expert heads per kept group, a pruned vocab -- sliced into a real student
        checkpoint, against the teacher with the same parts removed by hooks (dropped
        layers return their input, dropped heads / MLP channels zeroed before o_proj /
        down_proj). Residual width is kept whole: RMSNorm has no exact sliced equivalent.
        Compared on teacher-forced logits over the kept vocab and on same-seed denoisings.
        Pass: top-1 agreement >= 0.98 and same-seed ADE < 0.1 x the seed-noise ADE.
counts  the sliced student's parameter count equals slice_lib.formula_params.

Usage:
  source experiments/oodval/env.sh
  python experiments/distill/check_slicing.py --static
  CUDA_VISIBLE_DEVICES=0,1 python experiments/distill/check_slicing.py --masked
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

import slice_lib as pl

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_g0b"
STUDENT_TMP = Path("/mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/g0b_student")


def static_check():
    tcfg = pl.teacher_config()
    tw = pl.TeacherWeights()
    full = pl.keep_all(tcfg)
    full.validate(tcfg)
    cfg = pl.student_config(tcfg, full)
    cfg.pop("prune_spec_counts")
    cfg_equal = cfg == tcfg
    sub = pl.keep_all(tcfg)
    pick = [0, 31, 63]
    sub.layers = pick
    for k in ("kv", "q", "mlp", "q_e", "mlp_e"):
        setattr(sub, k, [getattr(sub, k)[i] for i in pick])
    sd = pl.slice_weights(tw, sub)
    bad, n = [], 0
    for name, t in sd.items():
        src = name
        for tower in (pl.VLM, pl.EXP):
            pre = f"{tower}.layers."
            if name.startswith(pre):
                i, rest = name[len(pre):].split(".", 1)
                src = f"{pre}{pick[int(i)]}.{rest}"
        ref = tw(src)
        n += 1
        if ref.shape != t.shape or not torch.equal(ref, t):
            bad.append(name)
    return {"config_equal": cfg_equal, "tensors_compared": n, "tensors_different": bad[:20],
            "pass": cfg_equal and not bad}


def partial_spec(tcfg, keep_ids, seed=0):
    rng = np.random.default_rng(seed)
    t, e = tcfg["vlm_config"]["text_config"], tcfg["expert_config"]["llm_config"]
    rep, rep_e = t["num_attention_heads"] // t["num_key_value_heads"], \
        e["num_attention_heads"] // e["num_key_value_heads"]
    layers = [0, 1, 2] + list(range(10, 64, 9))  # 9 layers: the student shares card 0
    kv, q, mlp, q_e, mlp_e = [], [], [], [], []
    for _ in layers:
        g = np.sort(rng.choice(t["num_key_value_heads"], 4, replace=False))
        kv.append(g)
        q.append(np.concatenate([np.sort(rng.choice(rep, 4, replace=False)) + gi * rep for gi in g]))
        q_e.append(np.concatenate([np.sort(rng.choice(rep_e, 1, replace=False)) + gi * rep_e
                                   for gi in g]))
        mlp.append(np.sort(rng.choice(t["intermediate_size"], t["intermediate_size"] // 2,
                                      replace=False)))
        mlp_e.append(np.sort(rng.choice(e["intermediate_size"], e["intermediate_size"] // 2,
                                        replace=False)))
    vocab = np.union1d(keep_ids, rng.choice(t["vocab_size"], 20_000, replace=False))
    spec = pl.PruneSpec(layers=layers, d=np.arange(t["hidden_size"]), kv=kv, q=q, mlp=mlp,
                        d_e=np.arange(e["hidden_size"]), q_e=q_e, mlp_e=mlp_e, vocab=vocab,
                        meta={"what": "G0b masked-equivalence test spec", "seed": seed})
    spec.validate(tcfg)
    return spec


class TeacherMasks:
    """Hooks that make the full teacher compute exactly the sub-network `spec` keeps."""

    def __init__(self, model, spec):
        self.handles = []
        t = model.config.vlm_config.text_config
        e = model.config.expert_config.llm_config
        kept = {src: i for i, src in enumerate(spec.layers)}
        towers = ((model.vlm.model.language_model.layers, t, spec.q, spec.mlp),
                  (model.expert.expert.layers, e, spec.q_e, spec.mlp_e))
        for layers, cfg, q_sets, mlp_sets in towers:
            for src, layer in enumerate(layers):
                if src not in kept:
                    self.handles.append(layer.register_forward_hook(self._skip, with_kwargs=True))
                    continue
                i = kept[src]
                hmask = torch.zeros(cfg.num_attention_heads * pl.HD)
                hmask[pl.head_rows(q_sets[i])] = 1
                mmask = torch.zeros(cfg.intermediate_size)
                mmask[mlp_sets[i]] = 1
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._mul(hmask)))
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._mul(mmask)))

    @staticmethod
    def _skip(module, args, kwargs, output):
        h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        if isinstance(output, tuple):
            return (h, *output[1:])
        return h

    @staticmethod
    def _mul(mask):
        cache = {}

        def hook(module, args):
            x = args[0]
            key = (x.device, x.dtype)
            if key not in cache:
                cache[key] = mask.to(device=x.device, dtype=x.dtype)
            return (x * cache[key], *args[1:])
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def masked_check(n_clips, k):
    import distill_cache as dc
    import teacher_lib as tl
    from run_oodval import set_determinism

    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    set_determinism()
    model, processor, _per_gpu = tl.load_teacher()
    man = pd.read_parquet(REPO / "outputs" / "distill_sets" / "calib_100.parquet").head(n_clips)
    clips = []
    for r in man.itertuples():
        data = dc.load_cached(dc.path_for(r.clip_id, r.t0_us, "calib_100"))
        inp = tl.driving_inputs(model, processor, data)
        g = tl.generate(model, inp, seed=7, driving=True)
        del g["out"]
        clips.append((r.clip_id, inp, g["gen_ids"]))
    ids = np.unique(np.concatenate([np.concatenate([c[1]["tok"]["input_ids"][0].cpu().numpy(),
                                                    c[2]]) for c in clips]))
    tcfg = pl.teacher_config()
    spec = partial_spec(tcfg, keep_ids=np.union1d(ids, pl.required_ids(tcfg)))

    t = time.time()
    sd = pl.slice_weights(pl.TeacherWeights(), spec)
    counts = pl.count_params(sd)
    formula = pl.formula_params(spec, V=len(spec.vocab))
    pl.save_student(STUDENT_TMP, tcfg, spec, sd)
    del sd
    print(f"student sliced + saved in {time.time() - t:.0f}s: {counts}", flush=True)
    student = Alpamayo2Super.from_pretrained(str(STUDENT_TMP), dtype=torch.bfloat16,
                                             device_map={"": 0})
    student.eval()
    vmap = torch.as_tensor(np.load(STUDENT_TMP / "vocab_map.npy"), device="cuda:0")
    kept = torch.as_tensor(spec.vocab, device="cuda:0")

    masks = TeacherMasks(model, spec)
    rows = []
    seeds = [100 + j for j in range(k)]
    for clip_id, inp, gen in clips:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen_t = torch.as_tensor(gen, device="cuda:0").view(1, -1)
            ids_full = torch.cat([inp["tok"]["input_ids"], gen_t], dim=1)  # (1, L)
            vis = {kk: inp["tok"][kk] for kk in tl.VISION_KEYS}
            out_t = model.vlm(input_ids=ids_full, attention_mask=torch.ones_like(ids_full),
                              use_cache=True, **vis)
            p = inp["tok"]["input_ids"].shape[1]
            lt = out_t.logits[0, p - 1:-1].float().to("cuda:0")[:, kept]  # (n, V')
            traj_t = tl.denoise(model, out_t.past_key_values, model.vlm.model.rope_deltas,
                                ids_full, ids_full.shape[1], inp, seeds)
            noise = tl.denoise(model, model.vlm(input_ids=ids_full,
                                                attention_mask=torch.ones_like(ids_full),
                                                use_cache=True, **vis).past_key_values,
                               model.vlm.model.rope_deltas, ids_full, ids_full.shape[1], inp,
                               [s + 50 for s in seeds])
            ids_s = vmap[ids_full]  # (1, L) student ids
            assert (ids_s >= 0).all()
            vis_s = {kk: v.to("cuda:0") for kk, v in vis.items()}
            out_s = student.vlm(input_ids=ids_s, attention_mask=torch.ones_like(ids_s),
                                use_cache=True, **vis_s)
            ls = out_s.logits[0, p - 1:-1].float()  # (n, V')
            traj_s = tl.denoise(student, out_s.past_key_values, student.vlm.model.rope_deltas,
                                ids_s, ids_s.shape[1],
                                {kk: v.to("cuda:0") for kk, v in inp.items() if kk != "tok"},
                                seeds)
        pt = torch.softmax(lt, -1)
        kl = (pt * (torch.log_softmax(lt, -1) - torch.log_softmax(ls, -1))).sum(-1)
        ade = lambda a, b: float(np.linalg.norm(a[:, :2] - b[:, :2], axis=-1).mean())
        rows.append({
            "clip_id": clip_id, "n_tokens": len(gen),
            "logit_maxabs": float((lt - ls).abs().max()), "logit_scale": float(lt.abs().max()),
            "top1_agree": float((lt.argmax(-1) == ls.argmax(-1)).float().mean()),
            "kl_mean": float(kl.mean()),
            "same_seed_ade": float(np.mean([ade(traj_t[j], traj_s[j]) for j in range(k)])),
            "seed_noise_ade": float(np.mean([ade(traj_t[j], noise[j]) for j in range(k)])),
        })
        print(rows[-1], flush=True)
    masks.remove()
    df = pd.DataFrame(rows)
    res = {"spec_counts": spec.counts(), "params": counts, "formula": formula,
           "counts_match": formula["budget_excl_vit"] == counts["budget_excl_vit"],
           "rows": rows, "top1_agree": float(df.top1_agree.mean()),
           "same_seed_ade": float(df.same_seed_ade.mean()),
           "seed_noise_ade": float(df.seed_noise_ade.mean()), "kl_mean": float(df.kl_mean.mean())}
    res["pass"] = bool(res["counts_match"] and res["top1_agree"] >= 0.98
                       and res["same_seed_ade"] < 0.1 * res["seed_noise_ade"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((OUT / "metrics.json").read_text()) if (OUT / "metrics.json").exists() else {}
    if args.static:
        res["static"] = static_check()
        print("static:", res["static"], flush=True)
    if args.masked:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise SystemExit("set CUDA_VISIBLE_DEVICES for --masked")
        res["masked"] = masked_check(args.n, args.k)
        print("masked pass:", res["masked"]["pass"], flush=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    lines = ["G0b slicing checks"]
    if "static" in res:
        s = res["static"]
        lines.append(f"static: config equal {s['config_equal']}, {s['tensors_compared']} tensors, "
                     f"{len(s['tensors_different'])} different -> {'PASS' if s['pass'] else 'FAIL'}")
    if "masked" in res:
        m = res["masked"]
        lines += [f"masked: spec {m['spec_counts']}",
                  f"  params {m['params']}",
                  f"  formula {m['formula']} -> counts match {m['counts_match']}",
                  (f"  top-1 agreement {m['top1_agree']:.4f}, KL {m['kl_mean']:.2e}, same-seed ADE "
                  f"{m['same_seed_ade']:.4f} vs seed noise {m['seed_noise_ade']:.4f} -> "
                  f"{'PASS' if m['pass'] else 'FAIL'}")]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
