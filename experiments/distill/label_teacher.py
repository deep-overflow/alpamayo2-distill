"""P1: Alpamayo 2 Super pseudo labels for the distill sets, one npz per sample.

task driving  meta_action prompt -> CoC + meta-action + <|traj_future_start|> (temperature 0.6,
              top_p 0.98, as text_tasks.generate_text). With --denoise-k K > 0 (evaluation sets)
              K seeded denoisings on that CoC's cache are stored too.
task det      front-wide detection prompt (det_probe_a2.PROMPTS[--det-prompt]) on the grounding
              profile, temperature 0.1, top_p 1.0 (the release grounding sampling).

Stored per sample: gen_ids, the teacher's raw top-64 logits + log-sum-exp per generated token
(so its full-vocab probability of every top-64 token is exact), decoded fields, timing.
Seeds: cache_a2.clip_seed(--seed, clip_id) -- independent of order, shards and restarts.
Resumable: existing npz files are skipped.

Usage (one replica per card pair):
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=0,1 python experiments/distill/label_teacher.py --task driving \
      --sets dev_238 val_500 ood_262 train_10k --denoise-sets dev_238 val_500 ood_262 \
      --denoise-k 8 --shard 0 --n-shards 2
  (and the same with CUDA_VISIBLE_DEVICES=2,3 --shard 1). A set whose cache is still being
  built is polled until its builder writes index.json.
"""

import os

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES to this replica's cards, e.g. 0,1")

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oodval"))

import distill_cache as dc
import teacher_lib as tl
from cache_a2 import clip_seed
from det_utils import PROMPTS
from run_oodval import set_determinism

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_teacher"


def out_path(set_name, task, clip_id, t0_us):
    return OUT / set_name / task / f"{clip_id}__t0_{int(t0_us)}.npz"


def task_inputs(model, processor, args, set_name, clip_id, t0_us):
    data = dc.load_cached(dc.path_for(clip_id, t0_us, set_name))
    if args.task == "driving":
        return tl.driving_inputs(model, processor, data)
    return tl.det_inputs(model, processor, data, PROMPTS[args.det_prompt])


def gen_kwargs(args):
    if args.task == "driving":
        return {"driving": True}
    return {"driving": False, "max_new": 1024, "temperature": 0.1, "top_p": 1.0}


def save(set_name, args, clip_id, t0_us, g, seed, batch, extra):
    p = out_path(set_name, args.task, clip_id, t0_us)
    tmp = p.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, gen_ids=g["gen_ids"], topk_ids=g["topk_ids"], topk_logits=g["topk_logits"],
                 lse=g["lse"], prompt_len=np.int32(g["prompt_len"]),
                 hit_stop=np.bool_(g["hit_stop"]),
                 text=json.dumps({k: g[k] for k in ("cot", "meta_action", "answer", "box", "raw")}),
                 sec=np.float32(g["sec"]), seed=np.int64(seed), batch=np.int32(batch),
                 clip_id=clip_id, t0_us=np.int64(t0_us), **extra)
    tmp.rename(p)


def label_one(model, processor, args, set_name, clip_id, t0_us, denoise_k):
    """Generate (and optionally denoise) one sample and write its npz. -> [(gen s, tokens)]."""
    inp = task_inputs(model, processor, args, set_name, clip_id, t0_us)
    seed = clip_seed(args.seed, clip_id)
    g = tl.generate(model, inp, seed=seed, **gen_kwargs(args))
    extra = {}
    if args.task == "driving" and denoise_k:
        out = g["out"]
        seq = out.sequences[:, :g["prompt_len"] + len(g["gen_ids"]) + 1]  # + padded tail
        kv_len = out.past_key_values.get_seq_length()
        extra["traj"] = tl.denoise_batched(model, out.past_key_values,
                                           model.vlm.model.rope_deltas, seq, kv_len, inp,
                                           k=denoise_k, seed=seed + 1000)
    del g["out"]
    save(set_name, args, clip_id, t0_us, g, seed, 1, extra)
    return [(g["sec"], len(g["gen_ids"]))]


def label_batch(model, processor, args, set_name, items, inps):
    """Batched generation for samples that need no denoising; seed = first clip's seed."""
    seed = clip_seed(args.seed, items[0][0])
    gs = tl.generate_batch(model, inps, seed=seed, **gen_kwargs(args))
    for (c, t), g in zip(items, gs):
        save(set_name, args, c, t, g, seed, len(items), {})
    return [(g["sec"], len(g["gen_ids"])) for g in gs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="labelled in this order")
    ap.add_argument("--task", choices=("driving", "det"), required=True)
    ap.add_argument("--det-prompt", default="p2_release_style")
    ap.add_argument("--denoise-k", type=int, default=0)
    ap.add_argument("--denoise-sets", nargs="*", default=[],
                    help="sets that also get --denoise-k teacher trajectories")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1,
                    help="samples per generate call for sets without denoising")
    ap.add_argument("--poll", type=int, default=120,
                    help="s between rescans while a set's cache is still being built")
    args = ap.parse_args()

    set_determinism()
    t_load = time.time()
    model, processor, per_gpu = tl.load_teacher()
    print(f"teacher loaded in {time.time() - t_load:.0f}s, layers per card {per_gpu}", flush=True)

    pool = ThreadPoolExecutor(max_workers=1)
    for set_name in args.sets:
        k = args.denoise_k if set_name in args.denoise_sets else 0
        man = pd.read_parquet(REPO / "outputs" / "distill_sets" / f"{set_name}.parquet")
        man = [(r.clip_id, int(r.t0_us)) for r in man.itertuples()][args.shard::args.n_shards]
        if args.limit:
            man = man[:args.limit]
        (OUT / set_name / args.task).mkdir(parents=True, exist_ok=True)
        (OUT / set_name / args.task / f"config_s{args.shard}of{args.n_shards}.json").write_text(
            json.dumps({
                "set": set_name, "task": args.task, "seed": args.seed,
                "seed_rule": ("sha256(f'{seed}:{clip_id}')[:4]; the K denoisings are one batch seeded +1000"
                              if k or args.batch == 1 else
                              f"batches of {args.batch}: one RNG stream seeded with the first "
                              "clip's sha256 seed"),
                "batch": 1 if k else args.batch,
                "det_prompt": PROMPTS[args.det_prompt] if args.task == "det" else None,
                "sampling": ({"temperature": 0.6, "top_p": 0.98} if args.task == "driving"
                             else {"temperature": 0.1, "top_p": 1.0}),
                "topk": tl.TOPK, "denoise_k": k, "model_revision": tl.MODEL_REV,
                "input_profile": "cameras [0,1,2,3,5,6] x 4 frames at 320x576 (distill cache)",
                "gpus": os.environ["CUDA_VISIBLE_DEVICES"]}, indent=2))
        t0, secs, lens, n_done = time.time(), [], [], 0
        while True:
            left = [(c, t) for c, t in man if not out_path(set_name, args.task, c, t).exists()]
            todo = [(c, t) for c, t in left if dc.path_for(c, t, set_name).exists()]
            if not left:
                break
            if not todo:
                if (dc.ROOT / set_name / "index.json").exists():  # builder finished; rest failed
                    print(f"{set_name}: {len(left)} samples have no cache, skipped", flush=True)
                    break
                time.sleep(args.poll)
                continue
            step = 1 if k else args.batch
            batches = [todo[i:i + step] for i in range(0, len(todo), step)]

            def prep(items, set_name=set_name):
                return [task_inputs(model, processor, args, set_name, c, t) for c, t in items]

            nxt = pool.submit(prep, batches[0]) if step > 1 else None
            for j, items in enumerate(batches):
                if step == 1:
                    res = label_one(model, processor, args, set_name, *items[0], k)
                else:  # inputs of the next batch are prepared on CPU while this one generates
                    inps = nxt.result()
                    if j + 1 < len(batches):
                        nxt = pool.submit(prep, batches[j + 1])
                    res = label_batch(model, processor, args, set_name, items, inps)
                for sec, n_tok in res:
                    secs.append(sec)
                    lens.append(n_tok)
                prev, n_done = n_done, n_done + len(items)
                if n_done // 25 > prev // 25:
                    el = time.time() - t0
                    print(f"{set_name} [{n_done}/{len(man)}] {el / n_done:.1f} s/sample "
                          f"(gen {np.mean(secs):.1f} s, {np.mean(lens):.0f} tok)", flush=True)
        print(f"{set_name}/{args.task} shard {args.shard}/{args.n_shards} done: {n_done} labelled "
              f"this run in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
