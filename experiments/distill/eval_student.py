"""P4: run a distilled student (camera or camera + LiDAR) on the report sets, one npz per sample.

Per sample, with the same sampling as the teacher labels (label_teacher.py):
  driving  the student samples its own CoC + meta-action (temperature 0.6, top_p 0.98, the
           discrete trajectory ids masked) until <|traj_future_start|>, then K = 8 trajectories
           are denoised on that cache in one batch (teacher_lib.denoise_batched)
  det      the detection answer (temperature 0.1, top_p 1.0) until <|im_end|>
Generated ids are mapped back to the teacher vocabulary before decoding, so the text fields are
comparable with the teacher's. The decode loop is our own (the student's vocabulary is pruned
and the LiDAR prefill passes inputs_embeds + explicit MRoPE positions).
Seeds: cache_a2.clip_seed(--seed, clip_id) -- the teacher's rule. Resumable.

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=0 python experiments/distill/eval_student.py --model <ckpt dir> \
      --tag cam_s2 --sets val_500 ood_262 --shard 0 --n-shards 4
"""

import os

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES, e.g. 0")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oodval"))

import build_lidar_distill as bl
import distill_cache as dc
import distill_data as dd
import lidar_lib as ll
import teacher_lib as tl
from cache_a2 import clip_seed

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
from alpamayo2_super.models.token_utils import extract_text_tokens

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_eval"


def load_student(path, device):
    model = Alpamayo2Super.from_pretrained(path, dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    vmap = np.load(Path(path) / "vocab_map.npy")
    inv = np.full(int(vmap.max()) + 1, -1, dtype=np.int64)
    inv[vmap[vmap >= 0]] = np.flatnonzero(vmap >= 0)
    enc = None
    if (Path(path) / "lidar_encoder.safetensors").exists():
        from safetensors.torch import load_file

        enc = ll.RangeViT(**json.loads((Path(path) / "lidar_config.json").read_text()))
        enc.load_state_dict(load_file(str(Path(path) / "lidar_encoder.safetensors")))
        enc = enc.to(device=device, dtype=torch.bfloat16).eval()
    return model, vmap, inv, enc


def top_p_sample(logits, temperature, top_p, gen):
    """Nucleus sampling on CPU (the per-clip generator lives there; V' is only ~4.9k)."""
    probs = torch.softmax(logits.float().cpu() / temperature, dim=-1)
    sp, si = probs.sort(descending=True)
    keep = sp.cumsum(-1) - sp < top_p
    sp = sp * keep
    return int(si[torch.multinomial(sp / sp.sum(), 1, generator=gen)])


@tl.bf16
def prefill(model, enc, seq, lidar, device):
    ids = seq["input_ids"].to(device)
    kw = {"use_cache": True, "logits_to_keep": 1}
    if enc is None:
        out = model.vlm(input_ids=ids, attention_mask=seq["attention_mask"].to(device),
                        pixel_values=seq["pixel_values"].to(device),
                        image_grid_thw=seq["image_grid_thw"].to(device), **kw)
    else:
        out = ll.lidar_vlm_forward(model, enc, ids, seq["attention_mask"].to(device),
                                   seq["pixel_values"].to(device),
                                   seq["image_grid_thw"].to(device), lidar, **kw)
    return ids, out


@tl.bf16
def decode(model, out, stop_id, max_new, temperature, top_p, mask, gen, device):
    """Sample until stop_id (fed once more so its KV is in the cache). -> (new ids, cache)."""
    logits, cache, toks = out.logits[0, -1].float(), out.past_key_values, []
    for _ in range(max_new):
        if mask is not None:
            logits[mask[0]:mask[1]] = float("-inf")
        tok = top_p_sample(logits, temperature, top_p, gen)
        toks.append(tok)
        # explicit positions: without cache_position Qwen3-VL recomputes MRoPE from 0 for the
        # new token; a decoded text token sits at cache length + rope_deltas on all 3 axes
        n_past = cache.get_seq_length()
        pos = (n_past + model.vlm.model.rope_deltas).view(1, 1, 1).expand(3, 1, 1).to(device)
        o = model.vlm(input_ids=torch.tensor([[tok]], device=device), past_key_values=cache,
                      position_ids=pos, cache_position=torch.tensor([n_past], device=device),
                      use_cache=True)
        cache = o.past_key_values
        if tok == stop_id:
            break
        logits = o.logits[0, -1].float()
    return toks, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="student checkpoint dir")
    ap.add_argument("--tag", required=True, help="output name, e.g. cam_s2 / lidar_s2")
    ap.add_argument("--sets", nargs="+", default=["val_500", "ood_262"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-lidar-clips", action="store_true",
                    help="restrict to clips with a LiDAR cache (paired camera / LiDAR comparison)")
    ap.add_argument("--det-max-new", type=int, default=1400,
                    help="detection answer length cap (512 for the A2-label students' runs)")
    ap.add_argument("--limit", type=int, default=None, help="first N items per set (smoke)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    device = "cuda:0"
    model, vmap, inv, enc = load_student(args.model, device)
    b = dd.SeqBuilder(vocab_map=vmap)
    tids = model.config.traj_ids
    mask = (min(tids["history_id0"], tids["future_id0"]),
            min(tids["history_id0"], tids["future_id0"]) + model.config.traj_vocab_size)
    im_end = int(vmap[b.tokenizer.convert_tokens_to_ids("<|im_end|>")])
    n_params = sum(p.numel() for p in model.parameters()) + (
        sum(p.numel() for p in enc.parameters()) if enc is not None else 0)
    print(f"{args.tag}: {args.model} | {n_params / 1e9:.3f}B params | LiDAR {enc is not None}",
          flush=True)
    for set_name in args.sets:
        man = pd.read_parquet(REPO / "outputs" / "distill_sets" / f"{set_name}.parquet")
        items = [(r.clip_id, int(r.t0_us)) for r in man.itertuples()
                 if dc.path_for(r.clip_id, r.t0_us, set_name).exists()]
        if enc is not None or args.only_lidar_clips:
            items = [(c, t) for c, t in items if bl.path_for(c, t, set_name).exists()]
        items = items[args.shard::args.n_shards][:args.limit]
        od = OUT / args.tag / set_name
        od.mkdir(parents=True, exist_ok=True)
        (od / f"config_s{args.shard}of{args.n_shards}.json").write_text(json.dumps({
            **vars(args), "set": set_name, "n_items": len(items), "params": n_params,
            "lidar": enc is not None, "sampling": {"driving": [0.6, 0.98], "det": [0.1, 1.0]},
            "det_max_new": args.det_max_new},
            indent=2))
        t_all = time.time()
        for j, (clip_id, t0_us) in enumerate(items):
            p = od / f"{clip_id}__t0_{t0_us}.npz"
            if p.exists():
                continue
            data = dc.load_cached(dc.path_for(clip_id, t0_us, set_name))
            lidar = None
            if enc is not None:
                lidar = ll.lidar_tensors(bl.load_cached(bl.path_for(clip_id, t0_us, set_name)), device)
                lidar = {k: v.to(torch.bfloat16) for k, v in lidar.items()}
            seed = clip_seed(args.seed, clip_id)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            torch.cuda.synchronize()
            t = time.time()
            seq = b.driving(data, None, lidar=enc is not None)
            ids, out = prefill(model, enc, seq, lidar, device)
            torch.cuda.synchronize()
            t_prefill = time.time() - t
            drv, cache = decode(model, out, tids["future_start"], 256, 0.6, 0.98, mask, gen, device)
            torch.cuda.synchronize()
            t_gen = time.time() - t - t_prefill
            seq_ids = torch.cat([ids, torch.tensor([drv], device=device)], dim=1)
            traj = tl.denoise_batched(model, cache, model.vlm.model.rope_deltas, seq_ids,
                                      cache.get_seq_length(),
                                      {k: data[k].to(device) for k in ("ego_history_xyz",
                                                                       "ego_history_rot")},
                                      k=args.k, seed=seed + 1000)
            torch.cuda.synchronize()
            t_den = time.time() - t - t_prefill - t_gen
            dseq = b.det(data, None, lidar=enc is not None)
            _, dout = prefill(model, enc, dseq, lidar, device)
            det, _ = decode(model, dout, im_end, args.det_max_new, 0.1, 1.0, mask, gen, device)
            t_det = time.time() - t - t_prefill - t_gen - t_den
            dtxt = extract_text_tokens(b.tokenizer, torch.tensor([inv[drv].tolist()]))
            ttxt = extract_text_tokens(b.tokenizer, torch.tensor([inv[det].tolist()]))
            tmp = p.with_suffix(".npz.tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, traj=traj, drv_ids=np.asarray(inv[drv]), det_ids=np.asarray(inv[det]),
                         text=json.dumps({"cot": dtxt["cot"][0], "meta_action": dtxt["meta_action"][0],
                                          "det_answer": ttxt["answer"][0],
                                          "drv_hit_stop": bool(drv and drv[-1] == tids["future_start"]),
                                          "det_hit_stop": bool(det and det[-1] == im_end)}),
                         timing=np.array([t_prefill, t_gen, t_den, t_det], np.float32),
                         n_drv=np.int32(len(drv)), n_det=np.int32(len(det)), seed=np.int64(seed),
                         clip_id=clip_id, t0_us=np.int64(t0_us))
            tmp.rename(p)
            if (j + 1) % 25 == 0:
                print(f"{set_name} [{j + 1}/{len(items)}] {(time.time() - t_all) / (j + 1):.1f} s/sample "
                      f"(prefill {t_prefill:.2f} gen {t_gen:.2f} denoise {t_den:.2f} det {t_det:.2f})",
                      flush=True)
        print(f"{args.tag}/{set_name} shard {args.shard}: done", flush=True)


if __name__ == "__main__":
    main()
