"""P2/P3: distil a pruned Alpamayo 2 student on the teacher-labelled clips.

Per clip (one micro-batch), two teacher-forced sequences (distill_data.DistillDataset):
  driving  CE on the teacher's CoC + meta-action tokens, KL to the teacher's top-64
           distribution at the same steps, and `--n-fm` flow-matching draws of the expert on
           the GT future over this sequence's KV cache
  det      CE + KL on the teacher's detection JSON
loss = w_ce * (CE_drv + CE_det) + w_kd * (KL_drv + KL_det) + w_fm * mean(FM)

The ViT blocks stay frozen (their outputs feed the trainable mergers, whose last layer was
sliced to the new width); everything else trains. FSDP2 (`fully_shard`) per decoder layer of
both towers and per ViT block, bf16 compute, fp32 master weights and reductions. The whole
step runs inside one wrapper module so FSDP's root hooks see every call.

Checkpoint selection: teacher-forced dev_238 loss every `--eval-every` steps; the best and the
last weights are written (bf16 safetensors, loadable with Alpamayo2Super.from_pretrained) next
to the init checkpoint; config / metrics / summary / plots go to outputs/<exp-id>/.

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=1,2,3 .venv/bin/torchrun --nproc_per_node 3 experiments/distill/train_student.py \
      --init /mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/s1/init --exp-id distill_s1 \
      --epochs 3 --lr 5e-5
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oodval"))

import distill_data as dd

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

REPO = Path(__file__).resolve().parents[2]
VISION_KEYS = dd.VISION_KEYS


class SharedVision:
    """Run the frozen ViT blocks once per clip and only the trainable mergers per sequence.

    The driving and detection sequences of a clip carry the same 24 frames, and every ViT
    block is frozen, so the block outputs (final + the deepstack taps) are computed without
    grad on the first sequence and reused on the second. The body is
    Qwen3VLVisionModel.forward split at the mergers.
    """

    def __init__(self, visual):
        self.v, self.feats = visual, None
        visual.forward = self.forward

    @torch.no_grad()
    def _blocks(self, hidden_states, grid_thw, **kwargs):
        v = self.v
        h = v.patch_embed(hidden_states)
        h = h + v.fast_pos_embed_interpolate(grid_thw)
        rot = v.rot_pos_emb(grid_thw)
        seq_len, _ = h.size()
        h = h.reshape(seq_len, -1)
        rot = rot.reshape(seq_len, -1)
        emb = torch.cat((rot, rot), dim=-1)
        pos = (emb.cos(), emb.sin())
        cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32)
        cu = F.pad(cu, (1, 0), value=0)
        taps = []
        for i, blk in enumerate(v.blocks):
            h = blk(h, cu_seqlens=cu, position_embeddings=pos, **kwargs)
            if i in v.deepstack_visual_indexes:
                taps.append(h)
        return h, taps

    def forward(self, hidden_states, grid_thw, **kwargs):
        if self.feats is None:
            self.feats = self._blocks(hidden_states, grid_thw, **kwargs)
        h, taps = self.feats
        return self.v.merger(h), [m(t) for m, t in zip(self.v.deepstack_merger_list, taps)]


class Step(torch.nn.Module):
    """Wraps the student so that every model call of a step goes through FSDP's root."""

    def __init__(self, model, args):
        super().__init__()
        self.model, self.args = model, args
        self.vision = SharedVision(model.vlm.model.visual)

    def seq_losses(self, seq, dev):
        ids = seq["input_ids"].to(dev)  # (1, L)
        n = seq["n_target"]
        out = self.model.vlm(input_ids=ids, attention_mask=seq["attention_mask"].to(dev),
                             use_cache="traj" in seq, logits_to_keep=n + 1,
                             **{k: seq[k].to(dev) for k in VISION_KEYS})
        logits = out.logits[0, :-1].float()  # (n, V): predictions of the n target tokens
        tgt = ids[0, -n:]
        ce = F.cross_entropy(logits, tgt)
        logq = F.log_softmax(logits, dim=-1)  # (n, V)
        kd_ids, kd_p = seq["kd_ids"].to(dev), seq["kd_p"].to(dev)  # (n, k)
        kl = (kd_p * (torch.log(kd_p.clamp(min=1e-12)) - logq.gather(1, kd_ids))).sum(-1).mean()
        return ce, kl, out

    def forward(self, item, task):
        """Loss of one of the clip's two sequences; the caller backpropagates each separately
        so only one sequence's activations are alive at a time. -> (loss, parts (5,))."""
        dev = torch.device("cuda", torch.cuda.current_device())
        a = self.args
        zero = torch.zeros((), device=dev)
        if task == TASKS[0]:  # a new clip: its frozen ViT features are computed once
            self.vision.feats = None
        if task == "driving":
            ce, kl, out = self.seq_losses(item["driving"], dev)
            traj = {k: v.to(dev) for k, v in item["driving"]["traj"].items()}
            fm = torch.stack([dd.expert_fm(self.model, traj, out) for _ in range(a.n_fm)]).mean()
            loss = a.w_ce * ce + a.w_kd * kl + a.w_fm * fm
            parts = torch.stack([ce, kl, zero, zero, fm])
        else:
            ce, kl, _ = self.seq_losses(item["det"], dev)
            loss = a.w_ce * ce + a.w_kd * kl
            parts = torch.stack([zero, zero, ce, kl, zero])
        return loss, parts.detach()


PART_NAMES = ("ce_drv", "kl_drv", "ce_det", "kl_det", "fm")
TASKS = ("driving", "det")


def build(args, rank):
    model = Alpamayo2Super.from_pretrained(args.init, dtype=torch.float32)
    model.config.use_cache = True
    vis = model.vlm.model.visual
    for name, p in vis.named_parameters():
        p.requires_grad_("merger" in name)  # merger + deepstack mergers train, the ViT is frozen
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in vis.blocks:
        fully_shard(blk, mp_policy=mp)
    for layer in model.vlm.model.language_model.layers:
        fully_shard(layer, mp_policy=mp)
    for layer in model.expert.expert.layers:
        fully_shard(layer, mp_policy=mp)
    step = Step(model, args)
    # the root must not cast its inputs: trajectories (cholesky in traj_to_action), KD
    # probabilities and pixel values stay as built; autocast handles the compute dtype
    fully_shard(step, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                     reduce_dtype=torch.float32,
                                                     cast_forward_inputs=False))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"student {args.init}: {n_all / 1e9:.3f}B params, {n_train / 1e9:.3f}B trainable",
              flush=True)
    return model, step, n_all, n_train


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or "norm" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


@torch.no_grad()
def evaluate(step_mod, items, ds, rank, world):
    step_mod.eval()
    tot, n = torch.zeros(len(PART_NAMES) + 1, device="cuda"), 0
    # every rank must run the same number of forwards (each one issues FSDP all-gathers):
    # ranks short of items repeat item 0 with weight 0
    for k in range(math.ceil(len(items) / world)):
        i = rank + k * world
        w = 1.0 if i < len(items) else 0.0
        item = ds[i if w else 0]
        for task in TASKS:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = step_mod(item, task)
            tot += w * torch.cat([loss.detach().view(1), parts])
        n += int(w)
    cnt = torch.tensor([n], device="cuda", dtype=torch.float32)
    dist.all_reduce(tot)
    dist.all_reduce(cnt)
    step_mod.train()
    vals = (tot / cnt).tolist()
    return {"loss": vals[0], **dict(zip(PART_NAMES, vals[1:]))}


def save_weights(model, init_dir, out_dir, rank):
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True,
                                                              cpu_offload=True))
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        sd = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
        save_file(sd, str(out_dir / "model.safetensors.tmp"), metadata={"format": "pt"})
        (out_dir / "model.safetensors.tmp").rename(out_dir / "model.safetensors")
        for f in Path(init_dir).iterdir():
            if f.name != "model.safetensors" and f.is_file():
                shutil.copy(f, out_dir / f.name)
    dist.barrier()


def plot_curves(metrics, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = [m for m in metrics if m["kind"] == "train"]
    ev = [m for m in metrics if m["kind"] == "dev"]
    fig, axes = plt.subplots(1, len(PART_NAMES), figsize=(3.2 * len(PART_NAMES), 3.0))
    for ax, k in zip(axes, PART_NAMES):
        ax.plot([m["step"] for m in tr], [m[k] for m in tr], lw=0.8, label="train")
        ax.plot([m["step"] for m in ev], [m[k] for m in ev], "o-", ms=3, label="dev")
        ax.set_title(k)
        ax.set_xlabel("step")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=None, help="cap (short G2b runs, smoke)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--global-batch", type=int, default=32, help="clips per optimizer step")
    ap.add_argument("--w-ce", type=float, default=1.0)
    ap.add_argument("--w-kd", type=float, default=1.0)
    ap.add_argument("--w-fm", type=float, default=1.0)
    ap.add_argument("--n-fm", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many dev evaluations without improvement (0 = off)")
    ap.add_argument("--dev-limit", type=int, default=None)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    out = REPO / "outputs" / args.exp_id
    ckpt = Path(args.init).parent / args.exp_id
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    vmap_p = Path(args.init) / "vocab_map.npy"
    builder = dd.SeqBuilder(vocab_map=np.load(vmap_p) if vmap_p.exists() else None)
    train_items = dd.labelled_items("train_10k")[:args.train_limit]
    dev_items = dd.labelled_items("dev_238")[:args.dev_limit]
    train_ds = dd.DistillDataset("train_10k", train_items, builder)
    dev_ds = dd.DistillDataset("dev_238", dev_items, builder)

    model, step_mod, n_all, n_train = build(args, rank)
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.95),
                            fused=True)
    accum = max(1, round(args.global_batch / world))
    steps_per_epoch = len(train_items) // (accum * world)
    total = int(args.epochs * steps_per_epoch)
    if args.max_steps:
        total = min(total, args.max_steps)
    cfg = {**vars(args), "world": world, "accum": accum, "steps_per_epoch": steps_per_epoch,
           "total_steps": total, "n_train_clips": len(train_items), "n_dev_clips": len(dev_items),
           "params": n_all, "trainable": n_train, "ckpt_dir": str(ckpt),
           "gpus": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if rank == 0:
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        print(json.dumps(cfg), flush=True)

    sampler = torch.utils.data.DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                                  shuffle=True, seed=args.seed, drop_last=True)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=None, sampler=sampler,
                                         num_workers=args.workers, persistent_workers=True,
                                         prefetch_factor=2)
    metrics, best, t0, step, epoch = [], float("inf"), time.time(), 0, 0
    stale = 0  # dev evaluations since the last improvement
    run = torch.zeros(len(PART_NAMES), device="cuda")
    n_run, done = 0, False
    while not done:
        sampler.set_epoch(epoch)
        it = iter(loader)
        while True:
            micro = []
            for _ in range(accum):
                try:
                    micro.append(next(it))
                except StopIteration:
                    break
            if len(micro) < accum:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, args.warmup, args.lr)
            for j, item in enumerate(micro):
                for task in TASKS:  # backward per sequence: one sequence's graph at a time
                    step_mod.set_requires_gradient_sync(j == accum - 1 and task == TASKS[-1])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, parts = step_mod(item, task)
                    (loss / accum).backward()
                    run += parts
                n_run += 1
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total:
                r = run.clone()
                c = torch.tensor([n_run], device="cuda", dtype=torch.float32)
                dist.all_reduce(r)
                dist.all_reduce(c)
                row = {"kind": "train", "step": step, "epoch": epoch,
                       "lr": opt.param_groups[0]["lr"], "grad_norm": float(gnorm),
                       **dict(zip(PART_NAMES, (r / c).tolist())),
                       "sec_per_step": (time.time() - t0) / step}
                metrics.append(row)
                run.zero_()
                n_run = 0
                if rank == 0:
                    print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == total:
                ev = {"kind": "dev", "step": step, "epoch": epoch,
                      **evaluate(step_mod, dev_items, dev_ds, rank, world)}
                metrics.append(ev)
                if rank == 0:
                    print(json.dumps(ev), flush=True)
                if ev["loss"] < best:
                    best, stale = ev["loss"], 0
                    save_weights(model, args.init, ckpt / "best", rank)
                else:
                    stale += 1  # dev loss is all-reduced, so every rank stops together
                    if args.patience and stale >= args.patience:
                        done = True
                        if rank == 0:
                            print(f"early stop at step {step}: no dev improvement in {stale} "
                                  "evaluations", flush=True)
                if rank == 0:
                    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
                    plot_curves(metrics, out / "curves.png")
            if done or step >= total:
                done = True
                break
        epoch += 1

    save_weights(model, args.init, ckpt / "last", rank)
    if rank == 0:
        devs = [m for m in metrics if m["kind"] == "dev"]
        b = min(devs, key=lambda m: m["loss"])
        lines = [(f"{args.exp_id}: {step} of {total} steps ({step / max(steps_per_epoch, 1):.2f} epochs), "
                 f"{(time.time() - t0) / 3600:.2f} h on {world} GPUs"),
                 f"student {n_all / 1e9:.3f}B params ({n_train / 1e9:.3f}B trainable), init {args.init}",
                 f"best dev (step {b['step']}): loss {b['loss']:.4f} | " + ", ".join(
                     f"{k} {b[k]:.4f}" for k in PART_NAMES),
                 f"last dev (step {devs[-1]['step']}): loss {devs[-1]['loss']:.4f}",
                 f"weights: {ckpt / 'best'}, {ckpt / 'last'}"]
        (out / "summary.txt").write_text("\n".join(lines) + "\n")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        plot_curves(metrics, out / "curves.png")
        print("\n".join(lines), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
