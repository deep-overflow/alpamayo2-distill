"""Gradient-side helpers for dual-objective Taylor importance on Alpamayo 2 Super.

Ported from alpamayo-model-compression/experiments/head_analysis/{prune_lib,analysis_lib}.py
(UnitGates, TypedUnitGates/_TypedGate, coc_nll, kv_group_scores, expert_fm_grads,
cache_layer_kv, gt_actions, compute_spans' token typing). The math is unchanged; what
differs is that those files import `alpamayo1_5` and assume one device. Here:

  - every gate lives on its own layer's card, because the 34B model is split by layer
    index over several cards (`run_oodval.layer_device_map`);
  - module paths are Alpamayo 2's: the expert tower is `model.expert.expert`, its
    projections and diffusion `model.expert.{action_in_proj, action_out_proj, diffusion,
    action_space}`;
  - the FM sweep can record the expert gate gradient per denoising step, which
    `run_step_importance.py --mode fm --noise-mode per_step` measured in a separate run
    on 1.5. The cache-leaf gradients (what the VLM backward is seeded with) accumulate over
    the steps exactly as in the shipped path; only the expert gates are read and zeroed
    after each step, and the shipped expert score is recovered as |sum_s g_s|.

The FM estimator is the shipped one: t_s = (s + 0.5) / n_steps, a fresh noise draw per
step from a CPU generator seeded with the clip seed, x_t = t x1 + (1 - t) eps and target
x1 - eps, which is Alpamayo 2's own training convention (diffusion/flow_matching.py).
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

from alpamayo2_super.models.alpamayo2_super import (
    MaskDiscreteTrajectoryLogitsProcessor,
    _append_text_eos_mask,
)
from alpamayo2_super.models.expert_utils import (
    StopAfterEOS,
    build_expert_pos_ids_and_attn_mask,
    find_eos_offset,
    replace_padding_after_eos,
)
from alpamayo2_super.models.token_utils import extract_text_tokens
from alpamayo2_super.models.utils import SPECIAL_TOKENS, fuse_traj_tokens

TYPES = ("vision", "hist", "prompt_text", "coc", "sink")  # same order as run_gradient_anatomy


def module_device(module):
    return next(module.parameters()).device


class GpuFirstOffload(torch.autograd.graph.saved_tensors_hooks):
    """Keep saved-for-backward tensors on their card while that card has room, else host RAM.

    torch.autograd.graph.save_on_cpu copies every saved tensor, parameters included: on this
    model that is 67 GB of weights duplicated into pinned host memory on each forward
    (measured 225 GB of host RAM per process, and the shared box started swapping). Here
    parameters are never copied (they already sit on the card), and activations stay on
    their card up to `budget[device_index]` bytes; only the overflow goes to host RAM.
    """

    def __init__(self, keep_ptrs, budget, pin=True):
        self.used = dict.fromkeys(budget, 0)
        self.to_host = 0

        def pack(t):
            if not t.is_cuda or t.untyped_storage().data_ptr() in keep_ptrs:
                return t
            n = t.numel() * t.element_size()
            d = t.device.index
            if self.used.get(d, 0) + n <= budget.get(d, 0):
                self.used[d] += n
                return t
            self.to_host += n
            cpu = torch.empty(t.size(), dtype=t.dtype, layout=t.layout, pin_memory=pin)
            cpu.copy_(t)
            return (t.device, cpu)

        def unpack(p):
            if isinstance(p, torch.Tensor):
                return p
            dev, cpu = p
            return cpu.to(dev, non_blocking=pin)

        super().__init__(pack, unpack)


def param_storage_ptrs(model):
    return {p.untyped_storage().data_ptr() for p in model.parameters()} | \
        {b.untyped_storage().data_ptr() for b in model.buffers()}


def cache_layer_kv(cache, i):
    return cache.layers[i].keys, cache.layers[i].values


def set_cache_layer_kv(cache, i, k, v):
    cache.layers[i].keys = k
    cache.layers[i].values = v


class UnitGates:
    """Multiplicative gates (value 1.0) on Q heads and MLP channels, one per unit and layer."""

    def __init__(self, layers, n_heads, head_dim, intermediate, mlp=True):
        self.n_heads, self.head_dim, self.intermediate = n_heads, head_dim, intermediate
        self.q_gates = [torch.ones(n_heads, device=module_device(ly), requires_grad=True)
                        for ly in layers]
        self.mlp_gates = [torch.ones(intermediate, device=module_device(ly), requires_grad=True)
                          for ly in layers] if mlp else []
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._make_q_hook(i)))
            if mlp:
                self._handles.append(
                    layer.mlp.down_proj.register_forward_pre_hook(self._make_mlp_hook(i)))

    def _make_q_hook(self, i):
        def hook(module, args):
            x = args[0]  # (B, T, H*D)
            b, t, _ = x.shape
            x = x.view(b, t, self.n_heads, self.head_dim) * self.q_gates[i].view(1, 1, -1, 1)
            return (x.view(b, t, -1),)
        return hook

    def _make_mlp_hook(self, i):
        def hook(module, args):
            return (args[0] * self.mlp_gates[i].view(1, 1, -1),)  # (B, T, I)
        return hook

    @staticmethod
    def _signed(gates, size):
        return np.stack([g.grad.float().cpu().numpy() if g.grad is not None else np.zeros(size)
                         for g in gates])

    def q_signed(self):
        return self._signed(self.q_gates, self.n_heads)  # (L, H)

    def mlp_signed(self):
        return self._signed(self.mlp_gates, self.intermediate)  # (L, I)

    def zero_grads(self):
        for g in self.q_gates + self.mlp_gates:
            g.grad = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class _TypedGate(torch.autograd.Function):
    """y = x * gate[type(p)], with dL/dgate split by the token type of position p.

    A Function so nothing of size (T, U) is saved for backward: only x is kept, which a
    plain gate multiplication keeps as well.
    """

    @staticmethod
    def _per_pos(x, gate, type_idx):
        g = gate.index_select(0, type_idx)  # (T, U)
        return g.view(1, x.shape[1], x.shape[2], *([1] * (x.dim() - 3)))  # (1, T, U[, 1])

    @staticmethod
    def forward(ctx, x, gate, type_idx):
        # x (1, T, U) or (1, T, U, D); gate (n_types, U) fp32; type_idx (T,) long
        ctx.save_for_backward(x, gate, type_idx)
        return x * _TypedGate._per_pos(x, gate, type_idx)

    @staticmethod
    def backward(ctx, grad_y):
        x, gate, type_idx = ctx.saved_tensors
        prod = grad_y.float() * x.float()  # (1, T, U[, D])
        if prod.dim() == 4:
            prod = prod.sum(-1)  # (1, T, U)
        grad_gate = torch.zeros_like(gate).index_add_(0, type_idx, prod[0])  # (n_types, U)
        return (grad_y * _TypedGate._per_pos(x, gate, type_idx)).to(x.dtype), grad_gate, None


class TypedUnitGates:
    """UnitGates with one gate per (token type, unit): the same gradient, split by position type.

    Summing a gate's grad over types gives the plain gate's grad, so the shipped score is
    |sum_types| of what this records. Call set_types() once per clip, before the forward.
    """

    def __init__(self, layers, n_heads, head_dim, intermediate, n_types):
        self.n_heads, self.head_dim, self.intermediate = n_heads, head_dim, intermediate
        self.devices = [module_device(ly) for ly in layers]
        self.type_idx = {}
        self.q_gates = [torch.ones(n_types, n_heads, device=d, requires_grad=True)
                        for d in self.devices]
        self.mlp_gates = [torch.ones(n_types, intermediate, device=d, requires_grad=True)
                          for d in self.devices]
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._make_q_hook(i)))
            self._handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(self._make_mlp_hook(i)))

    def set_types(self, type_idx):
        """type_idx: (T,) long, the token type of every position of the coming forward."""
        self.type_idx = {d: type_idx.to(d) for d in set(self.devices)}

    def _make_q_hook(self, i):
        def hook(module, args):
            x = args[0]  # (1, T, H*D)
            b, t, _ = x.shape
            y = _TypedGate.apply(x.view(b, t, self.n_heads, self.head_dim), self.q_gates[i],
                                 self.type_idx[self.devices[i]])  # (1, T, H, D)
            return (y.view(b, t, -1),)
        return hook

    def _make_mlp_hook(self, i):
        def hook(module, args):
            return (_TypedGate.apply(args[0], self.mlp_gates[i],
                                     self.type_idx[self.devices[i]]),)  # (1, T, I)
        return hook

    @staticmethod
    def _signed(gates):
        return np.stack([g.grad.float().cpu().numpy() if g.grad is not None
                         else np.zeros(tuple(g.shape)) for g in gates], 1)

    def q_signed(self):
        return self._signed(self.q_gates)  # (n_types, L, H)

    def mlp_signed(self):
        return self._signed(self.mlp_gates)  # (n_types, L, I)

    def zero_grads(self):
        for g in self.q_gates + self.mlp_gates:
            g.grad = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.inference_mode()
def rollout_sequence(model, model_inputs, max_gen):
    """The release CoC rollout (the generation half of `sample_trajectories_from_data`).

    Seed the RNG before calling. Returns (seq (1, T) = prompt + generated tokens up to and
    including <|traj_future_start|>, prompt_len, has_eos, cot text).
    """
    tokenized_data = dict(model_inputs["tokenized_data"])
    traj_data = {"ego_history_xyz": model_inputs["ego_history_xyz"],
                 "ego_history_rot": model_inputs["ego_history_rot"]}
    input_ids = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer,
                                 tokenized_data["input_ids"], traj_data, model.config.traj_ids)
    tokenized_data["input_ids"] = input_ids  # (1, L_prompt)

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.top_p = 0.98
    generation_config.temperature = 0.6
    generation_config.do_sample = True
    generation_config.max_new_tokens = max_gen
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    traj_ids = model.config.traj_ids
    logits_processor = LogitsProcessorList([MaskDiscreteTrajectoryLogitsProcessor(
        traj_token_offset=min(traj_ids["history_id0"], traj_ids["future_id0"]),
        traj_vocab_size=model.config.traj_vocab_size)])
    eos_token_id = traj_ids["future_start"]
    _append_text_eos_mask(logits_processor, generation_config.eos_token_id,
                          preserved_token_id=eos_token_id)
    out = model._generate_with_shared_prefill(
        tokenized_data, generation_config=generation_config, n_samples_total=1,
        stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)]),
        logits_processor=logits_processor)
    seq = replace_padding_after_eos(token_ids=out.sequences, eos_token_id=eos_token_id,
                                    pad_token_id=model.tokenizer.pad_token_id)  # (1, L)
    has_eos = bool((seq == eos_token_id).any())
    end = int(find_eos_offset(sequences=seq, eos_token_id=eos_token_id, device=seq.device,
                              warn=False)[0])
    cot = extract_text_tokens(model.tokenizer, seq)["cot"][0]
    return seq[:, :end].clone(), input_ids.shape[1], has_eos, cot


def token_types(model, seq, prompt_len):
    """(T,) long: index into TYPES for every position of the teacher-forced sequence.

    The rule of alpamayo-model-compression analysis_lib.compute_spans: vision = image pad
    tokens, hist = the discrete trajectory tokens after fusion, sink = position 0,
    coc = everything generated, prompt_text = the rest (camera and frame labels,
    vision start/end, the instruction).
    """
    ids = seq[0].cpu()  # (T,)
    tr = model.config.traj_ids
    lo = min(tr["history_id0"], tr["future_id0"])
    idx = torch.full((ids.shape[0],), TYPES.index("prompt_text"), dtype=torch.long)
    idx[ids == model.vlm.config.image_token_id] = TYPES.index("vision")
    idx[(ids >= lo) & (ids < lo + model.config.traj_vocab_size)] = TYPES.index("hist")
    idx[0] = TYPES.index("sink")
    idx[prompt_len:] = TYPES.index("coc")
    return idx


def special_ids(model):
    tok = model.tokenizer
    return {k: tok.convert_tokens_to_ids(SPECIAL_TOKENS[k])
            for k in ("cot_start", "cot_end", "traj_future_start")}


def vlm_forward_with_grad(model, seq, tokenized_data):
    """Teacher-forced VLM forward that builds the KV cache the expert reads. Grad enabled."""
    out = model.vlm.model(
        input_ids=seq, attention_mask=torch.ones_like(seq),
        pixel_values=tokenized_data["pixel_values"],
        image_grid_thw=tokenized_data["image_grid_thw"], use_cache=True)
    return out.last_hidden_state, out.past_key_values, out.rope_deltas


def coc_nll(model, hidden, seq, coc_start, coc_end):
    """CoC NLL over the generated positions; hidden at p predicts token p+1."""
    h = hidden[:, coc_start - 1 : coc_end - 1]  # (1, Tc, 5120)
    logits = model.vlm.lm_head(h).float()  # (1, Tc, V)
    return F.cross_entropy(logits[0], seq[0, coc_start:coc_end].to(logits.device))


def kv_group_scores(tensors):
    """Taylor score per KV group from retained cache grads. Returns (L, KV) twice."""
    n_kv = tensors[0][0].shape[1]
    k_s, v_s = np.zeros((len(tensors), n_kv)), np.zeros((len(tensors), n_kv))
    with torch.no_grad():
        for i, (k, v) in enumerate(tensors):
            if k.grad is not None:
                k_s[i] = (k * k.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
            if v.grad is not None:
                v_s[i] = (v * v.grad).abs().sum((0, 2, 3)).float().cpu().numpy()
    return k_s, v_s


def gt_actions(model, data, device):
    """GT future trajectory in the expert's normalised action space. (1, 64, 2)."""
    hx = data["ego_history_xyz"][:, 0].to(device).float()  # (1, 16, 3)
    hr = data["ego_history_rot"][:, 0].to(device).float()  # (1, 16, 3, 3)
    fx = data["ego_future_xyz"][:, 0].to(device).float()  # (1, 64, 3)
    fr = data["ego_future_rot"][:, 0].to(device).float()  # (1, 64, 3, 3)
    action = model.expert.action_space.traj_to_action(
        traj_history_xyz=hx, traj_history_rot=hr, traj_future_xyz=fx, traj_future_rot=fr)
    return action.reshape(1, *model.expert.action_space.get_action_space_dims())


def expert_fm_grads(model, cache, rope_deltas, x1, fm_steps, seed, prefill, step_gates=None):
    """FM loss backward through the expert onto detached cache leaves.

    Returns (mean FM loss, [(dL/dk, dL/dv)] per cache layer summed over steps, leaves,
    per-step signed expert gate grads {"q": (S, L, H), "mlp": (S, L, I)} or None).
    step_gates, when given, is the expert UnitGates: read and zeroed after every step.
    """
    dev0 = x1.device
    n_layers = len(model.expert.expert.layers)
    leaves = []
    for i in range(n_layers):
        k, v = cache_layer_kv(cache, i)
        leaves.append((k.detach().requires_grad_(True), v.detach().requires_grad_(True)))

    n_tok = model.expert.action_space.get_action_space_dims()[0]  # 64
    position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=torch.tensor([prefill], device=dev0), rope_deltas=rope_deltas.to(dev0),
        kv_cache_seq_len=prefill, n_diffusion_tokens=n_tok, b_star=1, device=dev0,
        prefix_mask=torch.ones(1, prefill, device=dev0, dtype=torch.long))
    forward_kwargs = {"is_causal": False} if model.expert.config.expert_non_causal_attention else {}

    losses, step_q, step_mlp = [], [], []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for s in range(fm_steps):
        t_val = (s + 0.5) / fm_steps
        noise = torch.randn(x1.shape, generator=gen).to(dev0)  # (1, 64, 2)
        x_t = t_val * x1 + (1.0 - t_val) * noise
        v_target = x1 - noise  # (1, 64, 2)
        t = torch.full((1, 1, 1), t_val, device=dev0)
        # the previous step's update()+crop() left graph-attached slices in the cache
        for i, (k, v) in enumerate(leaves):
            set_cache_layer_kv(cache, i, k, v)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.expert.action_in_proj(x_t, t)  # (1, 64, 1536)
            if embeds.dim() == 2:
                embeds = embeds.view(1, n_tok, -1)
            out = model.expert.expert(
                inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
                attention_mask=attention_mask, use_cache=True, **forward_kwargs)
            cache.crop(prefill)
            pred = model.expert.action_out_proj(out.last_hidden_state)  # (1, 64, 2)
        loss = F.mse_loss(pred.float(), v_target.to(pred.device))
        loss.backward()
        losses.append(loss.item())
        if step_gates is not None:
            step_q.append(step_gates.q_signed())
            step_mlp.append(step_gates.mlp_signed())
            step_gates.zero_grads()

    grads = [(k.grad, v.grad) for k, v in leaves]
    steps = {"q": np.stack(step_q), "mlp": np.stack(step_mlp)} if step_gates is not None else None
    return float(np.mean(losses)), grads, leaves, steps


def cache_read(leaves, grads, type_idx, n_types):
    """|k dL/dk| + |v dL/dv| on the expert-side leaves, by (cache layer, position type). (L, n_types)."""
    d = np.zeros((len(leaves), n_types))
    with torch.no_grad():
        for m, ((k, v), (gk, gv)) in enumerate(zip(leaves, grads)):
            if gk is None or gv is None:
                raise RuntimeError(f"expert left no gradient on cache layer {m}")
            a = (k * gk).abs().sum((0, 1, 3)) + (v * gv).abs().sum((0, 1, 3))  # (T,)
            ti = type_idx.to(a.device)
            d[m] = torch.zeros(n_types, device=a.device, dtype=torch.float64).index_add_(
                0, ti, a.double()).cpu().numpy()
    return d
