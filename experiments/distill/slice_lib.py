"""Structured slicing of Alpamayo 2 Super into a smaller `Alpamayo2Super` checkpoint.

A `PruneSpec` lists what survives, per axis:
  layers   decoder layer indices kept in BOTH towers (expert layer i reads VLM cache layer i)
  d        VLM residual channels (embeddings, every VLM layer, final norm, lm_head, and the
           output rows of the ViT merger + deepstack mergers)
  kv       per kept layer, KV groups kept; shared by the two towers because expert K/V are
           appended to the VLM cache of the same layer
  q, mlp   per kept layer, VLM query heads (listed group by group in `kv` order, the same
           count under every kept group) and MLP channels
  d_e, q_e, mlp_e   the same for the expert (q_e listed in `kv` order as well)
  vocab    kept token ids (None = all); embeddings and lm_head rows, ids remapped by order
ViT blocks are never sliced. head_dim stays 128, so q_norm / k_norm are copied as they are.

Qwen3-VL injects the deepstack features into decoder layers 0..2, so a spec that drops any of
them changes what those features feed (validate() warns).
"""

import copy
import glob
import json
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

TEACHER_SNAPSHOT = glob.glob(
    "/mnt/nvme1n1/ad_vla/cache/hub/models--nvidia--Alpamayo2-Super/snapshots/00554695*/")[0]
HD = 128  # head_dim, both towers
COPY_FILES = ("added_tokens.json", "chat_template.jinja", "generation_config.json", "merges.txt",
              "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json",
              "tokenizer.json", "video_preprocessor_config.json", "vocab.json")
VLM = "vlm.model.language_model"
EXP = "expert.expert"
MERGERS = ["vlm.model.visual.merger"] + [f"vlm.model.visual.deepstack_merger_list.{i}"
                                          for i in range(3)]


@dataclass
class PruneSpec:
    layers: list
    d: np.ndarray
    kv: list
    q: list
    mlp: list
    d_e: np.ndarray
    q_e: list
    mlp_e: list
    vocab: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def counts(self):
        return {"L": len(self.layers), "d": len(self.d), "nkv": len(self.kv[0]),
                "nq": len(self.q[0]), "m": len(self.mlp[0]), "d_e": len(self.d_e),
                "nq_e": len(self.q_e[0]), "m_e": len(self.mlp_e[0]),
                "V": None if self.vocab is None else len(self.vocab)}

    def validate(self, tcfg):
        t = tcfg["vlm_config"]["text_config"]
        e = tcfg["expert_config"]["llm_config"]
        rep, rep_e = t["num_attention_heads"] // t["num_key_value_heads"], \
            e["num_attention_heads"] // e["num_key_value_heads"]
        L = len(self.layers)
        assert self.layers == sorted(set(self.layers)) and self.layers[-1] < t["num_hidden_layers"]
        for name in ("kv", "q", "mlp", "q_e", "mlp_e"):
            v = getattr(self, name)
            assert len(v) == L, name
            assert len({len(x) for x in v}) == 1, f"{name}: unequal counts across layers"
        for name, lim in (("d", t["hidden_size"]), ("d_e", e["hidden_size"])):
            v = getattr(self, name)
            assert np.all(np.diff(v) > 0) and v[-1] < lim, name
        for kv, q, q_e in zip(self.kv, self.q, self.q_e):
            for heads, r in ((q, rep), (q_e, rep_e)):
                per = len(heads) // len(kv)
                assert per * len(kv) == len(heads)
                for p, g in enumerate(kv):  # heads listed group by group, in kv order
                    assert all(h // r == g for h in heads[p * per:(p + 1) * per])
        if self.vocab is not None:
            assert np.all(np.diff(self.vocab) > 0)
        if self.layers[:3] != [0, 1, 2]:
            warnings.warn("layers 0..2 not all kept: deepstack features land on other layers")

    def save(self, path):
        np.savez(path, layers=np.asarray(self.layers), d=self.d, kv=np.stack(self.kv),
                 q=np.stack(self.q), mlp=np.stack(self.mlp), d_e=self.d_e, q_e=np.stack(self.q_e),
                 mlp_e=np.stack(self.mlp_e),
                 vocab=np.asarray([] if self.vocab is None else self.vocab),
                 meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(layers=z["layers"].tolist(), d=z["d"], kv=list(z["kv"]), q=list(z["q"]),
                   mlp=list(z["mlp"]), d_e=z["d_e"], q_e=list(z["q_e"]), mlp_e=list(z["mlp_e"]),
                   vocab=z["vocab"] if len(z["vocab"]) else None, meta=json.loads(str(z["meta"])))


def teacher_config():
    return json.loads((Path(TEACHER_SNAPSHOT) / "config.json").read_text())


def keep_all(tcfg):
    t, e = tcfg["vlm_config"]["text_config"], tcfg["expert_config"]["llm_config"]
    L = t["num_hidden_layers"]
    rng = lambda n: np.arange(n)
    return PruneSpec(layers=list(range(L)), d=rng(t["hidden_size"]),
                     kv=[rng(t["num_key_value_heads"])] * L, q=[rng(t["num_attention_heads"])] * L,
                     mlp=[rng(t["intermediate_size"])] * L, d_e=rng(e["hidden_size"]),
                     q_e=[rng(e["num_attention_heads"])] * L, mlp_e=[rng(e["intermediate_size"])] * L)


def head_rows(heads):
    """Head indices -> row indices of a (n_heads * 128, ...) projection."""
    return (np.asarray(heads)[:, None] * HD + np.arange(HD)[None]).reshape(-1)


class TeacherWeights:
    """Lazy reader over a checkpoint's safetensors (sharded teacher or single-file student)."""

    def __init__(self, snapshot=TEACHER_SNAPSHOT):
        self.snapshot = Path(snapshot)
        index = self.snapshot / "model.safetensors.index.json"
        if index.exists():
            self.where = json.loads(index.read_text())["weight_map"]
        else:  # a student checkpoint: one file
            with safe_open(str(self.snapshot / "model.safetensors"), framework="pt") as f:
                self.where = dict.fromkeys(f.keys(), "model.safetensors")
        self._handles = {}

    def __call__(self, name):
        f = self.where[name]
        if f not in self._handles:
            self._handles[f] = safe_open(str(self.snapshot / f), framework="pt", device="cpu")
        return self._handles[f].get_tensor(name)

    def names(self):
        return list(self.where)


def _ix(t, rows=None, cols=None):
    if rows is not None:
        t = t.index_select(0, torch.as_tensor(np.asarray(rows), dtype=torch.long))
    if cols is not None:
        t = t.index_select(1, torch.as_tensor(np.asarray(cols), dtype=torch.long))
    return t.contiguous()


def required_ids(tcfg):
    """Ids every pruned vocab keeps: Qwen's special tokens (151643..), the history-trajectory
    block (kept whole so it stays contiguous after the remap) and the trajectory markers. The
    3000 future-trajectory ids only serve discrete future decoding, which the expert path masks."""
    tid = tcfg["traj_ids"]
    qwen_special = np.arange(151643, tid["history_id0"])
    hist = np.arange(tid["history_id0"], tid["history_id0"] + tcfg["history_vocab_size"])
    markers = np.asarray([v for k, v in tid.items() if not k.endswith("_id0")])
    return np.unique(np.concatenate([qwen_special, hist, markers]))


def remap_ids(spec, size):
    """old id -> new id (-1 for dropped) over the source's whole vocabulary of `size` ids, or
    None when the vocab is kept whole. (Sized by the source vocab, not the largest kept id:
    the teacher's top-k can name any id, padding ids included.)"""
    if spec.vocab is None:
        return None
    m = np.full(max(int(size), int(spec.vocab[-1]) + 1), -1, dtype=np.int64)
    m[spec.vocab] = np.arange(len(spec.vocab))
    return m


def student_config(tcfg, spec):
    c = copy.deepcopy(tcfg)
    n = spec.counts()
    t, v = c["vlm_config"]["text_config"], c["vlm_config"]["vision_config"]
    t.update(hidden_size=n["d"], num_hidden_layers=n["L"], num_attention_heads=n["nq"],
             num_key_value_heads=n["nkv"], intermediate_size=n["m"])
    v["out_hidden_size"] = n["d"]
    e = c["expert_config"]["llm_config"]
    e.update(hidden_size=n["d_e"], num_hidden_layers=n["L"], num_attention_heads=n["nq_e"],
             num_key_value_heads=n["nkv"], intermediate_size=n["m_e"])
    c["expert_config"]["expert_update_cfg"] = {"head_dim": HD, "hidden_size": n["d_e"],
                                               "intermediate_size": n["m_e"],
                                               "num_attention_heads": n["nq_e"]}
    m = remap_ids(spec, t["vocab_size"])
    if m is not None:
        t["vocab_size"] = e["vocab_size"] = n["V"]

        def new(i):
            assert i < len(m) and m[i] >= 0, f"special id {i} not kept"
            return int(m[i])

        for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            c["vlm_config"][k] = new(c["vlm_config"][k])
        for cfg in (t, e):
            cfg["bos_token_id"], cfg["eos_token_id"] = new(cfg["bos_token_id"]), new(cfg["eos_token_id"])
        tid = {k: new(i) for k, i in c["traj_ids"].items() if not k.startswith("future_id")}
        # the 3000 discrete future ids are gone: an empty future block right after the history
        # block keeps every traj_ids user (logits mask, label masks) valid and history-only
        tid["future_id0"] = tid["history_id0"] + c["history_vocab_size"]
        c["traj_ids"] = tid
        c["future_vocab_size"] = 0
    c["prune_spec_counts"] = n
    return c


def slice_weights(tw, spec):
    """-> {student name: tensor}. Layer i of the student is teacher layer spec.layers[i]."""
    sd = {}
    d, d_e, voc = spec.d, spec.d_e, spec.vocab
    for name in tw.names():
        if name.startswith("vlm.model.visual."):
            if any(name.startswith(p + ".linear_fc2.") for p in MERGERS):
                sd[name] = _ix(tw(name), rows=d)  # (d, 4608) / (d,)
            else:
                sd[name] = tw(name)
    for name in (f"{VLM}.embed_tokens.weight", "vlm.lm_head.weight"):
        sd[name] = _ix(tw(name), rows=voc, cols=d)
    sd[f"{VLM}.norm.weight"] = _ix(tw(f"{VLM}.norm.weight"), rows=d)

    for i, src in enumerate(spec.layers):
        kv, q, mlp = spec.kv[i], spec.q[i], spec.mlp[i]
        p, s = f"{VLM}.layers.{i}", f"{VLM}.layers.{src}"
        sd[f"{p}.input_layernorm.weight"] = _ix(tw(f"{s}.input_layernorm.weight"), rows=d)
        sd[f"{p}.post_attention_layernorm.weight"] = _ix(tw(f"{s}.post_attention_layernorm.weight"),
                                                         rows=d)
        sd[f"{p}.self_attn.q_proj.weight"] = _ix(tw(f"{s}.self_attn.q_proj.weight"),
                                                 rows=head_rows(q), cols=d)
        for k in ("k_proj", "v_proj"):
            sd[f"{p}.self_attn.{k}.weight"] = _ix(tw(f"{s}.self_attn.{k}.weight"),
                                                  rows=head_rows(kv), cols=d)
        sd[f"{p}.self_attn.o_proj.weight"] = _ix(tw(f"{s}.self_attn.o_proj.weight"), rows=d,
                                                 cols=head_rows(q))
        for k in ("q_norm", "k_norm"):
            sd[f"{p}.self_attn.{k}.weight"] = tw(f"{s}.self_attn.{k}.weight")
        for k in ("gate_proj", "up_proj"):
            sd[f"{p}.mlp.{k}.weight"] = _ix(tw(f"{s}.mlp.{k}.weight"), rows=mlp, cols=d)
        sd[f"{p}.mlp.down_proj.weight"] = _ix(tw(f"{s}.mlp.down_proj.weight"), rows=d, cols=mlp)

        q_e, mlp_e = spec.q_e[i], spec.mlp_e[i]
        p, s = f"{EXP}.layers.{i}", f"{EXP}.layers.{src}"
        sd[f"{p}.input_layernorm.weight"] = _ix(tw(f"{s}.input_layernorm.weight"), rows=d_e)
        sd[f"{p}.post_attention_layernorm.weight"] = _ix(tw(f"{s}.post_attention_layernorm.weight"),
                                                         rows=d_e)
        sd[f"{p}.self_attn.q_proj.weight"] = _ix(tw(f"{s}.self_attn.q_proj.weight"),
                                                 rows=head_rows(q_e), cols=d_e)
        for k in ("k_proj", "v_proj"):
            sd[f"{p}.self_attn.{k}.weight"] = _ix(tw(f"{s}.self_attn.{k}.weight"),
                                                  rows=head_rows(kv), cols=d_e)
        sd[f"{p}.self_attn.o_proj.weight"] = _ix(tw(f"{s}.self_attn.o_proj.weight"), rows=d_e,
                                                 cols=head_rows(q_e))
        for k in ("q_norm", "k_norm"):
            sd[f"{p}.self_attn.{k}.weight"] = tw(f"{s}.self_attn.{k}.weight")
        for k in ("gate_proj", "up_proj"):
            sd[f"{p}.mlp.{k}.weight"] = _ix(tw(f"{s}.mlp.{k}.weight"), rows=mlp_e, cols=d_e)
        sd[f"{p}.mlp.down_proj.weight"] = _ix(tw(f"{s}.mlp.down_proj.weight"), rows=d_e, cols=mlp_e)

    sd[f"{EXP}.norm.weight"] = _ix(tw(f"{EXP}.norm.weight"), rows=d_e)
    for name in tw.names():
        if name.endswith(".freqs"):  # Fourier buffers a trained student's checkpoint carries;
            continue                  # the model rebuilds them from its config
        if name.startswith("expert.action_in_proj."):
            if name.startswith(("expert.action_in_proj.encoder.trunk.6.", "expert.action_in_proj.norm.")):
                sd[name] = _ix(tw(name), rows=d_e)  # last trunk Linear (d_e, 512) + LayerNorm(d_e)
            else:
                sd[name] = tw(name)
    sd["expert.action_out_proj.weight"] = _ix(tw("expert.action_out_proj.weight"), cols=d_e)
    sd["expert.action_out_proj.bias"] = tw("expert.action_out_proj.bias")
    return sd


def count_params(sd):
    """Totals by part, in parameters."""
    parts = {"vit": 0, "decoder_layers": 0, "embed_head": 0, "expert": 0, "other": 0}
    for k, v in sd.items():
        if k.startswith("vlm.model.visual."):
            parts["vit"] += v.numel()
        elif k.startswith(f"{VLM}.layers.") or k == f"{VLM}.norm.weight":
            parts["decoder_layers"] += v.numel()
        elif k in (f"{VLM}.embed_tokens.weight", "vlm.lm_head.weight"):
            parts["embed_head"] += v.numel()
        elif k.startswith("expert."):
            parts["expert"] += v.numel()
        else:
            parts["other"] += v.numel()
    parts["total"] = sum(parts.values())
    parts["budget_excl_vit"] = parts["total"] - parts["vit"]
    return parts


def formula_params(spec, V):
    """Closed-form count of the budget part (plan section 4.3), for the G0b cross-check."""
    n = spec.counts()
    layer = lambda d, nq, nkv, m: d * HD * nq * 2 + 2 * d * HD * nkv + 3 * d * m + 2 * d + 2 * HD
    vlm = n["L"] * layer(n["d"], n["nq"], n["nkv"], n["m"]) + n["d"]
    exp = n["L"] * layer(n["d_e"], n["nq_e"], n["nkv"], n["m_e"]) + n["d_e"]
    head = 2 * (n["V"] or V) * n["d"]
    proj = 512 * 60 + 512 + 512 + 512 * 512 + 512 + 512 + n["d_e"] * 512 + n["d_e"] \
        + 2 * n["d_e"] + 2 * n["d_e"] + 2
    return {"decoder_layers": vlm, "embed_head": head, "expert": exp + proj,
            "budget_excl_vit": vlm + head + exp + proj}


def save_student(out_dir, tcfg, spec, sd):
    """Write a from_pretrained-loadable checkpoint dir plus the spec and the vocab map."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = student_config(tcfg, spec)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in COPY_FILES:
        shutil.copy(Path(TEACHER_SNAPSHOT) / f, out / f)
    spec.save(out / "prune_spec.npz")
    m = remap_ids(spec, tcfg["vlm_config"]["text_config"]["vocab_size"])
    if m is not None:
        np.save(out / "vocab_map.npy", m)
    return cfg
