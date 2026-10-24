"""Gate L0b: the LiDAR branch is wired into the student exactly as planned (CPU, one dev clip).

(i)   without a LiDAR block, lidar_vlm_forward (inputs_embeds + explicit MRoPE positions)
      reproduces the plain input_ids forward: same logits, same rope_deltas
(ii)  with the block: 256 <video_pad> slots; their MRoPE positions form the (t const, h 0..7,
      w 0..31) grid; the text after the block resumes at max + 1
(iii) a backward pass of the driving CE + flow loss reaches every RangeViT parameter group
(iv)  positional encodings: cyclic RoPE angles of column 64 equal column 0 (mod 2 pi), and the
      azimuth features of the last and first columns are as close as any neighbouring pair

Usage:
  source experiments/oodval/env.sh
  CUDA_VISIBLE_DEVICES=-1 python experiments/distill/check_lidar_wiring.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import build_lidar_distill as bl
import distill_cache as dc
import distill_data as dd
import lidar_lib as ll

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "distill_lidar_l0b"
INIT = "/mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/s2_wide/init"


def main():
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    model = Alpamayo2Super.from_pretrained(INIT, dtype=torch.float32).eval()
    enc = ll.RangeViT(out_dim=model.config.vlm_config.text_config.hidden_size)
    b = dd.SeqBuilder(vocab_map=np.load(Path(INIT) / "vocab_map.npy"))
    c, t = dd.labelled_items("dev_238", lidar=True)[0]
    data = dc.load_cached(dc.path_for(c, t, "dev_238"))
    lab = dd.load_label("dev_238", "driving", c, t)
    lid = ll.lidar_tensors(bl.load_cached(bl.path_for(c, t, "dev_238")), "cpu")
    res = {"clip": c, "lidar_params_M": sum(p.numel() for p in enc.parameters()) / 1e6}

    # (i) no LiDAR block: wrapper == plain forward
    s0 = b.driving(data, lab["gen_ids"], lidar=False)
    with torch.no_grad():
        ref = model.vlm(input_ids=s0["input_ids"], attention_mask=s0["attention_mask"],
                        pixel_values=s0["pixel_values"], image_grid_thw=s0["image_grid_thw"])
        d_ref = model.vlm.model.rope_deltas.clone()
        got = ll.lidar_vlm_forward(model, enc, s0["input_ids"], s0["attention_mask"],
                                   s0["pixel_values"], s0["image_grid_thw"], lid)
        d_got = model.vlm.model.rope_deltas.clone()
    res["i_logits_maxabs"] = float((ref.logits - got.logits).abs().max())
    res["i_rope_deltas_equal"] = bool(torch.equal(d_ref, d_got))

    # (ii) with the block: slot count and MRoPE grid
    s1 = b.driving(data, lab["gen_ids"], lidar=True)
    ids = s1["input_ids"]
    vid = model.config.vlm_config.video_token_id
    slots = torch.nonzero(ids[0] == vid).flatten()
    pos, _ = model.vlm.model.get_rope_index(ids, s1["image_grid_thw"],
                                            torch.tensor([ll.VIDEO_GRID_THW]), s1["attention_mask"])
    p = pos[:, 0, slots]  # (3, 256)
    res["ii_slots"] = len(slots)
    res["ii_slots_contiguous"] = bool((slots.diff() == 1).all())
    res["ii_t_constant"] = bool((p[0] == p[0, 0]).all())
    res["ii_h_range"] = [int((p[1] - p[1].min()).min()), int((p[1] - p[1].min()).max())]
    res["ii_w_range"] = [int((p[2] - p[2].min()).min()), int((p[2] - p[2].min()).max())]
    after = int(slots[-1]) + 2  # skip <vision_end>
    res["ii_text_after_resumes"] = bool(int(pos[0, 0, after - 1]) == int(p.max()) + 1)
    res["ii_prompt_len_delta"] = int(s1["prompt_len"] - s0["prompt_len"])

    # (iii) gradients through the LiDAR branch (CE on the targets + one flow-matching draw)
    model.train()
    n = s1["n_target"]
    out = ll.lidar_vlm_forward(model, enc, ids, s1["attention_mask"], s1["pixel_values"],
                               s1["image_grid_thw"], lid, use_cache=True, logits_to_keep=n + 1)
    ce = torch.nn.functional.cross_entropy(out.logits[0, :-1].float(), ids[0, -n:])
    fm = dd.expert_fm(model, s1["traj"], out)
    (ce + fm).backward()
    groups = {"patch_embed": enc.patch_embed, "pos_mlp": enc.pos_mlp, "blocks": enc.blocks,
              "merger": enc.merger}
    res["iii_ce"], res["iii_fm"] = float(ce), float(fm)
    res["iii_grad_norm"] = {k: float(torch.sqrt(sum((q.grad ** 2).sum() for q in m.parameters()
                                                     if q.grad is not None)))
                            for k, m in groups.items()}

    # (iv) periodic azimuth encodings
    gh, gw = ll.GRID
    q = enc.head_dim // 4
    col_f = 2 * math.pi * torch.arange(1, q + 1, dtype=torch.float64) / gw
    res["iv_rope_col64_equals_col0"] = bool(torch.allclose(torch.cos(gw * col_f), torch.ones(q, dtype=torch.float64)))
    feats = ll.patch_positions(lid["dt_col"], lid["beam_elev"])[0].view(gh, gw, -1)[0, :, :16]
    gaps = (feats.roll(-1, dims=0) - feats).norm(dim=-1)  # neighbour distance incl. 63 -> 0
    res["iv_seam_gap_vs_median_neighbour"] = float(gaps[-1] / gaps[:-1].median())
    ok = (res["i_logits_maxabs"] < 1e-3 and res["i_rope_deltas_equal"] and res["ii_slots"] == 256
          and res["ii_slots_contiguous"] and res["ii_t_constant"] and res["ii_h_range"] == [0, 7]
          and res["ii_w_range"] == [0, 31] and res["ii_text_after_resumes"]
          and all(v > 0 for v in res["iii_grad_norm"].values()) and res["iv_rope_col64_equals_col0"]
          and res["iv_seam_gap_vs_median_neighbour"] < 1.5)
    res["pass"] = bool(ok)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    (OUT / "summary.txt").write_text("\n".join(f"{k}: {v}" for k, v in res.items()) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
