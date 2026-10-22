"""LiDAR range-image branch of the camera + LiDAR student (plan 2026-10-20, section 4.2).

RangeViT turns one deskewed range image (4, 128, 1024) into 256 LLM tokens:
  patch 8 x 16 -> 16 x 64 patches, 6 pre-norm blocks (d 384, 6 heads), 2 x 2 merge -> 8 x 32,
  MLP to the LLM width.
Positional information, all inside the encoder:
  additive  Fourier features of the patch azimuth (sin/cos k*phi, k = 1..8: periodic, so the
            +-180 deg seam is continuous; forward = 0), of the beams' measured elevation, and
            of the patch's capture time dt = t0 - t (ms; the spin sweeps azimuth over 100 ms,
            so time is NOT periodic and jumps at the seam), through an MLP
  rotary    2D RoPE in attention: elevation rows with standard frequencies, azimuth columns
            with cyclic frequencies 2*pi*k / 64, so column 64 rotates like column 0
In the LLM the 256 tokens sit between <|vision_start|> and <|vision_end|> as <|video_pad|>
placeholders right after the last camera image; Qwen3-VL's get_rope_index then gives them an
(t=0, h=row, w=column) MRoPE grid from video_grid_thw = (1, 16, 64).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

H, W = 128, 1024
PATCH = (8, 16)
GRID = (H // PATCH[0], W // PATCH[1])  # (16, 64)
MERGE = 2
N_TOKENS = (GRID[0] // MERGE) * (GRID[1] // MERGE)  # 256
VIDEO_GRID_THW = (1, GRID[0], GRID[1])


def fourier(x, n, scale):
    """x (...,) -> (..., 2n) sin/cos at frequencies scale * 2^k, k = 0..n-1."""
    f = scale * (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype))
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], dim=-1)


def patch_positions(dt_col, beam_elev):
    """dt_col (B, 1024) ms, beam_elev (B, 128) deg -> positional features (B, 16*64, 51)."""
    B = dt_col.shape[0]
    gh, gw = GRID
    phi = -math.pi + (torch.arange(gw, device=dt_col.device, dtype=torch.float32) + 0.5) * (2 * math.pi / gw)
    k = torch.arange(1, 9, device=dt_col.device, dtype=torch.float32)
    az = torch.cat([(phi[:, None] * k).sin(), (phi[:, None] * k).cos()], dim=-1)  # (64, 16) periodic
    elev = beam_elev.float().view(B, gh, PATCH[0]).mean(-1) / 30.0  # (B, 16)
    el = torch.cat([elev[..., None], fourier(elev, 8, math.pi)], dim=-1)  # (B, 16, 17)
    dt = dt_col.float().view(B, gw, PATCH[1]).mean(-1) / 200.0  # (B, 64)
    tm = torch.cat([dt[..., None], fourier(dt, 8, math.pi)], dim=-1)  # (B, 64, 17)
    feats = torch.cat([az[None, None].expand(B, gh, -1, -1),
                       el[:, :, None].expand(-1, -1, gw, -1),
                       tm[:, None].expand(-1, gh, -1, -1)], dim=-1)  # (B, 16, 64, 50)
    return feats.reshape(B, gh * gw, -1)


def rope_tables(head_dim, device):
    """cos/sin (16*64, head_dim): first half rotates with the row, second half with the column."""
    gh, gw = GRID
    q = head_dim // 4  # rotation pairs per axis
    row_f = 1.0 / (100.0 ** (torch.arange(q, device=device, dtype=torch.float32) / q))
    col_f = 2 * math.pi * torch.arange(1, q + 1, device=device, dtype=torch.float32) / gw  # cyclic
    r = torch.arange(gh, device=device, dtype=torch.float32)[:, None, None] * row_f  # (16, 1, q)
    c = torch.arange(gw, device=device, dtype=torch.float32)[None, :, None] * col_f  # (1, 64, q)
    ang = torch.cat([r.expand(gh, gw, q), c.expand(gh, gw, q)], dim=-1).reshape(gh * gw, 2 * q)
    ang = torch.cat([ang, ang], dim=-1)  # (N, head_dim)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x (B, heads, N, d)."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x, cos, sin):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, d)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.n2(x))


class RangeViT(nn.Module):
    def __init__(self, out_dim, in_ch=4, dim=384, depth=6, heads=6):
        super().__init__()
        self.config = {"out_dim": out_dim, "in_ch": in_ch, "dim": dim, "depth": depth, "heads": heads}
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=PATCH, stride=PATCH)
        self.pos_mlp = nn.Sequential(nn.Linear(50, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.merger = nn.Sequential(nn.LayerNorm(dim * MERGE * MERGE),
                                    nn.Linear(dim * MERGE * MERGE, dim * MERGE * MERGE), nn.GELU(),
                                    nn.Linear(dim * MERGE * MERGE, out_dim))
        self.head_dim = dim // heads

    def forward(self, img, dt_col, beam_elev):
        """img (B, 4, 128, 1024), dt_col (B, 1024), beam_elev (B, 128) -> (B, 256, out_dim)."""
        x = self.patch_embed(img).flatten(2).transpose(1, 2)  # (B, 1024, dim) row-major (16 x 64)
        x = x + self.pos_mlp(patch_positions(dt_col, beam_elev).to(x.dtype))
        cos, sin = rope_tables(self.head_dim, x.device)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        B, gh, gw, D = x.shape[0], GRID[0], GRID[1], x.shape[-1]
        x = x.view(B, gh // MERGE, MERGE, gw // MERGE, MERGE, D).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, (gh // MERGE) * (gw // MERGE), MERGE * MERGE * D)  # (B, 256, 4 dim)
        return self.merger(x)


def insert_lidar_tokens(input_ids, vision_start, vision_end, video_pad):
    """Place <vision_start> <video_pad> x 256 <vision_end> right after the last camera image.
    input_ids (1, L) -> (1, L + 258); returns the new ids and the insertion index."""
    ids = input_ids[0]
    ends = torch.nonzero(ids == vision_end).flatten()
    at = int(ends[-1]) + 1
    block = torch.tensor([vision_start] + [video_pad] * N_TOKENS + [vision_end], dtype=ids.dtype)
    return torch.cat([ids[:at], block, ids[at:]]).view(1, -1), at


def lidar_vlm_forward(model, lidar_enc, ids, attention_mask, pixel_values, image_grid_thw, lidar,
                      **kw):
    """model.vlm(...) with the range tokens scattered into the <video_pad> slots and MRoPE positions
    computed with the range grid; sets model.vlm.model.rope_deltas for the expert."""
    vcfg = model.config.vlm_config
    emb = model.vlm.get_input_embeddings()(ids)  # (1, L, d)
    tok = lidar_enc(lidar["range_img"], lidar["dt_col"], lidar["beam_elev"])  # (1, 256, d)
    slot = (ids == vcfg.video_token_id)[..., None]
    emb = emb.masked_scatter(slot, tok.to(emb.dtype))
    vgrid = torch.tensor([VIDEO_GRID_THW], device=ids.device)
    pos, deltas = model.vlm.model.get_rope_index(ids, image_grid_thw, vgrid, attention_mask)
    model.vlm.model.rope_deltas = deltas
    return model.vlm(inputs_embeds=emb, attention_mask=attention_mask, position_ids=pos,
                     pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kw)


def lidar_tensors(cached, device):
    """build_lidar_distill.load_cached dict (numpy, or tensors FSDP already moved) -> batched
    tensors on device."""
    def t(x):
        return (x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))).to(device)[None]

    return {k: t(cached[k]) for k in ("range_img", "dt_col", "beam_elev")}
