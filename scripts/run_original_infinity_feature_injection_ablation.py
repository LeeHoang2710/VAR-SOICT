#!/usr/bin/env python3
"""Step-wise feature-injection ablation for original Infinity-2B weights.

This script compares three intervention operators for a content prompt plus a
style reference image:

  A. raw residual:              G_s + lambda * (S_s - G_s)
  B. SVD residual:              G_s + lambda * (svd(S_s) - svd(G_s))
  C. content-projected PFB:     G_s + lambda * (I - eta Pc)[svd(S_s) - svd(G_s)]

For every sampled prompt/style pair, it injects at exactly one scale at a time
and saves the final image.  It then reports:

  - content_similarity: CLIP(image, content prompt)
  - style_similarity:   CSD-S(image, style reference)
  - s_harmonic:         harmonic mean of content/style similarity

The implementation intentionally uses the original Infinity .pth weights under
weights/ and does not touch the GGUF/quantized path.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

METHODS = {
    "raw_residual": "Raw style residual",
    "svd_residual": "Top-k SVD style residual",
    "content_projection": "Content-projected top-k PFB",
}


@dataclass(frozen=True)
class OutputDirs:
    root: Path
    generated: Path
    grids: Path
    plots: Path
    metrics: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--infinity-dir", type=Path, default=PROJECT_ROOT / "Infinity")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--prompts-csv", type=Path, default=PROJECT_ROOT / "prompts/content_prompts_190.csv")
    parser.add_argument("--styles-csv", type=Path, default=PROJECT_ROOT / "styles/quantitative_eval_styles_10.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/original_infinity_feature_injection_ablation"))
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "1M"), default="0.25M")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--style-rank", type=int, default=1)
    parser.add_argument(
        "--content-rank",
        type=int,
        default=1,
        help="Fixed content-basis rank. Use 0 with --content-variance-threshold for adaptive rank without a cap.",
    )
    parser.add_argument(
        "--content-variance-threshold",
        type=float,
        default=None,
        help="Optional cumulative content-basis energy in (0,1], e.g. 0.9 for adaptive rank.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument(
        "--preserve-mean",
        action="store_true",
        help="Add a raw mean/palette delta before content projection. Off by default to match content_basis_rank.",
    )
    parser.add_argument("--vae-type", type=int, default=32)
    parser.add_argument("--model-type", type=str, default="infinity_2b")
    parser.add_argument("--text-channels", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip-model-id", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--csd-model-id", type=str, default=os.environ.get("CSD_EVAL_MODEL_ID", ""))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--package-zip", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.style_rank < 1:
        parser.error("--style-rank must be positive")
    if args.content_rank < 0:
        parser.error("--content-rank must be non-negative")
    if args.content_rank == 0 and args.content_variance_threshold is None:
        parser.error("--content-rank 0 requires --content-variance-threshold")
    if args.content_variance_threshold is not None and not 0 < args.content_variance_threshold <= 1:
        parser.error("--content-variance-threshold must be in (0, 1]")
    if not 0 <= args.projection_strength <= 1:
        parser.error("--projection-strength must be in [0, 1]")
    return args


def build_output_dirs(output_dir: Path) -> OutputDirs:
    dirs = OutputDirs(
        root=output_dir,
        generated=output_dir / "generated_step_images",
        grids=output_dir / "step_grids",
        plots=output_dir / "plots",
        metrics=output_dir / "metrics",
    )
    for directory in dirs.__dict__.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_")
    return (slug[:max_len] or "item").lower()


def resolve_path(workspace_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace_dir / path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def resolve_infinity_dir(workspace_dir: Path, explicit: Path) -> Path:
    candidates = [explicit, workspace_dir / "Infinity", Path.cwd() / "Infinity"]
    for candidate in candidates:
        if candidate and (candidate / "tools" / "run_infinity.py").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find Infinity/tools/run_infinity.py. Pass --infinity-dir.")


def resolve_weights_dir(workspace_dir: Path, weights_dir: Path) -> Path:
    weights_dir = resolve_path(workspace_dir, weights_dir)
    required = ["infinity_2b_reg.pth", "infinity_vae_d32reg.pth", "flan-t5-xl"]
    missing = [name for name in required if not (weights_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} under weights directory: {weights_dir}")
    return weights_dir.resolve()


def import_runtime(infinity_dir: Path):
    for import_dir in (str(SRC_DIR), str(infinity_dir), str(infinity_dir / "tools")):
        if import_dir not in sys.path:
            sys.path.insert(0, import_dir)

    from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
    from run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer
    from var_soict.feature_hypotheses import truncated_svd

    return SimpleNamespace(
        sample_with_top_k_top_p=sample_with_top_k_top_p_also_inplace_modifying_logits_,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
        h_div_w_templates=h_div_w_templates,
        load_tokenizer=load_tokenizer,
        load_transformer=load_transformer,
        load_visual_tokenizer=load_visual_tokenizer,
        truncated_svd=truncated_svd,
    )


def original_infinity_args(args: argparse.Namespace, weights_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=str(args.cfg),
        tau=args.tau,
        pn=args.pn,
        model_path=str(weights_dir / "infinity_2b_reg.pth"),
        cfg_insertion_layer=0,
        vae_type=args.vae_type,
        vae_path=str(weights_dir / "infinity_vae_d32reg.pth"),
        add_lvl_embeding_only_first_block=0,
        use_bit_label=1,
        model_type=args.model_type,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=str(weights_dir / "flan-t5-xl"),
        text_channels=args.text_channels,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        enable_positive_prompt=0,
        cache_dir="/dev/shm",
        enable_model_cache=0,
        checkpoint_type="torch",
        seed=args.seed,
        bf16=1,
    )


def load_original_models(args: argparse.Namespace, weights_dir: Path, api: SimpleNamespace):
    run_args = original_infinity_args(args, weights_dir)
    print("[1/3] Loading FLAN-T5:", run_args.text_encoder_ckpt)
    text_tokenizer, text_encoder = api.load_tokenizer(t5_path=run_args.text_encoder_ckpt)
    print("[2/3] Loading Infinity VAE:", run_args.vae_path)
    vae = api.load_visual_tokenizer(run_args).eval()
    if not hasattr(vae.quantizer, "lfq"):
        vae.quantizer.lfq = vae.quantizer.bsq
    print("[3/3] Loading Infinity-2B transformer:", run_args.model_path)
    infinity = api.load_transformer(vae, run_args).eval()
    infinity.requires_grad_(False)
    return text_tokenizer, text_encoder, vae, infinity


def build_scale_schedule(args: argparse.Namespace, api: SimpleNamespace):
    h_div_w_template = api.h_div_w_templates[np.argmin(np.abs(api.h_div_w_templates - 1.0))]
    raw = api.dynamic_resolution_h_w[h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for _, h, w in raw]
    image_size = 1024 if args.pn == "1M" else 512 if args.pn == "0.25M" else 256
    print("Scale schedule:", scale_schedule)
    return scale_schedule, image_size


def encode_prompts(text_tokenizer, text_encoder, prompts, device: str):
    if isinstance(prompts, str):
        prompts = [prompts]
    tokens = text_tokenizer(text=list(prompts), max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tokens.input_ids.to(device, non_blocking=True)
    mask = tokens.attention_mask.to(device, non_blocking=True)
    with torch.no_grad():
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"].float()
    lens = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum(0), (1, 0))
    max_seqlen_k = max(lens)
    kv_compact = torch.cat([feat_i[:len_i] for len_i, feat_i in zip(lens, text_features.unbind(0))], dim=0)
    return kv_compact, lens, cu_seqlens_k, max_seqlen_k


def bit_labels_to_codes(vae, idx_bld, pn):
    idx = idx_bld.reshape(idx_bld.shape[0], pn[1], pn[2], -1).unsqueeze(1)
    return vae.quantizer.lfq.indices_to_codes(idx, label_type="bit_label")


def next_raw_from_summed_codes(vae, infinity, summed_codes, next_scale):
    last_stage = F.interpolate(summed_codes, size=next_scale, mode=vae.quantizer.z_interplote_up)
    last_stage = last_stage.squeeze(-3)
    if infinity.apply_spatial_patchify:
        last_stage = F.pixel_unshuffle(last_stage, 2)
    return last_stage.reshape(*last_stage.shape[:2], -1).permute(0, 2, 1)


def self_attention_module(block):
    return block.sa if hasattr(block, "sa") else block.attn


def tensor_to_pil(image) -> Image.Image:
    if isinstance(image, (list, tuple)):
        image = image[0]
    tensor = image.detach().float().cpu() if torch.is_tensor(image) else torch.as_tensor(image).float()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor.permute(1, 2, 0)
    if tensor.shape[-1] > 3:
        tensor = tensor[..., :3]
    lo, hi = float(tensor.min()), float(tensor.max())
    if lo < -0.05:
        tensor = (tensor + 1.0) / 2.0
    elif hi > 1.05:
        tensor = tensor / 255.0
    array = (tensor.clamp(0, 1).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


@torch.no_grad()
def decode_summed_codes_to_image_01(vae, summed_codes, device: str):
    codes = summed_codes.to(device)
    if codes.ndim == 5 and codes.shape[2] == 1:
        codes = codes.squeeze(-3)
    elif codes.ndim == 5 and codes.shape[1] == 1:
        codes = codes.squeeze(1)
    image = vae.decode(codes)
    return image.add(1).mul(0.5).clamp(0, 1)


@torch.no_grad()
def style_features_from_image(vae, image_path: Path, image_size: int, scale_schedule, device: str):
    image = ImageOps.fit(Image.open(image_path).convert("RGB"), (image_size, image_size), Image.Resampling.LANCZOS)
    image_m11 = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).to(device).mul(2).sub(1)
    with torch.amp.autocast("cuda", enabled=False):
        _, _, _, all_bit_indices, _, _ = vae.encode(image_m11.float(), scale_schedule=scale_schedule)
    final_size = scale_schedule[-1]
    summed = None
    features = []
    for step_id, bit_indices in enumerate(all_bit_indices):
        codes = vae.quantizer.lfq.indices_to_codes(bit_indices, label_type="bit_label")
        if step_id != len(scale_schedule) - 1:
            codes = F.interpolate(codes, size=final_size, mode=vae.quantizer.z_interplote_up)
        summed = codes if summed is None else summed + codes
        features.append(summed.detach().float().cpu())
    return features


def _feature_matrix(feature: torch.Tensor) -> torch.Tensor:
    return feature.detach().float().reshape(feature.shape[0], feature.shape[1], -1)


def estimate_content_basis(
    content_feature: torch.Tensor,
    *,
    content_rank: int | None = 1,
    variance_threshold: float | None = None,
):
    """Estimate per-sample channel bases from centered content activations.

    This mirrors ``feat/content_basis_rank``.  A fixed ``content_rank`` is the
    simple baseline; ``variance_threshold`` chooses the smallest content rank
    that explains enough centered content energy, optionally capped by
    ``content_rank``.
    """
    if content_rank is not None and content_rank < 1:
        raise ValueError("content_rank must be positive or None")
    if variance_threshold is not None and not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")
    if content_rank is None and variance_threshold is None:
        raise ValueError("Specify content_rank, variance_threshold, or both")

    bases, diagnostics = [], []
    for content_matrix in _feature_matrix(content_feature):
        centered = content_matrix - content_matrix.mean(dim=1, keepdim=True)
        u, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
        energy = singular_values.square()
        total_energy = energy.sum()
        if centered.shape[1] <= 1 or float(total_energy) <= torch.finfo(centered.dtype).eps:
            used_rank = 0
        elif variance_threshold is None:
            used_rank = min(int(content_rank), singular_values.numel())
        else:
            cumulative = energy.cumsum(0) / total_energy
            threshold_rank = int(torch.searchsorted(cumulative, variance_threshold).item()) + 1
            used_rank = threshold_rank if content_rank is None else min(threshold_rank, int(content_rank))
        basis = u[:, :used_rank]
        explained = 0.0 if used_rank == 0 else float(energy[:used_rank].sum() / total_energy)
        gap = 0.0
        if 0 < used_rank < singular_values.numel():
            gap = float(singular_values[used_rank - 1] / singular_values[used_rank].clamp_min(1e-12))
        bases.append(basis)
        diagnostics.append({"rank": used_rank, "explained_variance": explained, "spectral_gap": gap})
    return bases, diagnostics


def projected_pfb_content_blend(
    generation_feature: torch.Tensor,
    style_feature: torch.Tensor,
    content_feature: torch.Tensor,
    *,
    style_rank: int | None = 1,
    content_rank: int | None = 1,
    content_variance_threshold: float | None = None,
    alpha: float = 1.0,
    strength: float = 1.0,
    projection_strength: float = 1.0,
    preserve_mean: bool = False,
    api,
):
    """Apply PFB after suppressing its component in a content subspace.

    Implements ``Fg + strength * (I - eta Pc)[Phi(Fs) - Phi(Fg)]``.  The
    content projector ``Pc`` is estimated from the clean content stream at the
    same autoregressive scale.
    """
    if generation_feature.shape != style_feature.shape or generation_feature.shape != content_feature.shape:
        raise ValueError(
            f"Feature shape mismatch: generation={generation_feature.shape}, "
            f"style={style_feature.shape}, content={content_feature.shape}"
        )
    if not 0.0 <= projection_strength <= 1.0:
        raise ValueError("projection_strength must be in [0, 1]")

    style_feature = style_feature.to(generation_feature)
    content_feature = content_feature.to(generation_feature)
    delta = api.truncated_svd(style_feature, rank=style_rank, alpha=alpha) - api.truncated_svd(
        generation_feature, rank=style_rank, alpha=alpha
    )
    if preserve_mean:
        raw_delta = _feature_matrix(style_feature - generation_feature)
        mean_delta = raw_delta.mean(dim=2, keepdim=True).expand_as(raw_delta)
        delta = delta + mean_delta.reshape_as(delta).to(delta)

    bases, diagnostics = estimate_content_basis(
        content_feature,
        content_rank=content_rank,
        variance_threshold=content_variance_threshold,
    )
    projected = []
    for delta_matrix, basis, diagnostic in zip(_feature_matrix(delta), bases, diagnostics):
        content_component = basis @ (basis.transpose(0, 1) @ delta_matrix)
        edit = delta_matrix - float(projection_strength) * content_component
        input_norm = delta_matrix.norm().clamp_min(1e-12)
        diagnostic["removed_fraction"] = float(content_component.norm() / input_norm)
        diagnostic["orthogonality_residual"] = (
            0.0 if basis.shape[1] == 0 else float((basis.transpose(0, 1) @ edit).norm() / input_norm)
        )
        projected.append(edit.reshape(generation_feature.shape[1:]))
    projected_delta = torch.stack(projected).to(generation_feature)
    return generation_feature + float(strength) * projected_delta, diagnostics


def apply_injection(method: str, generation_feature, style_feature, content_feature, args, api):
    style_feature = style_feature.to(generation_feature)
    if method == "raw_residual":
        return generation_feature + float(args.strength) * (style_feature - generation_feature), []
    if method == "svd_residual":
        style_component = api.truncated_svd(style_feature, rank=args.style_rank, alpha=args.alpha)
        generation_component = api.truncated_svd(generation_feature, rank=args.style_rank, alpha=args.alpha)
        return generation_feature + float(args.strength) * (style_component - generation_component), []
    if method == "content_projection":
        content_rank = None if args.content_rank == 0 else args.content_rank
        return projected_pfb_content_blend(
            generation_feature,
            style_feature,
            content_feature,
            style_rank=args.style_rank,
            content_rank=content_rank,
            content_variance_threshold=args.content_variance_threshold,
            alpha=args.alpha,
            strength=args.strength,
            projection_strength=args.projection_strength,
            preserve_mean=args.preserve_mean,
            api=api,
        )
    raise ValueError(f"Unknown method: {method}")


def sample_step_indices(logits, api, args, rng):
    logits = logits.reshape(logits.shape[0], -1, 2).clone()
    sampled = api.sample_with_top_k_top_p(
        logits,
        rng=rng,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples=1,
    )[:, :, 0]
    return sampled


@torch.no_grad()
def generate_with_single_step_injection(
    *,
    prompt: str,
    seed: int,
    inject_step: int,
    method: str,
    style_features,
    args: argparse.Namespace,
    api: SimpleNamespace,
    text_tokenizer,
    text_encoder,
    vae,
    infinity,
    scale_schedule,
) -> Image.Image:
    device = args.device
    content_rng = torch.Generator(device=device).manual_seed(int(seed))
    generation_rng = torch.Generator(device=device).manual_seed(int(seed))

    prompts = [prompt, prompt]
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = encode_prompts(text_tokenizer, text_encoder, prompts, device)
    kv_compact_un = kv_compact.clone()
    offset = 0
    for prompt_len in lens:
        kv_compact_un[offset : offset + prompt_len] = infinity.cfg_uncond[:prompt_len]
        offset += prompt_len
    kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
    cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
    batch_size = 4

    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_bd = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = infinity.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(batch_size, 1, -1) + infinity.pos_start.expand(batch_size, 1, -1)

    with torch.amp.autocast("cuda", enabled=False):
        cond_bd_or_gss = infinity.shared_ada_lin(cond_bd.float()).float().contiguous()

    final_size = scale_schedule[-1]
    content_summed = last_stage.new_zeros(1, infinity.d_vae, *final_size)
    generation_summed = last_stage.new_zeros(1, infinity.d_vae, *final_size)
    injection_diagnostics = []

    for block in infinity.unregistered_blocks:
        self_attention_module(block).kv_caching(True)

    try:
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            for step_id, pn in enumerate(scale_schedule):
                cur_len = sum(np.array(x).prod() for x in scale_schedule[: step_id + 1])
                prev_len = sum(np.array(x).prod() for x in scale_schedule[:step_id])
                attn_fn = None
                if infinity.use_flex_attn:
                    attn_fn = infinity.attn_fn_compile_dict.get(tuple(scale_schedule[: step_id + 1]), None)

                for block_idx, block_chunk in enumerate(infinity.block_chunks):
                    if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
                        last_stage = infinity.add_lvl_embeding(last_stage, step_id, scale_schedule, need_to_pad=0)
                    if not infinity.add_lvl_embeding_only_first_block:
                        last_stage = infinity.add_lvl_embeding(last_stage, step_id, scale_schedule, need_to_pad=0)

                    for block in block_chunk.module:
                        last_stage = block(
                            x=last_stage,
                            cond_BD=cond_bd_or_gss,
                            ca_kv=ca_kv,
                            attn_bias_or_two_vector=None,
                            attn_fn=attn_fn,
                            scale_schedule=scale_schedule,
                            rope2d_freqs_grid=infinity.rope2d_freqs_grid,
                            scale_ind=step_id,
                        )

                logits = infinity.get_logits(last_stage, cond_bd).mul(1 / float(args.tau))
                logits = float(args.cfg) * logits[:2] + (1 - float(args.cfg)) * logits[2:]
                sampled_content = sample_step_indices(logits[:1], api, args, content_rng)
                sampled_generation = sample_step_indices(logits[1:2], api, args, generation_rng)
                step_token_count = cur_len - prev_len
                content_idx = sampled_content.reshape(1, step_token_count, -1)
                generation_idx = sampled_generation.reshape(1, step_token_count, -1)
                content_codes = bit_labels_to_codes(vae, content_idx, pn)
                generation_codes = bit_labels_to_codes(vae, generation_idx, pn)
                if step_id != len(scale_schedule) - 1:
                    content_codes = F.interpolate(content_codes, size=final_size, mode=vae.quantizer.z_interplote_up)
                    generation_codes = F.interpolate(generation_codes, size=final_size, mode=vae.quantizer.z_interplote_up)

                content_summed = content_summed + content_codes
                generation_summed = generation_summed + generation_codes
                if step_id == inject_step:
                    generation_summed, injection_diagnostics = apply_injection(
                        method,
                        generation_summed,
                        style_features[step_id].to(device),
                        content_summed,
                        args,
                        api,
                    )

                if step_id != len(scale_schedule) - 1:
                    content_next = next_raw_from_summed_codes(vae, infinity, content_summed, scale_schedule[step_id + 1])
                    generation_next = next_raw_from_summed_codes(vae, infinity, generation_summed, scale_schedule[step_id + 1])
                    last_stage = infinity.word_embed(infinity.norm0_ve(torch.cat([content_next, generation_next], dim=0)))
                    last_stage = last_stage.repeat(2, 1, 1)

        image_01 = decode_summed_codes_to_image_01(vae, generation_summed, device)
        return tensor_to_pil(image_01), injection_diagnostics
    finally:
        for block in infinity.unregistered_blocks:
            self_attention_module(block).kv_caching(False)


class Metrics:
    def __init__(self, device: str, clip_model_id: str, csd_model_id: str = ""):
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

        self.device = device
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
        self.clip_model.requires_grad_(False)
        self.AutoImageProcessor = AutoImageProcessor
        self.AutoProcessor = AutoProcessor
        self.AutoModel = AutoModel
        candidates = [csd_model_id.strip(), "bigshanedogg/CSD", "tomg-group-umd/CSD-ViT-L"]
        self.csd_candidates = [name for i, name in enumerate(candidates) if name and name not in candidates[:i]]
        self.csd_processor = None
        self.csd_model = None
        self.text_cache = {}
        self.style_cache = {}

    @staticmethod
    def cosine(a, b) -> float:
        return float((a.float() * b.float()).sum(dim=-1).clamp(-1, 1).detach().cpu().item())

    @staticmethod
    def image_for_metrics(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    @torch.no_grad()
    def clip_text_embedding(self, text: str):
        if text in self.text_cache:
            return self.text_cache[text]
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        embedding = self.clip_model.get_text_features(**inputs).float()
        embedding = F.normalize(embedding, dim=-1).detach().cpu()
        self.text_cache[text] = embedding
        return embedding

    @torch.no_grad()
    def clip_image_embedding(self, image):
        inputs = self.clip_processor(images=self.image_for_metrics(image), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        embedding = self.clip_model.get_image_features(**inputs).float()
        return F.normalize(embedding, dim=-1).detach().cpu()

    def get_csd_model(self):
        if self.csd_processor is not None and self.csd_model is not None:
            return self.csd_processor, self.csd_model
        last_error = None
        for model_id in self.csd_candidates:
            try:
                try:
                    processor = self.AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
                except Exception:
                    processor = self.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                model = self.AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
                model.requires_grad_(False)
                self.csd_processor = processor
                self.csd_model = model
                print("Loaded CSD-S style encoder:", model_id)
                return processor, model
            except Exception as exc:
                last_error = exc
                print(f"CSD-S load failed for {model_id}: {repr(exc)}")
        raise RuntimeError("Could not load a real CSD-S style encoder.") from last_error

    @staticmethod
    def extract_named_tensor(output, names):
        for name in names:
            value = getattr(output, name, None)
            if torch.is_tensor(value):
                return value
        if isinstance(output, dict):
            for name in names:
                value = output.get(name)
                if torch.is_tensor(value):
                    return value
        return None

    @classmethod
    def fallback_output_tensor(cls, output):
        if torch.is_tensor(output):
            return output
        value = cls.extract_named_tensor(output, ("pooler_output", "last_hidden_state"))
        if torch.is_tensor(value):
            return value[:, 0] if value.ndim >= 3 else value
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item[:, 0] if item.ndim >= 3 else item
        raise TypeError(f"Could not extract a CSD tensor from output type {type(output)}")

    @torch.no_grad()
    def csd_style_embedding(self, image, cache_key=None):
        key = str(cache_key) if cache_key is not None else None
        if key is not None and key in self.style_cache:
            return self.style_cache[key]
        processor, model = self.get_csd_model()
        inputs = processor(images=self.image_for_metrics(image), return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        outputs = model(**inputs)
        style = self.extract_named_tensor(
            outputs,
            (
                "embeddings",
                "style_embeddings",
                "style_embeds",
                "style_embedding",
                "image_style_embeds",
                "image_embeds",
            ),
        )
        if style is None:
            style = self.fallback_output_tensor(outputs)
        style = F.normalize(style.float(), dim=-1).detach().cpu()
        if key is not None:
            self.style_cache[key] = style
        return style


def build_sample_manifest(args: argparse.Namespace) -> pd.DataFrame:
    prompts = read_csv_rows(resolve_path(args.workspace_dir, args.prompts_csv))
    styles = read_csv_rows(resolve_path(args.workspace_dir, args.styles_csv))
    if len(prompts) < args.num_samples:
        raise ValueError(f"Need at least {args.num_samples} prompts, found {len(prompts)}.")
    if not styles:
        raise ValueError("No styles found.")

    rng = random.Random(args.seed)
    sampled_prompts = rng.sample(prompts, args.num_samples)
    rows = []
    for sample_index, prompt_row in enumerate(sampled_prompts):
        style_row = styles[sample_index % len(styles)]
        style_path = resolve_path(args.workspace_dir, Path(style_row["style_reference_image"])).resolve()
        if not style_path.is_file():
            raise FileNotFoundError(style_path)
        content_prompt = prompt_row["content_prompt"]
        superclass = prompt_row.get("superclass", "").strip()
        if superclass:
            content_prompt = f"{content_prompt}, {superclass}"
        rows.append(
            {
                "sample_id": f"sample_{sample_index:03d}",
                "content_id": prompt_row.get("content_id", f"content_{sample_index:03d}"),
                "content_prompt": content_prompt,
                "style_id": style_row.get("style_id", f"style_{sample_index % len(styles):02d}"),
                "style_name": style_row.get("style_name", style_row.get("style_descriptor", "")),
                "style_reference_image": str(style_path),
            }
        )
    return pd.DataFrame(rows)


def image_output_path(dirs: OutputDirs, method: str, sample_id: str, step: int) -> Path:
    return dirs.generated / method / sample_id / f"inject_step_{step:02d}.png"


def make_step_grid(style_path: Path, step_images: list[Image.Image], content_prompt: str, title: str, output_path: Path):
    thumb = 128
    label_h = 32
    bottom_h = 60
    labels = ["style ref"] + ["ŝ = 1"] + [str(i) for i in range(2, len(step_images) + 1)]
    style_image = ImageOps.fit(Image.open(style_path).convert("RGB"), (thumb, thumb), Image.Resampling.LANCZOS)
    images = [style_image] + [ImageOps.fit(image.convert("RGB"), (thumb, thumb), Image.Resampling.LANCZOS) for image in step_images]
    cols = len(images)
    canvas = Image.new("RGB", (cols * thumb, label_h + thumb + bottom_h + 8), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    for col, (label, image) in enumerate(zip(labels, images)):
        x = col * thumb
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (thumb - (bbox[2] - bbox[0])) / 2, 7), label, fill=(0, 0, 0), font=font)
        canvas.paste(image, (x, label_h))

    caption = f'{title}, content prompt = "{content_prompt}"'
    max_width = canvas.width - 20
    shown = caption
    while len(shown) > 8 and draw.textbbox((0, 0), shown, font=small)[2] > max_width:
        shown = shown[:-2]
    if shown != caption:
        shown = shown.rstrip() + "..."
    bbox = draw.textbbox((0, 0), shown, font=small)
    draw.text(((canvas.width - (bbox[2] - bbox[0])) / 2, label_h + thumb + 18), shown, fill=(0, 0, 0), font=small)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def harmonic_mean(a: float, b: float, eps: float = 1e-8) -> float:
    return float(2.0 * a * b / (a + b + eps))


def run_experiment(args, dirs, api, text_tokenizer, text_encoder, vae, infinity, scale_schedule, image_size):
    manifest = build_sample_manifest(args)
    manifest_path = dirs.metrics / "sample_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    generated_rows = []
    total = len(manifest) * len(METHODS) * len(scale_schedule)
    progress = tqdm(total=total, desc="Step-wise feature injection ablation")
    for row in manifest.to_dict("records"):
        sample_id = row["sample_id"]
        prompt = row["content_prompt"]
        style_path = Path(row["style_reference_image"])
        style_features = style_features_from_image(vae, style_path, image_size, scale_schedule, args.device)

        for method, method_label in METHODS.items():
            step_images = []
            for step_index in range(len(scale_schedule)):
                output_path = image_output_path(dirs, method, sample_id, step_index + 1)
                diagnostics = []
                if output_path.exists() and not args.force:
                    generated = Image.open(output_path).convert("RGB")
                else:
                    generated, diagnostics = generate_with_single_step_injection(
                        prompt=prompt,
                        seed=args.seed + int(sample_id.split("_")[-1]),
                        inject_step=step_index,
                        method=method,
                        style_features=style_features,
                        args=args,
                        api=api,
                        text_tokenizer=text_tokenizer,
                        text_encoder=text_encoder,
                        vae=vae,
                        infinity=infinity,
                        scale_schedule=scale_schedule,
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    generated.save(output_path)

                step_images.append(generated)
                generated_rows.append(
                    {
                        "sample_id": sample_id,
                        "content_id": row["content_id"],
                        "style_id": row["style_id"],
                        "style_name": row["style_name"],
                        "method": method,
                        "method_label": method_label,
                        "inject_step": step_index + 1,
                        "content_prompt": prompt,
                        "style_reference_image": str(style_path),
                        "image_path": str(output_path),
                        "projection_diagnostics": json.dumps(diagnostics),
                    }
                )
                progress.update(1)
                gc.collect()
                torch.cuda.empty_cache()

            grid_path = dirs.grids / method / f"{sample_id}_{safe_slug(row['style_id'])}.png"
            make_step_grid(style_path, step_images, prompt, method_label, grid_path)

        del style_features
        gc.collect()
        torch.cuda.empty_cache()
    progress.close()

    generated_manifest_path = dirs.metrics / "generated_image_manifest.csv"
    pd.DataFrame(generated_rows).to_csv(generated_manifest_path, index=False)

    print("Generation finished. Loading CLIP and CSD-S backbones for scoring saved images.")
    metrics = Metrics(args.device, clip_model_id=args.clip_model_id, csd_model_id=args.csd_model_id)
    metric_rows = []
    for row in tqdm(generated_rows, desc="Scoring generated images"):
        generated = Image.open(row["image_path"]).convert("RGB")
        prompt_embedding = metrics.clip_text_embedding(row["content_prompt"])
        style_embedding = metrics.csd_style_embedding(row["style_reference_image"], cache_key=f"style::{row['style_id']}")
        content_similarity = metrics.cosine(metrics.clip_image_embedding(generated), prompt_embedding)
        style_similarity = metrics.cosine(
            metrics.csd_style_embedding(generated, cache_key=f"{row['method']}::{row['sample_id']}::{row['inject_step']}"),
            style_embedding,
        )
        metric_rows.append(
            {
                **row,
                "content_similarity": content_similarity,
                "style_similarity": style_similarity,
                "s_harmonic": harmonic_mean(content_similarity, style_similarity),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = dirs.metrics / "stepwise_feature_injection_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary = metrics_df.groupby(["method", "method_label", "inject_step"], as_index=False).agg(
        content_similarity_mean=("content_similarity", "mean"),
        content_similarity_std=("content_similarity", "std"),
        style_similarity_mean=("style_similarity", "mean"),
        style_similarity_std=("style_similarity", "std"),
        s_harmonic_mean=("s_harmonic", "mean"),
        s_harmonic_std=("s_harmonic", "std"),
    )
    summary_path = dirs.metrics / "stepwise_feature_injection_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_method_charts(summary, dirs)
    return metrics_path, summary_path, manifest_path


def plot_method_charts(summary: pd.DataFrame, dirs: OutputDirs):
    import matplotlib.pyplot as plt

    for method, method_label in METHODS.items():
        df = summary[summary["method"] == method].sort_values("inject_step")
        x = df["inject_step"].to_numpy()
        content = df["content_similarity_mean"].to_numpy()
        style = df["style_similarity_mean"].to_numpy()
        harmonic = df["s_harmonic_mean"].to_numpy()

        fig, ax_style = plt.subplots(figsize=(13, 2.3))
        ax_content = ax_style.twinx()
        width = 0.36
        ax_style.bar(x - width / 2, style, width=width, color="#ffd98e", alpha=0.75, label="Style Similarity")
        ax_content.bar(x + width / 2, content, width=width, color="#ffadc0", alpha=0.78, label="Content Similarity")
        ax_style.plot(x, harmonic, color="#333333", linewidth=1.8, marker="o", markersize=4, label="S_harmonic")

        ax_style.set_xlabel("Generation step (s)", fontweight="bold")
        ax_style.set_ylabel("Style Similarity", color="#ff9900", fontweight="bold")
        ax_content.set_ylabel("Content Similarity", color="#ff9fb6", fontweight="bold")
        ax_style.set_xticks(x)
        ax_style.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax_style.set_title(f'{method_label}: inject style reference at $\\hat{{s}}$-th scale, content prompt fixed', fontsize=11)
        lines1, labels1 = ax_style.get_legend_handles_labels()
        lines2, labels2 = ax_content.get_legend_handles_labels()
        ax_content.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(dirs.plots / f"{method}_stepwise_content_style_harmonic.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def package_outputs(dirs: OutputDirs):
    package_path = dirs.root / "feature_injection_ablation_outputs.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in dirs.root.rglob("*"):
            if path.is_file() and path != package_path:
                archive.write(path, path.relative_to(dirs.root))
    print("Saved package:", package_path)


def main():
    args = parse_args()
    args.workspace_dir = args.workspace_dir.resolve()
    args.prompts_csv = resolve_path(args.workspace_dir, args.prompts_csv)
    args.styles_csv = resolve_path(args.workspace_dir, args.styles_csv)
    args.output_dir = resolve_path(args.workspace_dir, args.output_dir).resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for original Infinity-2B inference.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dirs = build_output_dirs(args.output_dir)
    infinity_dir = resolve_infinity_dir(args.workspace_dir, args.infinity_dir)
    weights_dir = resolve_weights_dir(args.workspace_dir, args.weights_dir)
    api = import_runtime(infinity_dir)

    print("Workspace:", args.workspace_dir)
    print("Infinity source:", infinity_dir)
    print("Weights:", weights_dir)
    print("Outputs:", dirs.root)
    print("Methods:", ", ".join(METHODS.values()))

    text_tokenizer, text_encoder, vae, infinity = load_original_models(args, weights_dir, api)
    scale_schedule, image_size = build_scale_schedule(args, api)
    with (dirs.root / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, stream, indent=2)

    metrics_path, summary_path, manifest_path = run_experiment(
        args, dirs, api, text_tokenizer, text_encoder, vae, infinity, scale_schedule, image_size
    )
    print("Saved manifest:", manifest_path)
    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved plots:", dirs.plots)
    print("Saved image grids:", dirs.grids)
    if args.package_zip:
        package_outputs(dirs)


if __name__ == "__main__":
    main()
