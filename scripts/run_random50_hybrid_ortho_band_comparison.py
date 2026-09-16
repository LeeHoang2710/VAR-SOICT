#!/usr/bin/env python3
"""Run a random-50 comparison for hybrid normal/content-projection schedules.

This tests three scale-aware hypotheses:

  A. scales 0,1,2 use normal top-1; scales 3,4,5 use normal SVD-band PFB.
     This is the Variant 2 spectrum schedule.
  B. scales 0,1,2 use content-orthogonal top-1; scales 3,4,5 use normal
     SVD-band PFB.
  C. scales 0,1,2 use normal top-1; scales 3,4,5 use content-orthogonal
     SVD-band PFB.
  D. scales 1,2 use normal top-1; scales 3,4,5 use content-projected PFB
     from feat/content_basis_rank.

For both variants, early scales receive only top-1. Later scales receive the
second singular band, with a small amount of top-1 retained for global
background/style consistency:

    late_delta = late_top1_weight * top1_delta + (top2_delta - top1_delta)

By default, this reuses the sample_cases.csv from
random50_three_variant_comparison_seed2026 so the result can be compared
directly with the previous three-variant run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT / "src", PROJECT_ROOT / "Infinity", PROJECT_ROOT / "Infinity/tools", PROJECT_ROOT / "scripts"):
    sys.path.insert(0, str(directory))

import run_content_ortho_batch as runtime


EARLY_STEPS = [0, 1, 2]
LATE_STEPS = [3, 4, 5]
INJECT_STEPS = EARLY_STEPS + LATE_STEPS
EARLY_RANK = 1
LATE_RANK = 2
LATE_TOP1_WEIGHT = 0.25
STYLE_DECAY = 0.75
STYLE_STRENGTH = 1.0
CONTENT_RANK = 1
PROJECTION_STRENGTH = 1.0
PROJECTION_STYLE_RANK = 1
SAC_START_STEP = 2

VARIANTS = [
    {
        "name": "normal012_normal345_band",
        "label": "Variant 2 spectrum: Normal 0-2 top1 + 3-5 top2 band",
        "directory": "normal012_normal345_band",
        "early_mode": "normal",
        "late_mode": "normal",
    },
    {
        "name": "ortho012_normal345_band",
        "label": "Ortho 0-2 top1 + Normal 3-5 band",
        "directory": "ortho012_normal345_band",
        "early_mode": "ortho",
        "late_mode": "normal",
    },
    {
        "name": "normal012_ortho345_band",
        "label": "Normal 0-2 top1 + Ortho 3-5 band",
        "directory": "normal012_ortho345_band",
        "early_mode": "normal",
        "late_mode": "ortho",
    },
    {
        "name": "normal12_projection345",
        "label": "Normal 1-2 top1 + Content Projection 3-5",
        "directory": "normal12_projection345",
        "early_mode": "normal",
        "late_mode": "projection",
        "early_steps": [1, 2],
        "late_steps": [3, 4, 5],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-csv", type=Path, default=PROJECT_ROOT / "prompts/content_prompts_190.csv")
    parser.add_argument("--styles-csv", type=Path, default=PROJECT_ROOT / "styles/quantitative_eval_styles_10.csv")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/random50_hybrid_ortho_band_comparison_seed2026")
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/random50_three_variant_comparison_seed2026/sample_cases.csv",
        help="Sample case CSV to reuse. Defaults to the previous three-variant comparison sample.",
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "0.60M", "1M"), default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900, help="Sampling top-k, not SVD rank.")
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=STYLE_DECAY)
    parser.add_argument("--late-top1-weight", type=float, default=LATE_TOP1_WEIGHT)
    parser.add_argument("--projection-strength", type=float, default=PROJECTION_STRENGTH)
    parser.add_argument("--content-rank", type=int, default=CONTENT_RANK)
    parser.add_argument(
        "--content-variance-threshold",
        type=float,
        default=None,
        help="Optional adaptive content-basis energy threshold in (0,1], e.g. 0.9.",
    )
    parser.add_argument("--projection-style-rank", type=int, default=PROJECTION_STYLE_RANK)
    parser.add_argument(
        "--preserve-mean",
        action="store_true",
        help="Add raw mean/palette delta before content projection. Off by default.",
    )
    parser.add_argument(
        "--variant",
        choices=[variant["name"] for variant in VARIANTS],
        default=None,
        help="Run only one variant. Defaults to all variants.",
    )
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None, help="CLIP device; defaults to CUDA when available.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resample", action="store_true", help="Draw a fresh sample instead of reusing --sample-csv.")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--inject-step", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    if args.generate_only and args.metrics_only:
        parser.error("--generate-only and --metrics-only are mutually exclusive")
    if args.content_rank < 0:
        parser.error("--content-rank must be non-negative")
    if args.content_rank == 0 and args.content_variance_threshold is None:
        parser.error("--content-rank 0 requires --content-variance-threshold")
    if args.content_variance_threshold is not None and not 0 < args.content_variance_threshold <= 1:
        parser.error("--content-variance-threshold must be in (0, 1]")
    if args.projection_style_rank < 1:
        parser.error("--projection-style-rank must be positive")
    if args.projection_strength < 0 or args.projection_strength > 1:
        parser.error("--projection-strength must be in [0, 1]")
    if args.late_top1_weight < 0:
        parser.error("--late-top1-weight must be non-negative")
    return args


def selected_variants(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.variant is None:
        return VARIANTS
    return [variant for variant in VARIANTS if variant["name"] == args.variant]


def early_steps_for_variant(variant: dict[str, str]) -> list[int]:
    return list(variant.get("early_steps", EARLY_STEPS))


def late_steps_for_variant(variant: dict[str, str]) -> list[int]:
    return list(variant.get("late_steps", LATE_STEPS))


def inject_steps_for_variant(variant: dict[str, str]) -> list[int]:
    return early_steps_for_variant(variant) + late_steps_for_variant(variant)


def evaluation_prompt(content: dict[str, str], style: dict[str, str]) -> str:
    descriptor = style["style_descriptor"].strip()
    suffix = descriptor if descriptor.lower().endswith("style") else f"{descriptor} style"
    return f"{content['content_prompt']}, {content['superclass']}, in {suffix}"


def tensor_to_pil(image):
    array = image.detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return runtime.Image.fromarray(array, mode="RGB")


def read_sample(path: Path, contents_by_id: dict[str, dict[str, str]], styles_by_id: dict[str, dict[str, str]]):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append((contents_by_id[row["content_id"]], styles_by_id[row["style_id"]]))
    return rows


def write_sample(path: Path, sample_cases: list[tuple[dict[str, str], dict[str, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case_index",
                "content_id",
                "content_prompt",
                "superclass",
                "style_id",
                "style_name",
                "style_descriptor",
                "style_reference_image",
                "evaluation_prompt",
            ],
        )
        writer.writeheader()
        for index, (content, style) in enumerate(sample_cases):
            writer.writerow({
                "case_index": index,
                "content_id": content["content_id"],
                "content_prompt": content["content_prompt"],
                "superclass": content["superclass"],
                "style_id": style["style_id"],
                "style_name": style["style_name"],
                "style_descriptor": style["style_descriptor"],
                "style_reference_image": style["style_reference_image"],
                "evaluation_prompt": evaluation_prompt(content, style),
            })


def select_sample_cases(args: argparse.Namespace, contents: list[dict[str, str]], styles: list[dict[str, str]]):
    output_sample_path = args.output_root.resolve() / "sample_cases.csv"
    contents_by_id = {row["content_id"]: row for row in contents}
    styles_by_id = {row["style_id"]: row for row in styles}
    sample_csv = args.sample_csv.resolve() if args.sample_csv is not None else output_sample_path
    if sample_csv.exists() and not args.resample:
        sample_cases = read_sample(sample_csv, contents_by_id, styles_by_id)
        output_sample_path.parent.mkdir(parents=True, exist_ok=True)
        if sample_csv != output_sample_path:
            output_sample_path.write_text(sample_csv.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Loaded existing sample: {sample_csv} ({len(sample_cases)} cases)")
        print(f"Copied comparison sample to: {output_sample_path}")
        return sample_cases

    population = [(content, style) for content in contents for style in styles]
    if args.sample_size > len(population):
        raise ValueError(f"--sample-size={args.sample_size} exceeds {len(population)} available cases")
    rng = random.Random(args.sample_seed)
    sample_cases = rng.sample(population, args.sample_size)
    write_sample(output_sample_path, sample_cases)
    print(f"Saved sampled cases: {output_sample_path}")
    return sample_cases


def save_comparison(content_image, style_path: str, generated_image, prompt: str, variant_label: str, path: Path) -> None:
    size = generated_image.size
    style_image = runtime.ImageOps.fit(runtime.Image.open(style_path).convert("RGB"), size, runtime.Image.Resampling.LANCZOS)
    content_image = runtime.ImageOps.fit(content_image.convert("RGB"), size, runtime.Image.Resampling.LANCZOS)
    header = 80
    canvas = runtime.Image.new("RGB", (size[0] * 3, size[1] + header), "white")
    draw = runtime.ImageDraw.Draw(canvas)
    labels = (f"Prompt: {prompt}", "Style reference", variant_label)
    for column, (label, image) in enumerate(zip(labels, (content_image, style_image, generated_image.convert("RGB")))):
        x = column * size[0]
        canvas.paste(image, (x, header))
        draw.text((x + 12, 24), label[:72], fill="black")
    runtime.save_image_atomic(canvas, path)


def normal_top1_delta(generation_feature, style_feature, *, alpha: float):
    from var_soict.feature_hypotheses import truncated_svd

    style_feature = style_feature.to(generation_feature)
    return truncated_svd(style_feature, rank=EARLY_RANK, alpha=alpha) - truncated_svd(
        generation_feature, rank=EARLY_RANK, alpha=alpha
    )


def normal_late_band_delta(generation_feature, style_feature, *, alpha: float, top1_weight: float):
    from var_soict.feature_hypotheses import truncated_svd

    style_feature = style_feature.to(generation_feature)
    style_top1 = truncated_svd(style_feature, rank=1, alpha=alpha)
    generation_top1 = truncated_svd(generation_feature, rank=1, alpha=alpha)
    style_top2 = truncated_svd(style_feature, rank=LATE_RANK, alpha=alpha)
    generation_top2 = truncated_svd(generation_feature, rank=LATE_RANK, alpha=alpha)
    top1_delta = style_top1 - generation_top1
    band_2_delta = (style_top2 - style_top1) - (generation_top2 - generation_top1)
    return float(top1_weight) * top1_delta + band_2_delta


def ortho_top1_delta(style_feature, content_feature, *, alpha: float, content_rank: int, projection_strength: float):
    from var_soict.feature_hypotheses import remove_content_subspace, truncated_svd

    residual = remove_content_subspace(
        style_feature,
        content_feature,
        content_rank=content_rank,
        projection_strength=projection_strength,
        preserve_mean=True,
    )
    return truncated_svd(residual, rank=1, alpha=alpha)


def ortho_late_band_delta(
    style_feature,
    content_feature,
    *,
    alpha: float,
    content_rank: int,
    projection_strength: float,
    top1_weight: float,
):
    from var_soict.feature_hypotheses import remove_content_subspace, truncated_svd

    residual = remove_content_subspace(
        style_feature,
        content_feature,
        content_rank=content_rank,
        projection_strength=projection_strength,
        preserve_mean=True,
    )
    residual_top1 = truncated_svd(residual, rank=1, alpha=alpha)
    residual_top2 = truncated_svd(residual, rank=LATE_RANK, alpha=alpha)
    return float(top1_weight) * residual_top1 + (residual_top2 - residual_top1)


def projected_content_delta(
    generation_feature,
    style_feature,
    content_feature,
    *,
    alpha: float,
    style_rank: int,
    content_rank: int | None,
    content_variance_threshold: float | None,
    projection_strength: float,
    preserve_mean: bool,
):
    from var_soict.feature_hypotheses import projected_pfb_content_blend

    edited, diagnostics = projected_pfb_content_blend(
        generation_feature,
        style_feature,
        content_feature,
        style_rank=style_rank,
        content_rank=content_rank,
        content_variance_threshold=content_variance_threshold,
        alpha=alpha,
        strength=1.0,
        projection_strength=projection_strength,
        preserve_mean=preserve_mean,
        return_diagnostics=True,
    )
    return edited - generation_feature, diagnostics


def hybrid_delta_for_step(
    *,
    step_id: int,
    variant: dict[str, str],
    generation_feature,
    style_feature,
    content_feature,
    args: argparse.Namespace,
):
    early_steps = early_steps_for_variant(variant)
    late_steps = late_steps_for_variant(variant)
    if step_id in early_steps:
        mode = variant["early_mode"]
        if mode == "normal":
            return normal_top1_delta(generation_feature, style_feature, alpha=args.alpha), []
        if mode == "ortho":
            return ortho_top1_delta(
                style_feature,
                content_feature,
                alpha=args.alpha,
                content_rank=args.content_rank,
                projection_strength=args.projection_strength,
            ).to(generation_feature), []
    if step_id in late_steps:
        mode = variant["late_mode"]
        if mode == "normal":
            return normal_late_band_delta(
                generation_feature,
                style_feature,
                alpha=args.alpha,
                top1_weight=args.late_top1_weight,
            ), []
        if mode == "ortho":
            return ortho_late_band_delta(
                style_feature,
                content_feature,
                alpha=args.alpha,
                content_rank=args.content_rank,
                projection_strength=args.projection_strength,
                top1_weight=args.late_top1_weight,
            ).to(generation_feature), []
        if mode == "projection":
            content_rank = None if args.content_rank == 0 else args.content_rank
            return projected_content_delta(
                generation_feature,
                style_feature,
                content_feature,
                alpha=args.alpha,
                style_rank=args.projection_style_rank,
                content_rank=content_rank,
                content_variance_threshold=args.content_variance_threshold,
                projection_strength=args.projection_strength,
                preserve_mean=args.preserve_mean,
            )
    raise ValueError(f"Step {step_id} is not configured for hybrid injection.")


def hybrid_generate(engine, prompt: str, style_features: list, variant: dict[str, str], args: argparse.Namespace):
    import torch
    import torch.nn.functional as F
    from var_soict.style_transfer import PaperSACPatch, SACController

    if args.cfg < 1.0:
        raise ValueError("CFG must be >= 1.0 for this dual-stream experiment.")
    inject_steps = inject_steps_for_variant(variant)
    if max(inject_steps) >= len(engine.scale_schedule):
        raise ValueError(f"Need at least {max(inject_steps) + 1} scales, got {len(engine.scale_schedule)}")

    engine.model.eval()
    base_batch = 1
    condition_batch = 2
    content_rng = torch.Generator(device=engine.device).manual_seed(args.seed)
    generation_rng = torch.Generator(device=engine.device).manual_seed(args.seed)

    kv_compact, lens, cu_seqlens_k, max_seqlen_k = engine.encode_prompts([prompt, prompt])
    kv_compact_un = kv_compact.clone()
    total = 0
    for length in lens:
        kv_compact_un[total : total + length] = engine.model.cfg_uncond[:length]
        total += length
    kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
    cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
    bs = 4

    kv_compact = engine.model.text_norm(kv_compact)
    sos = cond_bd = engine.model.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = engine.model.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + engine.model.pos_start.expand(bs, 1, -1)

    with torch.amp.autocast("cuda", enabled=False):
        cond_bd_or_gss = engine.model.shared_ada_lin(cond_bd.float()).float().contiguous()

    final_size = engine.scale_schedule[-1]
    content_summed = last_stage.new_zeros(base_batch, engine.model.d_vae, *final_size)
    generation_summed = torch.zeros_like(content_summed)
    content_trace, generation_trace = [], []
    pfb_relative_change_by_step = {}
    projection_diagnostics_by_step = {}

    controller = SACController(base_batch)
    controller.sac_strength = 1.0
    controller.reset_statistics()

    for block in engine.model.unregistered_blocks:
        block.sa.kv_caching(True)

    try:
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            with PaperSACPatch(engine.model, controller):
                for step_id, pn in enumerate(engine.scale_schedule):
                    controller.active = step_id >= SAC_START_STEP
                    need_to_pad = 0
                    attn_fn = None
                    if engine.model.use_flex_attn:
                        attn_fn = engine.model.attn_fn_compile_dict.get(tuple(engine.scale_schedule[: step_id + 1]), None)

                    for block_idx, block_chunk in enumerate(engine.model.block_chunks):
                        if engine.model.add_lvl_embeding_only_first_block and block_idx == 0:
                            last_stage = engine.model.add_lvl_embeding(
                                last_stage, step_id, engine.scale_schedule, need_to_pad=need_to_pad
                            )
                        if not engine.model.add_lvl_embeding_only_first_block:
                            last_stage = engine.model.add_lvl_embeding(
                                last_stage, step_id, engine.scale_schedule, need_to_pad=need_to_pad
                            )

                        for block in block_chunk.module:
                            last_stage = block(
                                x=last_stage,
                                cond_BD=cond_bd_or_gss,
                                ca_kv=ca_kv,
                                attn_bias_or_two_vector=None,
                                attn_fn=attn_fn,
                                scale_schedule=engine.scale_schedule,
                                rope2d_freqs_grid=engine.model.rope2d_freqs_grid,
                                scale_ind=step_id,
                            )

                    logits = engine.model.get_logits(last_stage, cond_bd).mul(1 / float(args.tau))
                    logits = float(args.cfg) * logits[:condition_batch] + (1 - float(args.cfg)) * logits[condition_batch:]
                    content_idx = engine._sample_bit_labels(logits[:1], content_rng, args.top_k, args.top_p)
                    generation_idx = engine._sample_bit_labels(logits[1:2], generation_rng, args.top_k, args.top_p)

                    content_codes = engine._bit_labels_to_codes(content_idx, pn)
                    generation_codes = engine._bit_labels_to_codes(generation_idx, pn)
                    if step_id != len(engine.scale_schedule) - 1:
                        content_codes = F.interpolate(content_codes, size=final_size, mode=engine.vae.quantizer.z_interplote_up)
                        generation_codes = F.interpolate(generation_codes, size=final_size, mode=engine.vae.quantizer.z_interplote_up)

                    content_summed = content_summed + content_codes
                    generation_summed = generation_summed + generation_codes

                    if step_id in inject_steps:
                        injection_order = inject_steps.index(step_id)
                        effective_strength = STYLE_STRENGTH * float(args.decay) ** injection_order
                        generation_before_edit = generation_summed.clone()
                        delta, diagnostics = hybrid_delta_for_step(
                            step_id=step_id,
                            variant=variant,
                            generation_feature=generation_summed,
                            style_feature=style_features[step_id],
                            content_feature=content_summed,
                            args=args,
                        )
                        delta = delta.to(generation_summed)
                        generation_summed = generation_summed + effective_strength * delta
                        pfb_relative_change_by_step[step_id] = float(
                            (generation_summed - generation_before_edit).norm()
                            / generation_before_edit.norm().clamp_min(1e-8)
                        )
                        if diagnostics:
                            projection_diagnostics_by_step[step_id] = diagnostics

                    content_trace.append(content_summed.detach().float().clone())
                    generation_trace.append(generation_summed.detach().float().clone())

                    if step_id != len(engine.scale_schedule) - 1:
                        next_scale = engine.scale_schedule[step_id + 1]
                        content_next = engine._next_raw_from_summed_codes(content_summed, next_scale)
                        generation_next = engine._next_raw_from_summed_codes(generation_summed, next_scale)
                        two_streams = torch.cat((content_next, generation_next), dim=0)
                        last_stage = engine.model.word_embed(engine.model.norm0_ve(two_streams))
                        last_stage = last_stage.repeat(bs // condition_batch, 1, 1)

        return {
            "content_image_01": engine._decode_summed_codes_to_image_01(content_summed),
            "stylized_image_01": engine._decode_summed_codes_to_image_01(generation_summed),
            "content_features": content_trace,
            "generation_features": generation_trace,
            "sac_calls": controller.total_calls,
            "max_q_copy_error": controller.max_q_copy_error,
            "max_k_copy_error": controller.max_k_copy_error,
            "pfb_relative_change_by_step": pfb_relative_change_by_step,
            "projection_diagnostics_by_step": projection_diagnostics_by_step,
        }
    finally:
        controller.active = False
        for block in engine.model.unregistered_blocks:
            block.sa.kv_caching(False)


def save_case_result(
    *,
    output_dir: Path,
    content: dict[str, str],
    style: dict[str, str],
    result: dict,
    variant: dict[str, str],
    args: argparse.Namespace,
) -> None:
    image_path, comparison_path, metadata_path = runtime.case_paths(output_dir, style["style_id"], content["content_id"])
    generated_image = tensor_to_pil(result["stylized_image_01"][0])
    content_image = tensor_to_pil(result["content_image_01"][0])
    early_steps = early_steps_for_variant(variant)
    late_steps = late_steps_for_variant(variant)
    inject_steps = inject_steps_for_variant(variant)
    runtime.save_image_atomic(generated_image, image_path)
    save_comparison(content_image, style["_resolved_path"], generated_image, content["content_prompt"], variant["label"], comparison_path)
    runtime.write_metadata(metadata_path, {
        "content_id": content["content_id"],
        "content_prompt": content["content_prompt"],
        "superclass": content["superclass"],
        "evaluation_prompt": evaluation_prompt(content, style),
        "style_id": style["style_id"],
        "style_name": style["style_name"],
        "style_reference_image": style["style_reference_image"],
        "seed": args.seed,
        "pn": args.pn,
        "method": variant["name"],
        "variant_label": variant["label"],
        "style_source": "vae_encoded_style_reference_image",
        "inject_steps": inject_steps,
        "early_steps": early_steps,
        "late_steps": late_steps,
        "early_mode": variant["early_mode"],
        "late_mode": variant["late_mode"],
        "early_rank": EARLY_RANK,
        "late_rank": LATE_RANK,
        "late_top1_weight": args.late_top1_weight,
        "content_rank": args.content_rank,
        "content_variance_threshold": args.content_variance_threshold,
        "projection_style_rank": args.projection_style_rank,
        "projection_strength": args.projection_strength,
        "preserve_mean": args.preserve_mean,
        "alpha": args.alpha,
        "style_strength": STYLE_STRENGTH,
        "style_decay": args.decay,
        "sac_start_step": SAC_START_STEP,
        "sampling_top_k": args.top_k,
        "top_p": args.top_p,
        "cfg": args.cfg,
        "tau": args.tau,
        "sac_calls": result["sac_calls"],
        "max_q_copy_error": result["max_q_copy_error"],
        "max_k_copy_error": result["max_k_copy_error"],
        "pfb_relative_change_by_step": result["pfb_relative_change_by_step"],
        "projection_diagnostics_by_step": result["projection_diagnostics_by_step"],
    })


def generate(args: argparse.Namespace, sample_cases: list[tuple[dict[str, str], dict[str, str]]]) -> bool:
    runtime.load_runtime_modules()
    import torch
    from tqdm.auto import tqdm
    from var_soict.config import ExperimentConfig
    from var_soict.style_transfer import StyleTransferEngine

    bundle = runtime.load_bundle(args)
    config = ExperimentConfig(
        model_pn=args.pn,
        cfg=args.cfg,
        tau=args.tau,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        paper_alpha=args.alpha,
        paper_sac_prediction_start=SAC_START_STEP,
    )
    engine = StyleTransferEngine(bundle, config)
    output_root = args.output_root.resolve()
    variants = selected_variants(args)
    variant_dirs = {variant["name"]: output_root / variant["directory"] for variant in variants}
    total = len(sample_cases) * len(variants)
    existing = 0 if args.overwrite else sum(
        runtime.case_complete(variant_dirs[variant["name"]], style["style_id"], content["content_id"])
        for content, style in sample_cases
        for variant in variants
    )
    generated = 0
    progress = tqdm(total=total, initial=existing, desc="Hybrid band comparison", unit="image")
    started = time.time()
    try:
        for content, style in sample_cases:
            style_features = engine.get_style_features(style["_resolved_path"])
            for variant in variants:
                output_dir = variant_dirs[variant["name"]]
                if not args.overwrite and runtime.case_complete(output_dir, style["style_id"], content["content_id"]):
                    continue
                result = hybrid_generate(engine, content["content_prompt"].strip(), style_features, variant, args)
                save_case_result(output_dir=output_dir, content=content, style=style, result=result, variant=variant, args=args)
                generated += 1
                progress.update(1)
                progress.set_postfix(content=content["content_id"], style=style["style_id"], variant=variant["name"])
                del result
                gc.collect()
                torch.cuda.empty_cache()
            del style_features
        print(f"Generation complete: total={total}, new={generated}, resumed={existing}, seconds={time.time()-started:.1f}")
        return True
    finally:
        progress.close()
        del engine, bundle
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    from var_soict.clip_metrics import CLIPMetricsEvaluator

    output_root = args.output_root.resolve()
    metrics_dir = output_root / "metrics"
    evaluator = CLIPMetricsEvaluator(output_dir=metrics_dir, model_id=args.model_id, device=args.device)
    all_metrics = []
    for variant in selected_variants(args):
        generated_root = output_root / variant["directory"]
        metrics = evaluator.compute_content_ortho_metrics(
            project_root=PROJECT_ROOT,
            generated_root=generated_root,
            prompts_csv=args.prompts_csv,
            styles_csv=args.styles_csv,
        )
        if metrics.empty:
            print(f"No generated images found for {variant['name']} under {generated_root}")
            continue
        metrics.insert(0, "variant", variant["name"])
        metrics.insert(1, "variant_label", variant["label"])
        all_metrics.append(metrics)

    if not all_metrics:
        raise ValueError("No generated images found for any variant.")

    combined = pd.concat(all_metrics, ignore_index=True)
    metric_columns = ["S_txt", "S_img", "S_harmonic"]
    detail_path = metrics_dir / "random50_hybrid_ortho_band_clip_metrics.csv"
    combined.to_csv(detail_path, index=False)

    summary = combined.groupby(["variant", "variant_label"], as_index=False)[metric_columns].mean()
    summary.insert(2, "num_images", combined.groupby(["variant", "variant_label"]).size().values)
    summary = summary.sort_values("S_harmonic", ascending=False)
    summary_path = metrics_dir / "random50_hybrid_ortho_band_clip_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_path = metrics_dir / "random50_hybrid_ortho_band_clip_metrics_summary.png"
    figure, axis = plt.subplots(figsize=(10, 5))
    x = range(len(summary))
    width = 0.24
    for offset, column in zip((-width, 0.0, width), metric_columns):
        axis.bar([value + offset for value in x], summary[column], width=width, label=column)
    axis.set_xticks(list(x))
    axis.set_xticklabels(summary["variant_label"], rotation=12, ha="right")
    axis.set_ylabel("CLIP cosine similarity")
    axis.set_title("Random-50 hybrid normal/content-orthogonal SVD-band comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    print("saved metrics:", detail_path)
    print("saved summary:", summary_path)
    print("saved plot:", plot_path)
    print("\nSummary ranked by S_harmonic:\n", summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    contents, styles = runtime.validate_inputs(args)
    print(f"Validated {len(contents)} prompts x {len(styles)} styles = {len(contents) * len(styles)} cases")
    print("Selected variants:", ", ".join(variant["name"] for variant in selected_variants(args)))
    sample_cases = select_sample_cases(args, contents, styles)
    print(f"Random comparison sample size: {len(sample_cases)}")
    if args.validate_only:
        return
    complete = True
    if not args.metrics_only:
        complete = generate(args, sample_cases)
    if not args.generate_only and complete:
        evaluate(args)


if __name__ == "__main__":
    main()
