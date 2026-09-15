#!/usr/bin/env python3
"""Generate and evaluate 0-5 content-orthogonal PFB+SAC with decay.

This extends the content-orthogonal idea from the friend branch from a single
injection scale to the Variant-2 schedule from notebook 17:

    scales 0,1,2,3,4,5
    style rank 1
    content rank 1
    strength 1.0, decay 0.75 by injection order
    SAC from scale 2 onward

At every selected scale, the content-orthogonal projection uses the clean
content stream at the same scale as the content basis. This is the important
difference from simply reusing the already-stylized generation stream.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT / "src", PROJECT_ROOT / "Infinity", PROJECT_ROOT / "Infinity/tools", PROJECT_ROOT / "scripts"):
    sys.path.insert(0, str(directory))

import run_content_ortho_batch as runtime


VARIANT_NAME = "content_ortho_multiscale_decay"
INJECT_STEPS = [0, 1, 2, 3, 4, 5]
STYLE_RANK = 1
CONTENT_RANK = 1
STYLE_STRENGTH = 1.0
STYLE_DECAY = 0.75
PROJECTION_STRENGTH = 1.0
SAC_START_STEP = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-csv", type=Path, default=PROJECT_ROOT / "prompts/content_prompts_190.csv")
    parser.add_argument("--styles-csv", type=Path, default=PROJECT_ROOT / "styles/quantitative_eval_styles_10.csv")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/content_ortho_multiscale_decay_0_1_2_3_4_5")
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "0.60M", "1M"), default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900, help="Sampling top-k, not SVD rank.")
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--inject-step", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--style-rank", type=int, default=STYLE_RANK)
    parser.add_argument("--content-rank", type=int, default=CONTENT_RANK)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=STYLE_STRENGTH)
    parser.add_argument("--decay", type=float, default=STYLE_DECAY)
    parser.add_argument("--projection-strength", type=float, default=PROJECTION_STRENGTH)
    parser.add_argument("--sac-start-step", type=int, default=SAC_START_STEP)
    parser.add_argument("--no-preserve-mean", action="store_true")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None, help="CLIP device; defaults to CUDA when available.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Generate at most N missing cases, then stop before metrics.")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.generate_only and args.metrics_only:
        parser.error("--generate-only and --metrics-only are mutually exclusive")
    if args.style_rank < 1:
        parser.error("--style-rank must be positive")
    if args.content_rank < 1:
        parser.error("--content-rank must be positive")
    if args.projection_strength < 0 or args.projection_strength > 1:
        parser.error("--projection-strength must be in [0, 1]")
    if args.sac_start_step < 0:
        parser.error("--sac-start-step must be non-negative")
    return args


def evaluation_prompt(content: dict[str, str], style: dict[str, str]) -> str:
    descriptor = style["style_descriptor"].strip()
    suffix = descriptor if descriptor.lower().endswith("style") else f"{descriptor} style"
    return f"{content['content_prompt']}, {content['superclass']}, in {suffix}"


def tensor_to_pil(image):
    array = image.detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return runtime.Image.fromarray(array, mode="RGB")


def save_comparison(content_image, style_path: str, generated_image, prompt: str, path: Path) -> None:
    size = generated_image.size
    style_image = runtime.ImageOps.fit(runtime.Image.open(style_path).convert("RGB"), size, runtime.Image.Resampling.LANCZOS)
    content_image = runtime.ImageOps.fit(content_image.convert("RGB"), size, runtime.Image.Resampling.LANCZOS)
    header = 80
    canvas = runtime.Image.new("RGB", (size[0] * 3, size[1] + header), "white")
    draw = runtime.ImageDraw.Draw(canvas)
    labels = (f"Prompt: {prompt}", "Style reference", "Content-ortho 0-5 decay")
    for column, (label, image) in enumerate(zip(labels, (content_image, style_image, generated_image.convert("RGB")))):
        x = column * size[0]
        canvas.paste(image, (x, header))
        draw.text((x + 12, 24), label[:72], fill="black")
    runtime.save_image_atomic(canvas, path)


def content_ortho_multiscale_generate(
    engine,
    prompt: str,
    style_features: list,
    *,
    seed: int,
    cfg: float,
    tau: float,
    top_k: int,
    top_p: float,
    inject_steps: list[int],
    sac_start_step: int,
    alpha: float,
    style_rank: int,
    content_rank: int,
    style_strength: float,
    style_decay: float,
    projection_strength: float,
    preserve_mean: bool,
):
    import torch
    import torch.nn.functional as F
    from var_soict.feature_hypotheses import content_orthogonal_feature_blend
    from var_soict.style_transfer import PaperSACPatch, SACController

    if cfg < 1.0:
        raise ValueError("CFG must be >= 1.0 for this dual-stream experiment.")
    inject_steps = sorted(set(int(step) for step in inject_steps))
    if not inject_steps or any(step < 0 or step >= len(engine.scale_schedule) for step in inject_steps):
        raise ValueError("Invalid content-orthogonal injection steps.")
    if sac_start_step >= len(engine.scale_schedule):
        raise ValueError("Invalid SAC start step.")

    engine.model.eval()
    base_batch = 1
    condition_batch = 2
    content_rng = torch.Generator(device=engine.device).manual_seed(seed)
    generation_rng = torch.Generator(device=engine.device).manual_seed(seed)

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

    controller = SACController(base_batch)
    controller.sac_strength = 1.0
    controller.reset_statistics()

    for block in engine.model.unregistered_blocks:
        block.sa.kv_caching(True)

    try:
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            sac_context = PaperSACPatch(engine.model, controller)
            with sac_context:
                for step_id, pn in enumerate(engine.scale_schedule):
                    controller.active = step_id >= sac_start_step
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

                    logits = engine.model.get_logits(last_stage, cond_bd).mul(1 / float(tau))
                    logits = float(cfg) * logits[:condition_batch] + (1 - float(cfg)) * logits[condition_batch:]
                    content_idx = engine._sample_bit_labels(logits[:1], content_rng, top_k, top_p)
                    generation_idx = engine._sample_bit_labels(logits[1:2], generation_rng, top_k, top_p)

                    content_codes = engine._bit_labels_to_codes(content_idx, pn)
                    generation_codes = engine._bit_labels_to_codes(generation_idx, pn)
                    if step_id != len(engine.scale_schedule) - 1:
                        content_codes = F.interpolate(content_codes, size=final_size, mode=engine.vae.quantizer.z_interplote_up)
                        generation_codes = F.interpolate(generation_codes, size=final_size, mode=engine.vae.quantizer.z_interplote_up)

                    content_summed = content_summed + content_codes
                    generation_summed = generation_summed + generation_codes

                    if step_id in inject_steps:
                        injection_order = inject_steps.index(step_id)
                        effective_strength = float(style_strength) * float(style_decay) ** injection_order
                        generation_before_edit = generation_summed.clone()
                        generation_summed = content_orthogonal_feature_blend(
                            generation_summed,
                            style_features[step_id].to(generation_summed),
                            content_summed,
                            style_rank=style_rank,
                            content_rank=content_rank,
                            alpha=alpha,
                            strength=effective_strength,
                            projection_strength=projection_strength,
                            preserve_mean=preserve_mean,
                        )
                        pfb_relative_change_by_step[step_id] = float(
                            (generation_summed - generation_before_edit).norm()
                            / generation_before_edit.norm().clamp_min(1e-8)
                        )

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
        }
    finally:
        controller.active = False
        for block in engine.model.unregistered_blocks:
            block.sa.kv_caching(False)


def generate(args: argparse.Namespace, contents: list[dict[str, str]], styles: list[dict[str, str]]) -> bool:
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
        paper_sac_prediction_start=args.sac_start_step,
    )
    engine = StyleTransferEngine(bundle, config)
    output_dir = args.output_dir.resolve()
    total = len(contents) * len(styles)
    existing = 0 if args.overwrite else sum(
        runtime.case_complete(output_dir, style["style_id"], content["content_id"])
        for content in contents
        for style in styles
    )
    generated = 0
    progress = tqdm(total=total, initial=existing, desc="Content-ortho 0-5 decay", unit="image")
    started = time.time()
    try:
        for content in contents:
            prompt = content["content_prompt"].strip()
            for style in styles:
                if not args.overwrite and runtime.case_complete(output_dir, style["style_id"], content["content_id"]):
                    continue
                if args.limit is not None and generated >= args.limit:
                    print(f"Stopped after --limit={args.limit}; rerun the same command to resume.")
                    return False
                style_features = engine.get_style_features(style["_resolved_path"])
                result = content_ortho_multiscale_generate(
                    engine,
                    prompt,
                    style_features,
                    seed=args.seed,
                    cfg=args.cfg,
                    tau=args.tau,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    inject_steps=INJECT_STEPS,
                    sac_start_step=args.sac_start_step,
                    alpha=args.alpha,
                    style_rank=args.style_rank,
                    content_rank=args.content_rank,
                    style_strength=args.strength,
                    style_decay=args.decay,
                    projection_strength=args.projection_strength,
                    preserve_mean=not args.no_preserve_mean,
                )
                image_path, comparison_path, metadata_path = runtime.case_paths(
                    output_dir, style["style_id"], content["content_id"]
                )
                generated_image = tensor_to_pil(result["stylized_image_01"][0])
                content_image = tensor_to_pil(result["content_image_01"][0])
                runtime.save_image_atomic(generated_image, image_path)
                save_comparison(content_image, style["_resolved_path"], generated_image, prompt, comparison_path)
                runtime.write_metadata(metadata_path, {
                    "content_id": content["content_id"],
                    "content_prompt": prompt,
                    "superclass": content["superclass"],
                    "evaluation_prompt": evaluation_prompt(content, style),
                    "style_id": style["style_id"],
                    "style_name": style["style_name"],
                    "style_reference_image": style["style_reference_image"],
                    "seed": args.seed,
                    "pn": args.pn,
                    "method": VARIANT_NAME,
                    "style_source": "vae_encoded_style_reference_image",
                    "inject_steps": INJECT_STEPS,
                    "inject_step_numbering": "zero_based",
                    "style_rank": args.style_rank,
                    "content_rank": args.content_rank,
                    "sampling_top_k": args.top_k,
                    "alpha": args.alpha,
                    "style_strength": args.strength,
                    "style_decay": args.decay,
                    "projection_strength": args.projection_strength,
                    "preserve_mean": not args.no_preserve_mean,
                    "sac_start_step": args.sac_start_step,
                    "sac_calls": result["sac_calls"],
                    "max_q_copy_error": result["max_q_copy_error"],
                    "max_k_copy_error": result["max_k_copy_error"],
                    "pfb_relative_change_by_step": result["pfb_relative_change_by_step"],
                })
                generated += 1
                progress.update(1)
                progress.set_postfix(content=content["content_id"], style=style["style_id"])
                del result, generated_image, content_image, style_features
                gc.collect()
                torch.cuda.empty_cache()
        print(f"Generation complete: total={total}, new={generated}, resumed={existing}, seconds={time.time()-started:.1f}")
        return True
    finally:
        progress.close()
        del engine, bundle
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(args: argparse.Namespace) -> None:
    import pandas as pd
    from var_soict.clip_metrics import CLIPMetricsEvaluator

    metrics_dir = args.output_dir.resolve() / "metrics"
    evaluator = CLIPMetricsEvaluator(output_dir=metrics_dir, model_id=args.model_id, device=args.device)
    metrics = evaluator.compute_content_ortho_metrics(
        project_root=PROJECT_ROOT,
        generated_root=args.output_dir,
        prompts_csv=args.prompts_csv,
        styles_csv=args.styles_csv,
    )
    if metrics.empty:
        raise ValueError(f"No generated images found under {args.output_dir}")
    columns = ["S_txt", "S_img", "S_harmonic"]
    details_path = metrics_dir / "content_ortho_multiscale_decay_clip_metrics.csv"
    metrics.to_csv(details_path, index=False)
    by_style = metrics.groupby(["style_id", "style_name"], as_index=False)[columns].mean()
    by_style.insert(2, "num_images", metrics.groupby(["style_id", "style_name"]).size().values)
    by_style_path = metrics_dir / "content_ortho_multiscale_decay_clip_metrics_by_style.csv"
    by_style.to_csv(by_style_path, index=False)
    overall = pd.DataFrame([{"num_images": len(metrics), **{column: float(metrics[column].mean()) for column in columns}}])
    overall_path = metrics_dir / "content_ortho_multiscale_decay_clip_metrics_overall.csv"
    overall.to_csv(overall_path, index=False)
    print("saved metrics:", details_path)
    print("saved style summary:", by_style_path)
    print("saved overall summary:", overall_path)
    print("\nOverall FineStyle CLIP metrics:\n", overall.to_string(index=False))


def main() -> None:
    args = parse_args()
    contents, styles = runtime.validate_inputs(args)
    print(f"Validated {len(contents)} prompts x {len(styles)} styles = {len(contents) * len(styles)} cases")
    if args.validate_only:
        return
    complete = True
    if not args.metrics_only:
        complete = generate(args, contents, styles)
    if not args.generate_only and complete:
        evaluate(args)


if __name__ == "__main__":
    main()
