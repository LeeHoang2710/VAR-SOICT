#!/usr/bin/env python3
"""Run a quick 50-case comparison across three PFB+SAC variants.

The sampled cases are drawn from the full 190 prompt x 10 style grid and saved
to ``sample_cases.csv`` so reruns compare the exact same cases.

Compared variants:
  1. original PFB+SAC: single F3 / zero-based scale 2, full-rank SVD
  2. Variant 2: top-1 PFB+SAC at scales 0,1,2,3,4,5 with decay 0.75
  3. content-orthogonal Variant 2: same 0-5 decay schedule, but each injected
     feature is projected away from the clean content stream at that scale
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT / "src", PROJECT_ROOT / "Infinity", PROJECT_ROOT / "Infinity/tools", PROJECT_ROOT / "scripts"):
    sys.path.insert(0, str(directory))

import run_content_ortho_batch as runtime
from run_content_ortho_multiscale_decay_batch import content_ortho_multiscale_generate


ORIGINAL_STEPS = [2]
ORIGINAL_RANK = None
VARIANT2_STEPS = [0, 1, 2, 3, 4, 5]
VARIANT2_RANK = 1
VARIANT2_DECAY = 0.75
CONTENT_ORTHO_STYLE_RANK = 1
CONTENT_ORTHO_CONTENT_RANK = 1
CONTENT_ORTHO_PROJECTION_STRENGTH = 1.0
SAC_START_STEP = 2

VARIANTS = [
    {
        "name": "original_pfb_sac",
        "label": "Original PFB+SAC F3 full-rank",
        "directory": "original_pfb_sac_f3_full_rank",
    },
    {
        "name": "variant2_top1_decay",
        "label": "Variant 2 top-1 0-5 PFB+SAC decay",
        "directory": "variant2_top1_decay_0_1_2_3_4_5",
    },
    {
        "name": "content_ortho_multiscale_decay",
        "label": "Content-orthogonal 0-5 PFB+SAC decay",
        "directory": "content_ortho_multiscale_decay_0_1_2_3_4_5",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-csv", type=Path, default=PROJECT_ROOT / "prompts/content_prompts_190.csv")
    parser.add_argument("--styles-csv", type=Path, default=PROJECT_ROOT / "styles/quantitative_eval_styles_10.csv")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/random50_three_variant_comparison_seed2026")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "0.60M", "1M"), default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900, help="Sampling top-k, not SVD rank.")
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None, help="CLIP device; defaults to CUDA when available.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resample", action="store_true", help="Ignore an existing sample_cases.csv and draw a new 50-case sample.")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--inject-step", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    if args.generate_only and args.metrics_only:
        parser.error("--generate-only and --metrics-only are mutually exclusive")
    return args


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
    sample_path = args.output_root.resolve() / "sample_cases.csv"
    contents_by_id = {row["content_id"]: row for row in contents}
    styles_by_id = {row["style_id"]: row for row in styles}
    if sample_path.exists() and not args.resample:
        sample_cases = read_sample(sample_path, contents_by_id, styles_by_id)
        print(f"Loaded existing sample: {sample_path} ({len(sample_cases)} cases)")
        return sample_cases

    population = [(content, style) for content in contents for style in styles]
    if args.sample_size > len(population):
        raise ValueError(f"--sample-size={args.sample_size} exceeds {len(population)} available cases")
    rng = random.Random(args.sample_seed)
    sample_cases = rng.sample(population, args.sample_size)
    write_sample(sample_path, sample_cases)
    print(f"Saved sampled cases: {sample_path}")
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


def save_case_result(
    *,
    output_dir: Path,
    content: dict[str, str],
    style: dict[str, str],
    result: dict,
    variant_name: str,
    variant_label: str,
    config_payload: dict,
    args: argparse.Namespace,
) -> None:
    image_path, comparison_path, metadata_path = runtime.case_paths(output_dir, style["style_id"], content["content_id"])
    generated_image = tensor_to_pil(result["stylized_image_01"][0])
    content_image = tensor_to_pil(result["content_image_01"][0])
    runtime.save_image_atomic(generated_image, image_path)
    save_comparison(content_image, style["_resolved_path"], generated_image, content["content_prompt"], variant_label, comparison_path)
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
        "method": variant_name,
        "variant_label": variant_label,
        "style_source": "vae_encoded_style_reference_image",
        "sampling_top_k": args.top_k,
        "top_p": args.top_p,
        "cfg": args.cfg,
        "tau": args.tau,
        **config_payload,
        "sac_calls": result["sac_calls"],
        "max_q_copy_error": result["max_q_copy_error"],
        "max_k_copy_error": result["max_k_copy_error"],
        "pfb_relative_change_by_step": result["pfb_relative_change_by_step"],
    })


def generate_original(engine, args: argparse.Namespace, content: dict[str, str], style: dict[str, str]):
    return engine.paper_dual_path_generate(
        content["content_prompt"].strip(),
        engine.get_style_features(style["_resolved_path"]),
        seed=args.seed,
        cfg=args.cfg,
        tau=args.tau,
        top_k=args.top_k,
        top_p=args.top_p,
        pfb_feature_indices=ORIGINAL_STEPS,
        sac_prediction_start=SAC_START_STEP,
        edit_mode="pfb",
        alpha=args.alpha,
        rank=ORIGINAL_RANK,
        style_strength=1.0,
        style_decay=1.0,
        sac_strength=1.0,
        enable_sac=True,
    )


def generate_variant2(engine, args: argparse.Namespace, content: dict[str, str], style: dict[str, str]):
    return engine.paper_dual_path_generate(
        content["content_prompt"].strip(),
        engine.get_style_features(style["_resolved_path"]),
        seed=args.seed,
        cfg=args.cfg,
        tau=args.tau,
        top_k=args.top_k,
        top_p=args.top_p,
        pfb_feature_indices=VARIANT2_STEPS,
        sac_prediction_start=SAC_START_STEP,
        edit_mode="pfb",
        alpha=args.alpha,
        rank=VARIANT2_RANK,
        style_strength=1.0,
        style_decay=VARIANT2_DECAY,
        sac_strength=1.0,
        enable_sac=True,
    )


def generate_content_ortho(engine, args: argparse.Namespace, content: dict[str, str], style: dict[str, str]):
    return content_ortho_multiscale_generate(
        engine,
        content["content_prompt"].strip(),
        engine.get_style_features(style["_resolved_path"]),
        seed=args.seed,
        cfg=args.cfg,
        tau=args.tau,
        top_k=args.top_k,
        top_p=args.top_p,
        inject_steps=VARIANT2_STEPS,
        sac_start_step=SAC_START_STEP,
        alpha=args.alpha,
        style_rank=CONTENT_ORTHO_STYLE_RANK,
        content_rank=CONTENT_ORTHO_CONTENT_RANK,
        style_strength=1.0,
        style_decay=VARIANT2_DECAY,
        projection_strength=CONTENT_ORTHO_PROJECTION_STRENGTH,
        preserve_mean=True,
    )


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
    variant_dirs = {variant["name"]: output_root / variant["directory"] for variant in VARIANTS}
    generators = {
        "original_pfb_sac": generate_original,
        "variant2_top1_decay": generate_variant2,
        "content_ortho_multiscale_decay": generate_content_ortho,
    }
    configs = {
        "original_pfb_sac": {
            "inject_steps": ORIGINAL_STEPS,
            "inject_step_numbering": "zero_based",
            "svd_rank": "full",
            "alpha": args.alpha,
            "style_strength": 1.0,
            "style_decay": 1.0,
            "sac_start_step": SAC_START_STEP,
        },
        "variant2_top1_decay": {
            "inject_steps": VARIANT2_STEPS,
            "inject_step_numbering": "zero_based",
            "svd_rank": VARIANT2_RANK,
            "alpha": args.alpha,
            "style_strength": 1.0,
            "style_decay": VARIANT2_DECAY,
            "sac_start_step": SAC_START_STEP,
        },
        "content_ortho_multiscale_decay": {
            "inject_steps": VARIANT2_STEPS,
            "inject_step_numbering": "zero_based",
            "style_rank": CONTENT_ORTHO_STYLE_RANK,
            "content_rank": CONTENT_ORTHO_CONTENT_RANK,
            "alpha": args.alpha,
            "style_strength": 1.0,
            "style_decay": VARIANT2_DECAY,
            "projection_strength": CONTENT_ORTHO_PROJECTION_STRENGTH,
            "preserve_mean": True,
            "sac_start_step": SAC_START_STEP,
        },
    }
    total = len(sample_cases) * len(VARIANTS)
    existing = 0 if args.overwrite else sum(
        runtime.case_complete(variant_dirs[variant["name"]], style["style_id"], content["content_id"])
        for content, style in sample_cases
        for variant in VARIANTS
    )
    generated = 0
    progress = tqdm(total=total, initial=existing, desc="Random-50 variant comparison", unit="image")
    started = time.time()
    try:
        for content, style in sample_cases:
            for variant in VARIANTS:
                output_dir = variant_dirs[variant["name"]]
                if not args.overwrite and runtime.case_complete(output_dir, style["style_id"], content["content_id"]):
                    continue
                result = generators[variant["name"]](engine, args, content, style)
                save_case_result(
                    output_dir=output_dir,
                    content=content,
                    style=style,
                    result=result,
                    variant_name=variant["name"],
                    variant_label=variant["label"],
                    config_payload=configs[variant["name"]],
                    args=args,
                )
                generated += 1
                progress.update(1)
                progress.set_postfix(content=content["content_id"], style=style["style_id"], variant=variant["name"])
                del result
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
    import matplotlib.pyplot as plt
    import pandas as pd
    from var_soict.clip_metrics import CLIPMetricsEvaluator

    output_root = args.output_root.resolve()
    metrics_dir = output_root / "metrics"
    evaluator = CLIPMetricsEvaluator(output_dir=metrics_dir, model_id=args.model_id, device=args.device)
    all_metrics = []
    for variant in VARIANTS:
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
    detail_path = metrics_dir / "random50_three_variant_clip_metrics.csv"
    combined.to_csv(detail_path, index=False)

    summary = combined.groupby(["variant", "variant_label"], as_index=False)[metric_columns].mean()
    summary.insert(2, "num_images", combined.groupby(["variant", "variant_label"]).size().values)
    summary = summary.sort_values("S_harmonic", ascending=False)
    summary_path = metrics_dir / "random50_three_variant_clip_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_path = metrics_dir / "random50_three_variant_clip_metrics_summary.png"
    figure, axis = plt.subplots(figsize=(11, 5))
    x = range(len(summary))
    width = 0.24
    for offset, column in zip((-width, 0.0, width), metric_columns):
        axis.bar([value + offset for value in x], summary[column], width=width, label=column)
    axis.set_xticks(list(x))
    axis.set_xticklabels(summary["variant_label"], rotation=12, ha="right")
    axis.set_ylabel("CLIP cosine similarity")
    axis.set_title("Random-50 initial comparison across PFB+SAC variants")
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
