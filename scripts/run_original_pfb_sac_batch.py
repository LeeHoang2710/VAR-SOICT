#!/usr/bin/env python3
"""Resumably generate and evaluate all 190 x 10 original PFB+SAC cases."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-csv", type=Path, default=PROJECT_ROOT / "prompts/content_prompts_190.csv")
    parser.add_argument("--styles-csv", type=Path, default=PROJECT_ROOT / "styles/quantitative_eval_styles_10.csv")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/pfb_sac_f3")
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "0.60M", "1M"), default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
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
    return args


def evaluation_prompt(content, style) -> str:
    descriptor = style["style_descriptor"].strip()
    suffix = descriptor if descriptor.lower().endswith("style") else f"{descriptor} style"
    return f"{content['content_prompt']}, {content['superclass']}, in {suffix}"


def tensor_to_pil(image):
    array = image.detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return runtime.Image.fromarray(array, mode="RGB")


def generate(args, contents, styles) -> bool:
    runtime.load_runtime_modules()
    import torch
    from tqdm.auto import tqdm
    from var_soict.config import ExperimentConfig
    from var_soict.style_transfer import StyleTransferEngine

    bundle = runtime.load_bundle(args)
    config = ExperimentConfig(
        model_pn=args.pn, cfg=args.cfg, tau=args.tau, top_k=args.top_k,
        top_p=args.top_p, seed=args.seed, paper_alpha=1.0,
        paper_pfb_feature_index=2, paper_sac_prediction_start=2,
        base_style_strength=1.0,
    )
    engine = StyleTransferEngine(bundle, config)
    output_dir = args.output_dir.resolve()
    total = len(contents) * len(styles)
    existing = 0 if args.overwrite else sum(
        runtime.case_complete(output_dir, style["style_id"], content["content_id"])
        for content in contents for style in styles
    )
    generated = 0
    progress = tqdm(total=total, initial=existing, desc="Original PFB+SAC", unit="image")
    started = time.time()
    try:
        for content in contents:
            for style in styles:
                if not args.overwrite and runtime.case_complete(output_dir, style["style_id"], content["content_id"]):
                    continue
                if args.limit is not None and generated >= args.limit:
                    print(f"Stopped after --limit={args.limit}; rerun the same command to resume.")
                    return False
                prompt = evaluation_prompt(content, style)
                result = engine.paper_dual_path_generate(
                    prompt, engine.get_style_features(style["_resolved_path"]),
                    pfb_feature_indices=[2], rank=None, alpha=1.0,
                    style_strength=1.0, style_decay=1.0,
                    sac_prediction_start=2, sac_strength=1.0, enable_sac=True,
                )
                image_path, comparison_path, metadata_path = runtime.case_paths(
                    output_dir, style["style_id"], content["content_id"]
                )
                generated_image = tensor_to_pil(result["stylized_image_01"][0])
                content_image = tensor_to_pil(result["content_image_01"][0])
                runtime.save_image_atomic(generated_image, image_path)
                runtime.save_comparison(content_image, style["_resolved_path"], generated_image, prompt, comparison_path)
                runtime.write_metadata(metadata_path, {
                    "content_id": content["content_id"], "content_prompt": content["content_prompt"],
                    "superclass": content["superclass"], "evaluation_prompt": prompt,
                    "style_id": style["style_id"], "style_name": style["style_name"],
                    "style_reference_image": style["style_reference_image"],
                    "seed": args.seed, "pn": args.pn, "method": "original_pfb_sac",
                    "inject_step": 2, "inject_step_numbering": "zero_based", "paper_feature": "F3",
                    "alpha": 1.0, "rank": "full", "style_strength": 1.0, "sac_start_step": 2,
                    "sac_calls": result["sac_calls"], "max_q_copy_error": result["max_q_copy_error"],
                    "max_k_copy_error": result["max_k_copy_error"],
                })
                generated += 1
                progress.update(1)
                progress.set_postfix(content=content["content_id"], style=style["style_id"])
                del result, generated_image, content_image
        print(f"Generation complete: total={total}, new={generated}, resumed={existing}, seconds={time.time()-started:.1f}")
        return True
    finally:
        progress.close()
        del engine, bundle
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(args) -> None:
    import pandas as pd
    from var_soict.clip_metrics import CLIPMetricsEvaluator

    metrics_dir = args.output_dir.resolve() / "metrics"
    evaluator = CLIPMetricsEvaluator(output_dir=metrics_dir, model_id=args.model_id, device=args.device)
    metrics = evaluator.compute_content_ortho_metrics(
        project_root=PROJECT_ROOT, generated_root=args.output_dir,
        prompts_csv=args.prompts_csv, styles_csv=args.styles_csv,
    )
    if metrics.empty:
        raise ValueError(f"No generated images found under {args.output_dir}")
    columns = ["S_txt", "S_img", "S_harmonic"]
    details_path = metrics_dir / "pfb_sac_clip_metrics.csv"
    metrics.to_csv(details_path, index=False)
    by_style = metrics.groupby(["style_id", "style_name"], as_index=False)[columns].mean()
    by_style.insert(2, "num_images", metrics.groupby(["style_id", "style_name"]).size().values)
    by_style_path = metrics_dir / "pfb_sac_clip_metrics_by_style.csv"
    by_style.to_csv(by_style_path, index=False)
    overall = pd.DataFrame([{"num_images": len(metrics), **{column: float(metrics[column].mean()) for column in columns}}])
    overall_path = metrics_dir / "pfb_sac_clip_metrics_overall.csv"
    overall.to_csv(overall_path, index=False)
    print("saved metrics:", details_path)
    print("saved style summary:", by_style_path)
    print("saved overall summary:", overall_path)
    print("\nOverall FineStyle CLIP metrics:\n", overall.to_string(index=False))
    print("\nPer-style metrics:\n", by_style.to_string(index=False))


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
