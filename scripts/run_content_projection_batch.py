#!/usr/bin/env python3
"""Generate 190 prompts x 10 styles with content-projected PFB, resumably."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_content_ortho_batch as runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-csv", type=Path, default=PROJECT_ROOT / "prompts/content_prompts_190.csv")
    parser.add_argument(
        "--styles-csv", type=Path,
        default=PROJECT_ROOT / "styles/quantitative_eval_styles_10.csv",
    )
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/content_projection_step_03",
    )
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "1M"), default="0.25M")
    parser.add_argument(
        "--inject-step", type=int, default=3,
        help="Zero-based AR scale. Step 3 is the default selected by the 5x5 pilot.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--style-rank", type=int, default=1)
    parser.add_argument(
        "--content-rank", type=int, default=1,
        help="Fixed content rank or maximum adaptive rank; use 0 for no rank cap.",
    )
    parser.add_argument(
        "--content-variance-threshold", type=float, default=None,
        help="Optional cumulative content energy in (0,1], e.g. 0.9 for adaptive rank.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument(
        "--preserve-mean", action="store_true",
        help="Add a separate raw mean/palette term before projection (off by default).",
    )
    parser.add_argument("--sac", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Generate at most N missing cases.")
    args = parser.parse_args()
    if args.inject_step < 0:
        parser.error("--inject-step must be non-negative")
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
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    contents, styles = runtime.validate_inputs(args)
    total = len(contents) * len(styles)
    print(f"Validated {len(contents)} prompts x {len(styles)} styles = {total} cases")
    if args.validate_only:
        return

    runtime.load_runtime_modules()
    bundle = runtime.load_bundle(args)
    if args.inject_step >= len(bundle.scale_schedule):
        raise ValueError(
            f"inject step {args.inject_step} outside {len(bundle.scale_schedule)}-step schedule"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    content_rank = None if args.content_rank == 0 else args.content_rank

    style_features = {}
    for style in runtime.tqdm(styles, desc="Encoding style references", unit="style"):
        style_features[style["style_id"]] = runtime.style_feature_at_step(
            bundle, style["_resolved_path"], args.inject_step
        )

    existing = 0 if args.overwrite else sum(
        runtime.case_complete(output_dir, style["style_id"], content["content_id"])
        for content in contents
        for style in styles
    )
    generated, skipped, failed = 0, existing, 0
    started = time.time()
    image_progress = runtime.tqdm(total=total, initial=existing, desc="Content Projection", unit="image")
    content_progress = runtime.tqdm(contents, desc="Content prompts", unit="prompt")

    for content in content_progress:
        missing_styles = [
            style for style in styles
            if args.overwrite or not runtime.case_complete(
                output_dir, style["style_id"], content["content_id"]
            )
        ]
        if not missing_styles:
            continue

        prompt = content["content_prompt"].strip()
        content_progress.set_postfix(content=content["content_id"], missing=len(missing_styles))
        content_run = runtime.run_model(
            bundle.infinity_model.autoregressive_infer_cfg,
            **runtime.infer_kwargs(bundle, [prompt], args, trace=True),
        )
        content_trace = content_run[3]
        content_image = runtime.result_image(content_run, 0)

        for style in missing_styles:
            if args.limit is not None and generated >= args.limit:
                image_progress.close()
                runtime.tqdm.write(f"Stopped after --limit={args.limit}; rerun to resume.")
                return

            image_path, comparison_path, metadata_path = runtime.case_paths(
                output_dir, style["style_id"], content["content_id"]
            )
            try:
                diagnostics = []
                result = runtime.run_model(
                    bundle.infinity_model.autoregressive_infer_content_projection,
                    **runtime.infer_kwargs(bundle, [prompt, prompt], args, trace=False),
                    style_feature=style_features[style["style_id"]],
                    inject_step=args.inject_step,
                    f_con=content_trace,
                    style_rank=args.style_rank,
                    content_rank=content_rank,
                    content_variance_threshold=args.content_variance_threshold,
                    alpha=args.alpha,
                    strength=args.strength,
                    projection_strength=args.projection_strength,
                    preserve_mean=args.preserve_mean,
                    projection_diagnostics=diagnostics,
                    sac=args.sac,
                )
                generated_image = runtime.result_image(result, 1)
                runtime.save_image_atomic(generated_image, image_path)
                runtime.save_comparison(
                    content_image, style["_resolved_path"], generated_image, prompt, comparison_path
                )
                runtime.write_metadata(metadata_path, {
                    "content_id": content["content_id"],
                    "content_prompt": prompt,
                    "style_id": style["style_id"],
                    "style_name": style["style_name"],
                    "style_reference_image": style["style_reference_image"],
                    "output_image": str(image_path),
                    "comparison_image": str(comparison_path),
                    "seed": args.seed,
                    "pn": args.pn,
                    "inject_step": args.inject_step,
                    "inject_step_numbering": "zero_based",
                    "method": "content_projection",
                    "style_rank": args.style_rank,
                    "content_rank": content_rank,
                    "content_variance_threshold": args.content_variance_threshold,
                    "alpha": args.alpha,
                    "strength": args.strength,
                    "projection_strength": args.projection_strength,
                    "preserve_mean": args.preserve_mean,
                    "sac": args.sac,
                    "projection_diagnostics": diagnostics,
                })
                generated += 1
                image_progress.update(1)
                image_progress.set_postfix(
                    content=content["content_id"], style=style["style_id"], failed=failed
                )
                del result, generated_image
            except Exception as exc:
                failed += 1
                image_progress.close()
                runtime.tqdm.write(
                    f"FAILED {content['content_id']} x {style['style_id']}: {exc}",
                    file=sys.stderr,
                )
                raise

        del content_run, content_trace, content_image
        gc.collect()
        runtime.torch.cuda.empty_cache()

    image_progress.close()
    elapsed = time.time() - started
    print(
        f"Complete: generated={generated}, skipped={skipped}, failed={failed}, "
        f"seconds={elapsed:.1f}, output={output_dir}"
    )


if __name__ == "__main__":
    main()
