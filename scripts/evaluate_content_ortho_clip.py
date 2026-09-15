#!/usr/bin/env python3
"""Evaluate content-ortho outputs with FineStyle CLIP metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from var_soict.clip_metrics import CLIPMetricsEvaluator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/content_ortho_step_02",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/content_ortho_step_02/metrics",
    )
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = CLIPMetricsEvaluator(
        output_dir=args.output_dir,
        model_id=args.model_id,
        device=args.device,
    )
    metrics = evaluator.compute_content_ortho_metrics(
        project_root=PROJECT_ROOT,
        generated_root=args.generated_root,
    )
    _, by_style, overall = evaluator.save_content_ortho_metrics(metrics)
    print("\nOverall FineStyle CLIP metrics:")
    print(overall.to_string(index=False))
    print("\nPer-style metrics:")
    print(by_style.to_string(index=False))


if __name__ == "__main__":
    main()
