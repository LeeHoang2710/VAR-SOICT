#!/usr/bin/env python3
"""Generate one image with the original PFB+SAC method."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
INFINITY_DIR = PROJECT_ROOT / "Infinity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help='Prompt including a style phrase, e.g. "a cat, in oil painting style".')
    parser.add_argument("--style-image", required=True, type=Path, help="Reference style image.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/pfb_sac.png")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "0.60M", "1M"), default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    style_image = args.style_image.resolve()
    if not style_image.is_file():
        raise FileNotFoundError(style_image)

    for import_dir in (SRC_DIR, INFINITY_DIR, INFINITY_DIR / "tools", PROJECT_ROOT / "scripts"):
        value = str(import_dir)
        if value not in sys.path:
            sys.path.insert(0, value)

    # Reuse the full-precision official Infinity loader used by the batch CLI.
    import run_content_ortho_batch as runtime

    runtime.load_runtime_modules()
    bundle = runtime.load_bundle(args)

    from var_soict.config import ExperimentConfig
    from var_soict.plotting import save_image_tensor
    from var_soict.style_transfer import StyleTransferEngine

    config = ExperimentConfig(
        model_pn=args.pn,
        cfg=args.cfg,
        tau=args.tau,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        paper_alpha=1.0,
        paper_pfb_feature_index=2,
        paper_sac_prediction_start=2,
        base_style_strength=1.0,
    )
    engine = StyleTransferEngine(bundle, config)
    result = engine.paper_dual_path_generate(
        args.prompt,
        engine.get_style_features(style_image),
        pfb_feature_indices=[2],
        rank=None,
        alpha=1.0,
        style_strength=1.0,
        style_decay=1.0,
        sac_prediction_start=2,
        sac_strength=1.0,
        enable_sac=True,
    )

    output = args.output.resolve()
    save_image_tensor(result["stylized_image_01"].detach().float().cpu(), output)
    print(f"Saved PFB+SAC image: {output}")
    print(
        "Diagnostics:",
        f"SAC calls={result['sac_calls']}",
        f"max Q copy error={result['max_q_copy_error']:.3g}",
        f"max K copy error={result['max_k_copy_error']:.3g}",
    )


if __name__ == "__main__":
    main()
