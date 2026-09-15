#!/usr/bin/env python3
"""Download the official Infinity-2B inference assets from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path


INFINITY_REPO = "FoundationVision/Infinity"
FILES = (
    "infinity_2b_reg.pth",
    "infinity_vae_d32reg.pth",
    "infinity_vae_d64.pth",
)
T5_REPO = "google/flan-t5-xl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token (normally HF_TOKEN is enough).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Infinity assets to {output_dir}")
    snapshot_download(
        repo_id=INFINITY_REPO,
        allow_patterns=list(FILES),
        local_dir=output_dir,
        token=args.token,
    )

    t5_dir = output_dir / "flan-t5-xl"
    print(f"Downloading FLAN-T5-XL tokenizer and encoder to {t5_dir}")
    snapshot_download(
        repo_id=T5_REPO,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "model-*.safetensors",
            "pytorch_model.bin",
            "pytorch_model-*.bin",
            "*.index.json",
            "spiece.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        local_dir=t5_dir,
        token=args.token,
    )
    print("Download complete.")
    print("Batch inference uses infinity_vae_d32reg.pth with infinity_2b_reg.pth.")
    print("infinity_vae_d64.pth is downloaded as requested, but is not shape-compatible with that 2B checkpoint.")


if __name__ == "__main__":
    main()
