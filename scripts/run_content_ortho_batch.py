#!/usr/bin/env python3
"""Generate all 190 x 10 content-orthogonal Infinity images, resumably."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFINITY_DIR = PROJECT_ROOT / "Infinity"
SRC_DIR = PROJECT_ROOT / "src"


EXPECTED_CONTENTS = 190
EXPECTED_STYLES = 10


def load_runtime_modules() -> None:
    """Delay heavyweight CUDA imports so --validate-only works anywhere."""
    global torch, F, Image, ImageDraw, ImageFont, ImageOps, to_tensor, tqdm
    global dynamic_resolution_h_w, h_div_w_templates
    global load_tokenizer, load_transformer, load_visual_tokenizer
    global content_orthogonal_feature_blend, encode_prompts

    for import_dir in (str(SRC_DIR), str(INFINITY_DIR), str(INFINITY_DIR / "tools")):
        if import_dir not in sys.path:
            sys.path.insert(0, import_dir)
    import torch as _torch
    import torch.nn.functional as _functional
    from PIL import Image as _image, ImageDraw as _image_draw, ImageFont as _image_font, ImageOps as _image_ops
    from torchvision.transforms.functional import to_tensor as _to_tensor
    from tqdm.auto import tqdm as _tqdm
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w as _resolutions
    from infinity.utils.dynamic_resolution import h_div_w_templates as _aspect_ratios
    from run_infinity import load_tokenizer as _load_tokenizer
    from run_infinity import load_transformer as _load_transformer
    from run_infinity import load_visual_tokenizer as _load_visual_tokenizer
    from var_soict.feature_hypotheses import content_orthogonal_feature_blend as _content_ortho
    from var_soict.stepwise_feature_experiment import encode_prompts as _encode_prompts

    torch, F, Image, ImageDraw, ImageFont, ImageOps, to_tensor = (
        _torch, _functional, _image, _image_draw, _image_font, _image_ops, _to_tensor
    )
    tqdm = _tqdm
    dynamic_resolution_h_w, h_div_w_templates = _resolutions, _aspect_ratios
    load_tokenizer, load_transformer = _load_tokenizer, _load_transformer
    load_visual_tokenizer = _load_visual_tokenizer
    content_orthogonal_feature_blend, encode_prompts = _content_ortho, _encode_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-csv", type=Path, default=PROJECT_ROOT / "prompts/content_prompts_190.csv")
    parser.add_argument("--styles-csv", type=Path, default=PROJECT_ROOT / "styles/quantitative_eval_styles_10.csv")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/content_ortho_step_01")
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "1M"), default="0.25M")
    parser.add_argument("--inject-step", type=int, default=1, help="Zero-based AR scale; 1 means inject after step 1.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--style-rank", type=int, default=1)
    parser.add_argument("--content-rank", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--sac", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Generate at most N missing cases (smoke testing).")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def validate_inputs(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    prompts = read_rows(args.prompts_csv.resolve())
    styles = read_rows(args.styles_csv.resolve())
    if len(prompts) != EXPECTED_CONTENTS:
        raise ValueError(f"Expected {EXPECTED_CONTENTS} prompts, found {len(prompts)} in {args.prompts_csv}")
    if len(styles) != EXPECTED_STYLES:
        raise ValueError(f"Expected {EXPECTED_STYLES} styles, found {len(styles)} in {args.styles_csv}")
    for style in styles:
        path = (PROJECT_ROOT / style["style_reference_image"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        style["_resolved_path"] = str(path)
    if args.inject_step < 0:
        raise ValueError("--inject-step must be non-negative")
    return prompts, styles


def scale_schedule(pn: str) -> list[tuple[int, int, int]]:
    aspect = min(h_div_w_templates, key=lambda value: abs(float(value) - 1.0))
    return [(1, h, w) for _, h, w in dynamic_resolution_h_w[aspect][pn]["scales"]]


class Bundle:
    def __init__(self, tokenizer, encoder, vae, model, schedule, image_size):
        self.device = "cuda"
        self.text_tokenizer = tokenizer
        self.text_encoder = encoder
        self.vae = vae
        self.infinity_model = model
        self.scale_schedule = schedule
        self.image_size = image_size


def load_bundle(args: argparse.Namespace) -> Bundle:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Infinity-2B inference")
    weights = args.weights_dir.resolve()
    required = {
        "model": weights / "infinity_2b_reg.pth",
        "vae": weights / "infinity_vae_d32reg.pth",
        "t5": weights / "flan-t5-xl",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing weights; run scripts/download_infinity_weights.py first:\n" + "\n".join(missing))

    safetensors_index = required["t5"] / "model.safetensors.index.json"
    if safetensors_index.exists():
        index = json.loads(safetensors_index.read_text(encoding="utf-8"))
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        missing_shards = [str(required["t5"] / name) for name in shard_names if not (required["t5"] / name).is_file()]
        if missing_shards:
            raise FileNotFoundError(
                "FLAN-T5 index exists but model shards are missing. Rerun the downloader:\n"
                "python3 scripts/download_infinity_weights.py --output-dir "
                f"{args.weights_dir}\nMissing:\n" + "\n".join(missing_shards)
            )

    runtime_args = argparse.Namespace(
        pn=args.pn, model_path=str(required["model"]), cfg_insertion_layer=0,
        vae_type=32, vae_path=str(required["vae"]), add_lvl_embeding_only_first_block=1,
        use_bit_label=1, model_type="infinity_2b", rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2, use_scale_schedule_embedding=0, sampling_per_bits=1,
        text_encoder_ckpt=str(required["t5"]), text_channels=2048, apply_spatial_patchify=0,
        h_div_w_template=1.0, use_flex_attn=0, cache_dir="/dev/shm",
        checkpoint_type="torch", seed=args.seed, bf16=1, enable_model_cache=0,
    )
    tokenizer, encoder = load_tokenizer(t5_path=runtime_args.text_encoder_ckpt)
    vae = load_visual_tokenizer(runtime_args)
    model = load_transformer(vae, runtime_args)
    if not hasattr(vae.quantizer, "lfq"):
        vae.quantizer.lfq = vae.quantizer.bsq
    image_size = {"0.06M": 256, "0.25M": 512, "1M": 1024}[args.pn]
    return Bundle(tokenizer, encoder, vae, model, scale_schedule(args.pn), image_size)


def style_feature_at_step(bundle: Bundle, image_path: str, step: int) -> torch.Tensor:
    with torch.inference_mode():
        image = ImageOps.fit(
            Image.open(image_path).convert("RGB"),
            (bundle.image_size, bundle.image_size),
            Image.Resampling.LANCZOS,
        )
        image_m11 = to_tensor(image).unsqueeze(0).cuda().mul(2).sub(1)
        with torch.amp.autocast("cuda", enabled=False):
            _, _, _, indices, _, _ = bundle.vae.encode(image_m11.float(), scale_schedule=bundle.scale_schedule)
        final_size = bundle.scale_schedule[-1]
        summed = None
        for index, bit_indices in enumerate(indices[: step + 1]):
            codes = bundle.vae.quantizer.lfq.indices_to_codes(bit_indices, label_type="bit_label")
            if index != len(bundle.scale_schedule) - 1:
                codes = F.interpolate(codes, size=final_size, mode=bundle.vae.quantizer.z_interplote_up)
            summed = codes if summed is None else summed + codes
        return summed.detach().float().cpu()


def infer_kwargs(bundle: Bundle, prompts: list[str], args: argparse.Namespace, *, trace: bool) -> dict:
    count = len(bundle.scale_schedule)
    return dict(
        vae=bundle.vae, scale_schedule=bundle.scale_schedule,
        label_B_or_BLT=encode_prompts(bundle, prompts), B=len(prompts), g_seed=args.seed,
        cfg_list=[args.cfg] * count, tau_list=[args.tau] * count,
        top_k=args.top_k, top_p=args.top_p, returns_vemb=1,
        cfg_insertion_layer=[0], vae_type=1, ret_img=True,
        inference_mode=True, return_feature_trace=trace,
    )


def run_model(method, **kwargs):
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        return method(**kwargs)


def result_image(result, index: int) -> Image.Image:
    return Image.fromarray(result[2][index].detach().cpu().flip(-1).numpy(), mode="RGB")


def save_image_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    image.save(temporary)
    temporary.replace(path)


def save_comparison(
    content_image: Image.Image,
    style_path: str,
    generated_image: Image.Image,
    prompt: str,
    path: Path,
) -> None:
    size = generated_image.size
    style_image = ImageOps.fit(Image.open(style_path).convert("RGB"), size, Image.Resampling.LANCZOS)
    content_image = ImageOps.fit(content_image.convert("RGB"), size, Image.Resampling.LANCZOS)
    font_size = max(24, round(size[0] * 0.055))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    header = max(82, round(font_size * 2.8))
    canvas = Image.new("RGB", (size[0] * 3, size[1] + header), "white")
    draw = ImageDraw.Draw(canvas)
    labels = (f"Prompt: \"{prompt}\"", "Style reference", "Content-ortho result")
    images = (content_image, style_image, generated_image.convert("RGB"))
    for column, (label, image) in enumerate(zip(labels, images)):
        x = column * size[0]
        canvas.paste(image, (x, header))
        max_width = size[0] - 24
        shown = label
        while len(shown) > 4 and draw.textbbox((0, 0), shown, font=font)[2] > max_width:
            shown = shown[:-2]
        if shown != label:
            shown = shown.rstrip() + "…"
        text_box = draw.textbbox((0, 0), shown, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        draw.text(
            (x + (size[0] - text_width) / 2, (header - text_height) / 2 - text_box[1]),
            shown,
            fill="black",
            font=font,
        )
    save_image_atomic(canvas, path)


def case_paths(output_dir: Path, style_id: str, content_id: str) -> tuple[Path, Path, Path]:
    case_dir = output_dir / style_id / content_id
    return case_dir / "generated.png", case_dir / "comparison.png", case_dir / "metadata.json"


def case_complete(output_dir: Path, style_id: str, content_id: str) -> bool:
    return all(path.is_file() for path in case_paths(output_dir, style_id, content_id))


def write_metadata(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    prompts, styles = validate_inputs(args)
    total = len(prompts) * len(styles)
    print(f"Validated {len(prompts)} prompts x {len(styles)} styles = {total} cases")
    if args.validate_only:
        return
    load_runtime_modules()
    bundle = load_bundle(args)
    if args.inject_step >= len(bundle.scale_schedule):
        raise ValueError(f"inject step {args.inject_step} outside {len(bundle.scale_schedule)}-step schedule")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    style_features = {}
    for style in tqdm(styles, desc="Encoding style references", unit="style"):
        style_features[style["style_id"]] = style_feature_at_step(
            bundle, style["_resolved_path"], args.inject_step
        )

    existing = 0 if args.overwrite else sum(
        case_complete(output_dir, style["style_id"], content["content_id"])
        for content in prompts
        for style in styles
    )
    generated, skipped, failed = 0, existing, 0
    started = time.time()
    image_progress = tqdm(total=total, initial=existing, desc="Generating images", unit="image")
    content_progress = tqdm(prompts, desc="Content prompts", unit="prompt")
    for content in content_progress:
        missing_styles = []
        for style in styles:
            if args.overwrite or not case_complete(output_dir, style["style_id"], content["content_id"]):
                missing_styles.append(style)
        if not missing_styles:
            continue

        prompt = content["content_prompt"].strip()
        content_progress.set_postfix(content=content["content_id"], missing=len(missing_styles))
        content_run = run_model(
            bundle.infinity_model.autoregressive_infer_cfg,
            **infer_kwargs(bundle, [prompt], args, trace=True),
        )
        content_trace = content_run[3]
        content_at_step = content_trace[args.inject_step]
        content_image = result_image(content_run, 0)
        for style in missing_styles:
            if args.limit is not None and generated >= args.limit:
                image_progress.close()
                tqdm.write(f"Stopped after --limit={args.limit}; rerun without it to resume.")
                return
            image_path, comparison_path, metadata_path = case_paths(
                output_dir, style["style_id"], content["content_id"]
            )
            try:
                edited = content_orthogonal_feature_blend(
                    content_at_step,
                    style_features[style["style_id"]].to(content_at_step),
                    content_at_step,
                    style_rank=args.style_rank, content_rank=args.content_rank,
                    alpha=args.alpha, strength=args.strength,
                    projection_strength=args.projection_strength, preserve_mean=True,
                )
                result = run_model(
                    bundle.infinity_model.autoregressive_infer_content_ortho,
                    **infer_kwargs(bundle, [prompt, prompt], args, trace=False),
                    content_ortho_feature=edited, inject_step=args.inject_step,
                    f_con=content_trace, sac=args.sac,
                )
                generated_image = result_image(result, 1)
                save_image_atomic(generated_image, image_path)
                save_comparison(
                    content_image, style["_resolved_path"], generated_image, prompt, comparison_path
                )
                write_metadata(metadata_path, {
                    "content_id": content["content_id"], "content_prompt": prompt,
                    "style_id": style["style_id"], "style_name": style["style_name"],
                    "style_reference_image": style["style_reference_image"],
                    "output_image": str(image_path), "comparison_image": str(comparison_path),
                    "seed": args.seed, "pn": args.pn,
                    "inject_step": args.inject_step, "inject_step_numbering": "zero_based",
                    "method": "content_ortho", "sac": args.sac,
                })
                generated += 1
                image_progress.update(1)
                image_progress.set_postfix(
                    content=content["content_id"], style=style["style_id"], failed=failed
                )
                del edited, result, generated_image
            except Exception as exc:
                failed += 1
                image_progress.close()
                tqdm.write(f"FAILED {content['content_id']} x {style['style_id']}: {exc}", file=sys.stderr)
                raise
        del content_run, content_trace, content_at_step, content_image
        gc.collect()
        torch.cuda.empty_cache()

    image_progress.close()
    elapsed = time.time() - started
    print(f"Complete: generated={generated}, skipped={skipped}, failed={failed}, seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
