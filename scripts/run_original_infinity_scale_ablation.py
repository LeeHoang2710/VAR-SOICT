#!/usr/bin/env python3
"""Run Notebook 16 Study 1/2 with original Infinity-2B weights.

This script intentionally avoids the GGUF path.  It expects a weights directory
containing:

  - infinity_2b_reg.pth
  - infinity_vae_d32reg.pth
  - flan-t5-xl/

Outputs are written under ``outputs/original_infinity_notebook16_ablation`` by
default: per-sample numerical CSVs, aggregate summaries, and plots.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
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
from PIL import Image, ImageDraw, ImageOps
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


OBJECTS = [
    "cat", "fox", "robot", "spaceship", "mushroom", "teapot", "violin", "camera", "lantern", "penguin",
    "castle", "dragon", "sailboat", "train", "butterfly", "flower", "owl", "bicycle", "chair", "lamp",
    "rocket", "turtle", "parrot", "watch", "piano", "watermelon", "umbrella", "balloon", "horse", "seashell",
    "snail", "duck", "bear", "notebook", "paintbrush", "saxophone", "crown", "compass", "microphone", "orchid",
    "frisbee", "leopard", "glass", "lollipop", "hammer", "kangaroo", "ladybug", "moose", "ninja", "seagull",
]

SUPERCLASSES = {
    "cat": "animals", "fox": "animals", "robot": "toys", "spaceship": "vehicles", "mushroom": "plants",
    "teapot": "objects", "violin": "musical instruments", "camera": "objects", "lantern": "objects",
    "penguin": "animals", "castle": "buildings", "dragon": "fantasy creatures", "sailboat": "vehicles",
    "train": "vehicles", "butterfly": "animals", "flower": "plants", "owl": "animals", "bicycle": "vehicles",
    "chair": "furniture", "lamp": "objects", "rocket": "vehicles", "turtle": "animals", "parrot": "animals",
    "watch": "objects", "piano": "musical instruments", "watermelon": "food", "umbrella": "objects",
    "balloon": "objects", "horse": "animals", "seashell": "natural objects", "snail": "animals",
    "duck": "animals", "bear": "animals", "notebook": "objects", "paintbrush": "objects",
    "saxophone": "musical instruments", "crown": "objects", "compass": "objects", "microphone": "objects",
    "orchid": "plants", "frisbee": "objects", "leopard": "animals", "glass": "objects", "lollipop": "food",
    "hammer": "tools", "kangaroo": "animals", "ladybug": "animals", "moose": "animals", "ninja": "characters",
    "seagull": "animals",
}

STYLES = [
    "watercolor painting", "glowing neon", "retro comic book", "origami paper craft", "blueprint drawing",
    "minimal pastel illustration", "mosaic tile art", "pixel art", "woodcut print", "chalk drawing",
    "digital glitch art", "cubist painting", "graffiti mural", "papercut collage", "sticker illustration",
    "impressionist painting", "surrealist painting", "flat vector illustration", "medieval fantasy illustration",
    "rainbow flowing smoke wave",
]

SETTINGS = [
    "centered composition", "studio lighting", "dark cinematic background", "simple clean background",
    "floating in space", "on a wooden table", "in a quiet forest", "beside a reflective lake",
    "under warm sunset light", "on a white museum pedestal",
]


@dataclass(frozen=True)
class OutputDirs:
    root: Path
    study1: Path
    study1_traces: Path
    study1_final: Path
    study1_grids: Path
    study1_plots: Path
    study2: Path
    study2_originals: Path
    study2_grids: Path
    study2_plots: Path


def build_output_dirs(output_dir: Path) -> OutputDirs:
    dirs = OutputDirs(
        root=output_dir,
        study1=output_dir / "study1_step_dynamics",
        study1_traces=output_dir / "study1_step_dynamics" / "traces",
        study1_final=output_dir / "study1_step_dynamics" / "final_images",
        study1_grids=output_dir / "study1_step_dynamics" / "step_grids",
        study1_plots=output_dir / "study1_step_dynamics" / "plots",
        study2=output_dir / "study2_scale_removal",
        study2_originals=output_dir / "study2_scale_removal" / "originals",
        study2_grids=output_dir / "study2_scale_removal" / "reconstruction_grids",
        study2_plots=output_dir / "study2_scale_removal" / "plots",
    )
    for directory in dirs.__dict__.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--infinity-dir", type=Path, default=PROJECT_ROOT / "Infinity")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--csd100-dir", type=Path, default=Path("csd100"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/original_infinity_notebook16_ablation"))
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pn", type=str, default="0.25M", choices=["0.06M", "0.25M", "1M"])
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=600)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--vae-type", type=int, default=32)
    parser.add_argument("--model-type", type=str, default="infinity_2b")
    parser.add_argument("--text-channels", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-study1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-study2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-grids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csd-model-id", type=str, default=os.environ.get("CSD_EVAL_MODEL_ID", ""))
    parser.add_argument("--package-zip", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_infinity_dir(workspace_dir: Path, explicit: Path | None) -> Path:
    candidates = [
        explicit,
        workspace_dir / "Infinity",
        workspace_dir.parent / "Infinity",
        Path.cwd() / "Infinity",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "tools" / "run_infinity.py").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find official Infinity source. Pass --infinity-dir pointing to a checkout "
        "that contains tools/run_infinity.py."
    )


def resolve_weights_dir(workspace_dir: Path, weights_dir: Path) -> Path:
    weights_dir = weights_dir if weights_dir.is_absolute() else workspace_dir / weights_dir
    required = ["infinity_2b_reg.pth", "infinity_vae_d32reg.pth", "flan-t5-xl"]
    missing = [name for name in required if not (weights_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} under weights directory: {weights_dir}")
    return weights_dir.resolve()


def import_original_infinity(infinity_dir: Path):
    sys.path.insert(0, str(infinity_dir))
    sys.path.insert(0, str(infinity_dir / "tools"))
    from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
    from run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer

    return SimpleNamespace(
        sample_with_top_k_top_p=sample_with_top_k_top_p_also_inplace_modifying_logits_,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
        h_div_w_templates=h_div_w_templates,
        load_tokenizer=load_tokenizer,
        load_transformer=load_transformer,
        load_visual_tokenizer=load_visual_tokenizer,
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
    print("[1/3] Loading original FLAN-T5-XL tokenizer/encoder:", run_args.text_encoder_ckpt)
    text_tokenizer, text_encoder = api.load_tokenizer(t5_path=run_args.text_encoder_ckpt)
    print("[2/3] Loading original Infinity VAE:", run_args.vae_path)
    vae = api.load_visual_tokenizer(run_args).eval()
    if not hasattr(vae.quantizer, "lfq"):
        vae.quantizer.lfq = vae.quantizer.bsq
    print("[3/3] Loading original Infinity-2B transformer:", run_args.model_path)
    infinity = api.load_transformer(vae, run_args).eval()
    infinity.requires_grad_(False)
    return text_tokenizer, text_encoder, vae, infinity


def build_scale_schedule(args: argparse.Namespace, api: SimpleNamespace):
    h_div_w_template = api.h_div_w_templates[np.argmin(np.abs(api.h_div_w_templates - 1.0))]
    raw = api.dynamic_resolution_h_w[h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for _, h, w in raw]
    image_size_hw = (1024, 1024) if args.pn == "1M" else (512, 512)
    print("Scale schedule:", scale_schedule)
    print("Image size:", image_size_hw)
    return scale_schedule, image_size_hw


def tensor_to_pil(image) -> Image.Image:
    if isinstance(image, (list, tuple)):
        image = image[0]
    tensor = image.detach().float().cpu() if torch.is_tensor(image) else torch.as_tensor(image).float()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Unexpected image shape: {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor.permute(1, 2, 0)
    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)
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


def save_image(image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image).save(path)
    return path


def build_prompt_rows(seed: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for obj in OBJECTS:
        style = rng.choice(STYLES)
        setting = rng.choice(SETTINGS)
        superclass = SUPERCLASSES.get(obj, "objects")
        prompt = f"A {obj}, {superclass}, in {style} style, {setting}"
        rows.append(
            {
                "case_id": len(rows),
                "object": obj,
                "superclass": superclass,
                "style": style,
                "setting": setting,
                "prompt": prompt,
            }
        )
    return rows[:count]


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


@torch.no_grad()
def generate_text_trace(
    *,
    prompt: str,
    seed: int,
    args: argparse.Namespace,
    api: SimpleNamespace,
    text_tokenizer,
    text_encoder,
    vae,
    infinity,
    scale_schedule,
):
    device = args.device
    rng = torch.Generator(device=device).manual_seed(int(seed))

    kv_compact, lens, cu_seqlens_k, max_seqlen_k = encode_prompts(text_tokenizer, text_encoder, [prompt], device)
    kv_compact_un = kv_compact.clone()
    kv_compact_un[: lens[0]] = infinity.cfg_uncond[: lens[0]]
    kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
    cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
    batch_size = 2

    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_bd = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = infinity.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(batch_size, 1, -1) + infinity.pos_start.expand(batch_size, 1, -1)

    with torch.amp.autocast("cuda", enabled=False):
        cond_bd_or_gss = infinity.shared_ada_lin(cond_bd.float()).float().contiguous()

    final_size = scale_schedule[-1]
    summed_codes = last_stage.new_zeros(1, infinity.d_vae, *final_size)
    residuals, cumulative_trace = [], []

    for block in infinity.unregistered_blocks:
        self_attention_module(block).kv_caching(True)

    try:
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            for step_id, pn in enumerate(scale_schedule):
                cur_len = sum(np.array(x).prod() for x in scale_schedule[: step_id + 1])
                need_to_pad = 0
                attn_fn = None
                if infinity.use_flex_attn:
                    attn_fn = infinity.attn_fn_compile_dict.get(tuple(scale_schedule[: step_id + 1]), None)

                for block_idx, block_chunk in enumerate(infinity.block_chunks):
                    if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
                        last_stage = infinity.add_lvl_embeding(last_stage, step_id, scale_schedule, need_to_pad=need_to_pad)
                    if not infinity.add_lvl_embeding_only_first_block:
                        last_stage = infinity.add_lvl_embeding(last_stage, step_id, scale_schedule, need_to_pad=need_to_pad)

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
                logits = float(args.cfg) * logits[:1] + (1 - float(args.cfg)) * logits[1:]
                logits = logits.reshape(logits.shape[0], -1, 2).clone()
                sampled = api.sample_with_top_k_top_p(
                    logits,
                    rng=rng,
                    top_k=args.top_k or infinity.top_k,
                    top_p=args.top_p or infinity.top_p,
                    num_samples=1,
                )[:, :, 0]
                idx = sampled.reshape(logits.shape[0], cur_len - sum(np.array(x).prod() for x in scale_schedule[:step_id]), -1)
                codes = bit_labels_to_codes(vae, idx, pn)
                if step_id != len(scale_schedule) - 1:
                    codes = F.interpolate(codes, size=final_size, mode=vae.quantizer.z_interplote_up)

                residuals.append(codes.detach().float().cpu())
                summed_codes = summed_codes + codes
                cumulative_trace.append(summed_codes.detach().float().cpu())

                if step_id != len(scale_schedule) - 1:
                    next_raw = next_raw_from_summed_codes(vae, infinity, summed_codes, scale_schedule[step_id + 1])
                    last_stage = infinity.word_embed(infinity.norm0_ve(next_raw)).repeat(batch_size, 1, 1)

        final_image_01 = decode_summed_codes_to_image_01(vae, summed_codes, device)
        return {
            "prompt": prompt,
            "seed": int(seed),
            "final_image_01": final_image_01.detach().float().cpu(),
            "trace": cumulative_trace,
            "residuals": residuals,
        }
    finally:
        for block in infinity.unregistered_blocks:
            self_attention_module(block).kv_caching(False)


class MetricBackbones:
    def __init__(self, device: str, csd_model_id: str = ""):
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        from torchvision.models import ResNet50_Weights, resnet50, VGG19_Weights, vgg19

        self.device = device
        self.AutoImageProcessor = AutoImageProcessor
        self.AutoModel = AutoModel
        self.AutoProcessor = AutoProcessor
        self.metric_tf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.vgg.requires_grad_(False)

        try:
            self.style_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
            self.style_backbone_name = "DINOv2 ViT-S/14"
        except Exception as exc:
            print("DINOv2 load failed; falling back to ResNet50:", repr(exc))
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
            self.style_backbone = torch.nn.Sequential(*(list(resnet.children())[:-1])).to(device).eval()
            self.style_backbone_name = "ResNet50 fallback"
        self.style_backbone.requires_grad_(False)

        candidates = [csd_model_id.strip(), "bigshanedogg/CSD", "tomg-group-umd/CSD-ViT-L"]
        self.csd_candidates = [name for i, name in enumerate(candidates) if name and name not in candidates[:i]]
        self.csd_processor = None
        self.csd_model = None
        self.csd_model_id = None
        self.csd_style_cache = {}
        self.csd_content_cache = {}

    def image_for_metrics(self, image):
        if torch.is_tensor(image):
            return tensor_to_pil(image)
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def metric_input(self, image):
        return self.metric_tf(self.image_for_metrics(image)).unsqueeze(0).to(self.device)

    @staticmethod
    def cosine01(a, b) -> float:
        a = a.float()
        b = b.float()
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(f"Cosine embedding dimension mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        return float((a * b).sum(dim=-1).clamp(-1, 1).detach().cpu().item())

    @torch.no_grad()
    def vgg_content_embedding(self, image):
        x = self.metric_input(image)
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i == 35:
                break
        x = F.adaptive_avg_pool2d(x.float(), output_size=1).flatten(1)
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def dino_style_embedding(self, image):
        x = self.metric_input(image)
        feat = self.style_backbone(x)
        if isinstance(feat, dict):
            value = feat.get("x_norm_clstoken")
            feat = value if value is not None else feat.get("x_prenorm")
            if feat is None:
                raise KeyError("Could not find a usable DINO feature tensor.")
        return F.normalize(feat.float().flatten(1), dim=-1)

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
                self.csd_model_id = model_id
                print("Loaded CSD style encoder:", model_id)
                return processor, model
            except Exception as exc:
                last_error = exc
                print(f"CSD load failed for {model_id}: {repr(exc)}")
        raise RuntimeError("Could not load a real CSD style encoder.") from last_error

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
    def csd_embeddings(self, image, cache_key=None):
        key = str(cache_key) if cache_key is not None else None
        if key is not None and key in self.csd_style_cache:
            return self.csd_style_cache[key], self.csd_content_cache[key]

        processor, model = self.get_csd_model()
        pil = self.image_for_metrics(image)
        inputs = processor(images=pil, return_tensors="pt")
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
        content = self.extract_named_tensor(
            outputs,
            ("content_embeddings", "content_embeds", "content_embedding", "image_content_embeds"),
        )
        if style is None:
            style = self.fallback_output_tensor(outputs)
        if content is None:
            content = style

        style = F.normalize(style.float(), dim=-1).detach().cpu()
        content = F.normalize(content.float(), dim=-1).detach().cpu()
        if key is not None:
            self.csd_style_cache[key] = style
            self.csd_content_cache[key] = content
        return style, content

    def rgb_chi_square_distance(self, image_a, image_b, bins=32, eps=1e-8) -> float:
        a = np.asarray(self.image_for_metrics(image_a).resize((224, 224)), dtype=np.float32) / 255.0
        b = np.asarray(self.image_for_metrics(image_b).resize((224, 224)), dtype=np.float32) / 255.0
        distances = []
        for channel in range(3):
            hist_a, _ = np.histogram(a[..., channel], bins=bins, range=(0, 1), density=False)
            hist_b, _ = np.histogram(b[..., channel], bins=bins, range=(0, 1), density=False)
            hist_a = hist_a.astype(np.float64) / max(hist_a.sum(), 1)
            hist_b = hist_b.astype(np.float64) / max(hist_b.sum(), 1)
            distances.append(0.5 * np.sum(((hist_a - hist_b) ** 2) / (hist_a + hist_b + eps)))
        return float(np.mean(distances))


def make_step_grid(images, title: str, path: Path, columns: int | None = None, thumb: int = 160):
    columns = columns or len(images)
    label_h = 28
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * thumb, rows * (thumb + label_h) + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title[:180], fill=(0, 0, 0))
    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        x = col * thumb
        y = 34 + row * (thumb + label_h)
        canvas.paste(image.resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 6, y + thumb + 6), f"scale {idx + 1}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


@torch.no_grad()
def decode_trace_payload(payload, vae, device: str):
    images = []
    for summed in payload["trace"]:
        image_01 = decode_summed_codes_to_image_01(vae, summed.to(device), device)
        images.append(tensor_to_pil(image_01))
        del image_01
    return images


def run_study1(
    *,
    args: argparse.Namespace,
    dirs: OutputDirs,
    api: SimpleNamespace,
    text_tokenizer,
    text_encoder,
    vae,
    infinity,
    scale_schedule,
    metrics: MetricBackbones,
):
    prompt_rows = build_prompt_rows(args.seed, args.num_samples)
    manifest_path = dirs.study1 / "study1_prompts_50.csv"
    pd.DataFrame(prompt_rows).to_csv(manifest_path, index=False)

    def trace_path(case_id):
        return dirs.study1_traces / f"case_{int(case_id):03d}_trace.pt"

    def final_path(case_id):
        return dirs.study1_final / f"case_{int(case_id):03d}.png"

    for row in tqdm(prompt_rows, desc="Study 1: generating traced images"):
        case_id = int(row["case_id"])
        if trace_path(case_id).exists() and final_path(case_id).exists() and not args.force:
            continue
        result = generate_text_trace(
            prompt=row["prompt"],
            seed=args.seed + case_id,
            args=args,
            api=api,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            infinity=infinity,
            scale_schedule=scale_schedule,
        )
        save_image(result["final_image_01"], final_path(case_id))
        torch.save(
            {
                "case_id": case_id,
                "prompt": row["prompt"],
                "seed": args.seed + case_id,
                "scale_schedule": scale_schedule,
                "trace": result["trace"],
                "residuals": result["residuals"],
            },
            trace_path(case_id),
        )
        del result
        gc.collect()
        torch.cuda.empty_cache()

    metric_rows = []
    for row in tqdm(prompt_rows, desc="Study 1: computing metrics"):
        case_id = int(row["case_id"])
        payload = torch.load(trace_path(case_id), map_location="cpu")
        step_images = decode_trace_payload(payload, vae, args.device)
        final_image = step_images[-1]
        if args.save_grids:
            make_step_grid(step_images, f"case {case_id:03d}: {row['prompt']}", dirs.study1_grids / f"case_{case_id:03d}_steps.png")

        final_content = metrics.vgg_content_embedding(final_image)
        final_style = metrics.dino_style_embedding(final_image)
        for step_idx, step_image in enumerate(step_images, start=1):
            metric_rows.append(
                {
                    "case_id": case_id,
                    "prompt": row["prompt"],
                    "step": step_idx,
                    "rgb_chi_square": metrics.rgb_chi_square_distance(step_image, final_image),
                    "content_similarity": metrics.cosine01(metrics.vgg_content_embedding(step_image), final_content),
                    "style_similarity": metrics.cosine01(metrics.dino_style_embedding(step_image), final_style),
                }
            )
        del payload, step_images, final_content, final_style
        gc.collect()
        torch.cuda.empty_cache()

    metrics_path = dirs.study1 / "study1_stepwise_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    plot_study1(dirs, metrics.style_backbone_name)
    print("Saved Study 1 metrics:", metrics_path)


def plot_study1(dirs: OutputDirs, style_backbone_name: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics_path = dirs.study1 / "study1_stepwise_metrics.csv"
    df = pd.read_csv(metrics_path)
    summary = df.groupby("step", as_index=False).agg(
        rgb_chi_square_mean=("rgb_chi_square", "mean"),
        rgb_chi_square_std=("rgb_chi_square", "std"),
        content_similarity_mean=("content_similarity", "mean"),
        content_similarity_std=("content_similarity", "std"),
        style_similarity_mean=("style_similarity", "mean"),
        style_similarity_std=("style_similarity", "std"),
    )
    summary_path = dirs.study1 / "study1_stepwise_summary.csv"
    summary.to_csv(summary_path, index=False)

    sns.set_theme(style="whitegrid", context="paper")
    plot_specs = [
        ("rgb_chi_square", "RGB histogram distance to final", "#ff7f0e"),
        ("content_similarity", "VGG content similarity to final", "#bcbd22"),
        ("style_similarity", f"{style_backbone_name} style similarity to final", "#556b2f"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (column, ylabel, color) in zip(axes, plot_specs):
        for _, group in df.groupby("case_id"):
            ax.plot(group["step"], group[column], color=color, alpha=0.12, linewidth=0.8)
        mean = df.groupby("step")[column].mean()
        std = df.groupby("step")[column].std()
        steps = mean.index.to_numpy()
        ax.plot(steps, mean.to_numpy(), color=color, linewidth=2.8, marker="o", label="mean over 50 samples")
        ax.fill_between(steps, mean.to_numpy() - std.to_numpy(), mean.to_numpy() + std.to_numpy(), color=color, alpha=0.16, linewidth=0)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Generation scale / step")
    fig.suptitle("Study 1: step-wise baseline generation dynamics", y=1.0, fontsize=14, fontweight="bold")
    fig.tight_layout()
    line_plot = dirs.study1_plots / "study1_stepwise_dynamics_50_lines.png"
    fig.savefig(line_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax2 = ax1.twinx()
    ax1.plot(summary["step"], summary["rgb_chi_square_mean"] * 1e4, color="#ff7f0e", marker="*", linewidth=2.4, markersize=9, label="RGB statistics (x1e4)")
    ax2.plot(summary["step"], summary["content_similarity_mean"], color="#bcbd22", marker="o", linewidth=2.4, label="Content similarity")
    ax2.plot(summary["step"], summary["style_similarity_mean"], color="#556b2f", marker="D", linewidth=2.4, linestyle=":", label="Style similarity")
    ax1.set_xlabel("Number of steps / scale")
    ax1.set_ylabel("RGB histogram distance x 1e4", color="#ff7f0e")
    ax2.set_ylabel("Similarity to final image")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    mean_plot = dirs.study1_plots / "study1_stepwise_dynamics_mean_curves.png"
    fig.savefig(mean_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved Study 1 plots:", line_plot, mean_plot)
    print("Saved Study 1 summary:", summary_path)


def load_image_m11(path: Path, size: int, device: str):
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    tensor_01 = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).to(device)
    return tensor_01.mul(2).sub(1), tensor_01, image


@torch.no_grad()
def encode_image_residuals(vae, image_m11, scale_schedule, device: str):
    with torch.amp.autocast("cuda", enabled=False):
        _, _, _, all_bit_indices, _, _ = vae.encode(image_m11.float(), scale_schedule=scale_schedule)
    residuals = []
    final_size = scale_schedule[-1]
    for step_id, bit_indices in enumerate(all_bit_indices):
        codes = vae.quantizer.lfq.indices_to_codes(bit_indices, label_type="bit_label")
        if step_id != len(scale_schedule) - 1:
            codes = F.interpolate(codes, size=final_size, mode=vae.quantizer.z_interplote_up)
        residuals.append(codes.detach().float().cpu())
    return residuals


@torch.no_grad()
def reconstruct_from_residuals(vae, residuals, device: str, remove_index=None):
    kept = [residual.to(device) for i, residual in enumerate(residuals) if remove_index is None or i != remove_index]
    if not kept:
        raise ValueError("At least one residual scale must remain for reconstruction.")
    summed = torch.stack(kept, dim=0).sum(dim=0)
    image_01 = decode_summed_codes_to_image_01(vae, summed, device)
    return tensor_to_pil(image_01)


def make_reconstruction_grid(original_image, recon_images, title: str, path: Path):
    labels = ["original"] + [f"-scale {i + 1}" for i in range(len(recon_images))]
    images = [original_image] + recon_images
    thumb = 128
    label_h = 24
    cols = min(6, len(images))
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * thumb, rows * (thumb + label_h) + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title[:160], fill=(0, 0, 0))
    for idx, (image, label) in enumerate(zip(images, labels)):
        row, col = divmod(idx, cols)
        x = col * thumb
        y = 34 + row * (thumb + label_h)
        canvas.paste(image.resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 5, y + thumb + 5), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def resolve_csd100_dir(workspace_dir: Path, explicit: Path | None) -> Path:
    candidates = [
        explicit,
        workspace_dir / "csd100",
        workspace_dir.parent / "VAR_SOICT" / "csd100",
        Path("/content/VAR_Style_Transfer_Workspace/csd100"),
        Path("/content/VAR_SOICT/csd100"),
    ]
    for candidate in candidates:
        if candidate is not None and len(list(candidate.glob("*/00.jpg"))) >= 1:
            return candidate.resolve()
    raise FileNotFoundError("Could not find CSD100. Pass --csd100-dir containing */00.jpg samples.")


def run_study2(
    *,
    args: argparse.Namespace,
    dirs: OutputDirs,
    vae,
    scale_schedule,
    image_size_hw,
    metrics: MetricBackbones,
):
    csd100_dir = resolve_csd100_dir(args.workspace_dir, args.csd100_dir)
    image_paths = sorted(csd100_dir.glob("*/00.jpg"))
    if len(image_paths) < args.num_samples:
        raise FileNotFoundError(f"Need at least {args.num_samples} CSD100 images, found {len(image_paths)} in {csd100_dir}")
    rng = random.Random(args.seed)
    sampled_paths = rng.sample(image_paths, args.num_samples)
    manifest = pd.DataFrame(
        [{"case_id": i, "image_path": str(path), "pair_id": path.parent.name} for i, path in enumerate(sampled_paths)]
    )
    manifest_path = dirs.study2 / "study2_csd100_sample_50.csv"
    manifest.to_csv(manifest_path, index=False)

    metric_rows = []
    for row in tqdm(manifest.to_dict("records"), desc="Study 2: CSD100 scale removal"):
        case_id = int(row["case_id"])
        pair_id = row["pair_id"]
        safe_pair_id = pair_id.replace("/", "_")
        original_m11, _, original_image = load_image_m11(Path(row["image_path"]), image_size_hw[0], args.device)
        original_save_path = dirs.study2_originals / f"case_{case_id:03d}_{safe_pair_id}.png"
        if args.force or not original_save_path.exists():
            original_image.save(original_save_path)

        residuals = encode_image_residuals(vae, original_m11, scale_schedule, args.device)
        original_dino = metrics.dino_style_embedding(original_image)
        original_csd_style, _ = metrics.csd_embeddings(original_image, cache_key=f"study2_original_{case_id}")
        original_content = metrics.vgg_content_embedding(original_image)

        recon_images = []
        for remove_index in range(len(residuals)):
            recon_image = reconstruct_from_residuals(vae, residuals, args.device, remove_index=remove_index)
            recon_images.append(recon_image)
            recon_csd_style, _ = metrics.csd_embeddings(recon_image, cache_key=f"study2_case_{case_id}_remove_{remove_index + 1}")
            metric_rows.append(
                {
                    "case_id": case_id,
                    "pair_id": pair_id,
                    "image_path": row["image_path"],
                    "removed_scale": remove_index + 1,
                    "dino_similarity": metrics.cosine01(metrics.dino_style_embedding(recon_image), original_dino),
                    "csd_s_similarity": metrics.cosine01(recon_csd_style, original_csd_style),
                    "content_similarity": metrics.cosine01(metrics.vgg_content_embedding(recon_image), original_content),
                }
            )

        if args.save_grids:
            make_reconstruction_grid(
                original_image,
                recon_images,
                f"case {case_id:03d}: {pair_id}",
                dirs.study2_grids / f"case_{case_id:03d}_{safe_pair_id}_remove_each_scale.png",
            )
        del residuals, recon_images, original_m11, original_dino, original_csd_style, original_content
        gc.collect()
        torch.cuda.empty_cache()

    metrics_path = dirs.study2 / "study2_scale_removal_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    plot_study2(dirs)
    print("Saved Study 2 manifest:", manifest_path)
    print("Saved Study 2 metrics:", metrics_path)


def plot_study2(dirs: OutputDirs):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    metrics_path = dirs.study2 / "study2_scale_removal_metrics.csv"
    df = pd.read_csv(metrics_path)
    summary = df.groupby("removed_scale", as_index=False).agg(
        dino_mean=("dino_similarity", "mean"),
        dino_std=("dino_similarity", "std"),
        csd_s_mean=("csd_s_similarity", "mean"),
        csd_s_std=("csd_s_similarity", "std"),
        content_mean=("content_similarity", "mean"),
        content_std=("content_similarity", "std"),
    )
    summary_path = dirs.study2 / "study2_scale_removal_summary.csv"
    summary.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for column, std_column, label, color, marker in [
        ("dino_mean", "dino_std", "DINO", "#ff6666", "o"),
        ("csd_s_mean", "csd_s_std", "CSD-S", "#4ecdc4", "s"),
        ("content_mean", "content_std", "VGG content", "#8d6eec", "^"),
    ]:
        x = summary["removed_scale"].to_numpy()
        y = summary[column].to_numpy()
        ystd = summary[std_column].fillna(0).to_numpy()
        ax.plot(x, y, marker=marker, linewidth=2.4, markersize=7, label=label, color=color)
        ax.fill_between(x, y - ystd, y + ystd, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("Removed scale")
    ax.set_ylabel("Similarity to original image")
    ax.set_title("Study 2: CSD100 scale-removal reconstruction sensitivity", fontweight="bold")
    ax.set_xticks(summary["removed_scale"])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    all_plot = dirs.study2_plots / "study2_scale_removal_similarity.png"
    fig.savefig(all_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(summary["removed_scale"], summary["dino_mean"], marker="o", linewidth=2.4, markersize=7, label="DINO", color="#ff6666")
    ax.plot(summary["removed_scale"], summary["csd_s_mean"], marker="s", linewidth=2.4, markersize=7, label="CSD-S", color="#4ecdc4")
    ax.set_xlabel("Removing Scale")
    ax.set_ylabel("Score")
    ax.set_title("Analysis of style-related scores across different scales", fontweight="bold")
    ax.set_xticks(summary["removed_scale"])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    style_plot = dirs.study2_plots / "study2_scale_removal_style_scores.png"
    fig.savefig(style_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved Study 2 plots:", all_plot, style_plot)
    print("Saved Study 2 summary:", summary_path)


def package_outputs(dirs: OutputDirs):
    package_path = dirs.root / "original_infinity_notebook16_study1_study2_outputs.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in dirs.root.rglob("*"):
            if path.is_file() and path != package_path:
                zf.write(path, path.relative_to(dirs.root))
    print("Saved package:", package_path)


def main():
    args = parse_args()
    args.workspace_dir = args.workspace_dir.resolve()
    args.output_dir = (args.output_dir if args.output_dir.is_absolute() else args.workspace_dir / args.output_dir).resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for practical original Infinity-2B inference.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dirs = build_output_dirs(args.output_dir)
    infinity_dir = resolve_infinity_dir(args.workspace_dir, args.infinity_dir)
    weights_dir = resolve_weights_dir(args.workspace_dir, args.weights_dir)
    print("Workspace:", args.workspace_dir)
    print("Infinity source:", infinity_dir)
    print("Weights:", weights_dir)
    print("Outputs:", dirs.root)

    api = import_original_infinity(infinity_dir)
    text_tokenizer, text_encoder, vae, infinity = load_original_models(args, weights_dir, api)
    scale_schedule, image_size_hw = build_scale_schedule(args, api)
    metrics = MetricBackbones(args.device, csd_model_id=args.csd_model_id)

    run_config_path = dirs.root / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as stream:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, stream, indent=2)

    if args.run_study1:
        run_study1(
            args=args,
            dirs=dirs,
            api=api,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            infinity=infinity,
            scale_schedule=scale_schedule,
            metrics=metrics,
        )
    if args.run_study2:
        run_study2(args=args, dirs=dirs, vae=vae, scale_schedule=scale_schedule, image_size_hw=image_size_hw, metrics=metrics)
    if args.package_zip:
        package_outputs(dirs)


if __name__ == "__main__":
    main()
