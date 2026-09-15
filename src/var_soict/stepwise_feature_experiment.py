"""Step-wise experiment driven exclusively by methods on ``Infinity``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from .feature_hypotheses import content_orthogonal_feature_blend, principal_feature_blend


def encode_prompts(bundle, prompts):
    tokens = bundle.text_tokenizer(text=list(prompts), max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tokens.input_ids.to(bundle.device, non_blocking=True)
    mask = tokens.attention_mask.to(bundle.device, non_blocking=True)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=False):
        features = bundle.text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"].float()
    lengths = mask.sum(-1).tolist()
    cu_seqlens = F.pad(mask.sum(-1).to(torch.int32).cumsum(0), (1, 0))
    compact = torch.cat([item[:length] for length, item in zip(lengths, features.unbind(0))])
    return compact, lengths, cu_seqlens, max(lengths)


def _infer_kwargs(bundle, config, prompts, seed):
    count = len(bundle.scale_schedule)
    return dict(
        vae=bundle.vae, scale_schedule=bundle.scale_schedule,
        label_B_or_BLT=encode_prompts(bundle, prompts), B=len(prompts),
        g_seed=config.seed if seed is None else seed,
        cfg_list=[config.cfg] * count, tau_list=[config.tau] * count,
        top_k=config.top_k, top_p=config.top_p, returns_vemb=1,
        cfg_insertion_layer=[0], vae_type=1, ret_img=True,
        inference_mode=True, return_feature_trace=True,
    )


def _run_infinity(infer_method, **kwargs):
    """Use the exact call-site autocast pattern from Infinity/tools/run_infinity.py."""
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        return infer_method(**kwargs)


def _cpu(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.float() if value.is_floating_point() else value
    if isinstance(value, list): return [_cpu(item) for item in value]
    if isinstance(value, tuple): return tuple(_cpu(item) for item in value)
    return value


def _image(result, index):
    return Image.fromarray(result[2][index].detach().cpu().flip(-1).numpy(), mode="RGB")


def _save_comparison(images, labels, path, title):
    width, height, header = max(x.width for x in images), max(x.height for x in images), 54
    canvas = Image.new("RGB", (width * len(images), height + header), "white")
    draw = ImageDraw.Draw(canvas); draw.text((8, 5), title, fill="black")
    for i, (image, label) in enumerate(zip(images, labels)):
        x = i * width
        canvas.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (x, header))
        draw.text((x + 8, 28), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


@dataclass
class StepwiseFeatureExperimentResult:
    content_prompt: str
    style_prompt: str
    content_trace: list
    style_trace: list
    content_baseline: tuple
    style_baseline: tuple
    pfb_by_step: dict
    content_ortho_by_step: dict
    comparison_paths: dict


@torch.inference_mode()
def run_stepwise_feature_experiment(
    bundle, config, *, content_prompt: str, style_prompt: str,
    inject_steps: Iterable[int] | None = None, seed=None, sac=False,
    style_rank=1, content_rank=1, alpha=None, strength=1.0,
    projection_strength=1.0, preserve_mean=True, output_dir=None,
    case_name="stepwise_case",
):
    model = bundle.infinity_model
    # _infer_kwargs performs T5 encoding before entering _run_infinity. This is
    # deliberately identical to tools/run_infinity.py: encode, then autocast model inference.
    content_kwargs = _infer_kwargs(bundle, config, [content_prompt], seed)
    style_kwargs = _infer_kwargs(bundle, config, [style_prompt], seed)
    content_baseline = _run_infinity(model.autoregressive_infer_cfg, **content_kwargs)
    style_baseline = _run_infinity(model.autoregressive_infer_cfg, **style_kwargs)
    content_trace = [_cpu(x) for x in content_baseline[3]]
    style_trace = [_cpu(x) for x in style_baseline[3]]
    content_baseline, style_baseline = _cpu(content_baseline), _cpu(style_baseline)
    count = len(bundle.scale_schedule)
    steps = list(range(count)) if inject_steps is None else sorted({int(x) for x in inject_steps})
    if any(x < 0 or x >= count for x in steps): raise ValueError(f"inject_steps must be within [0, {count-1}]")
    output_dir = None if output_dir is None else Path(output_dir)
    baseline_image = _image(content_baseline, 0)
    if output_dir: output_dir.mkdir(parents=True, exist_ok=True); baseline_image.save(output_dir / "baseline.png")
    pfb_results, ortho_results, comparisons = {}, {}, {}
    paired = _infer_kwargs(bundle, config, [content_prompt, content_prompt], seed)
    used_alpha = config.paper_alpha if alpha is None else alpha
    for step in steps:
        pfb_feature = principal_feature_blend(content_trace[step], style_trace[step], rank=style_rank, alpha=used_alpha, strength=strength)
        ortho_feature = content_orthogonal_feature_blend(
            content_trace[step], style_trace[step], content_trace[step],
            style_rank=style_rank, content_rank=content_rank, alpha=used_alpha,
            strength=strength, projection_strength=projection_strength, preserve_mean=preserve_mean,
        )
        pfb_results[step] = _cpu(_run_infinity(model.autoregressive_infer_pfb,
            **paired, pfb_feature=pfb_feature, inject_step=step, f_con=content_trace, sac=sac))
        ortho_results[step] = _cpu(_run_infinity(model.autoregressive_infer_content_ortho,
            **paired, content_ortho_feature=ortho_feature, inject_step=step, f_con=content_trace, sac=sac))
        if output_dir:
            pfb_image, ortho_image = _image(pfb_results[step], 1), _image(ortho_results[step], 1)
            pfb_image.save(output_dir / f"step_{step:02d}_pfb.png")
            ortho_image.save(output_dir / f"step_{step:02d}_content_ortho.png")
            comparison = output_dir / f"step_{step:02d}_comparison.png"
            _save_comparison([baseline_image, pfb_image, ortho_image], ["Baseline", "PFB", "Content-Orthogonal"], comparison, f"{case_name} | step {step}")
            comparisons[step] = comparison
    return StepwiseFeatureExperimentResult(content_prompt, style_prompt, content_trace, style_trace, content_baseline, style_baseline, pfb_results, ortho_results, comparisons)


def display_stepwise_comparisons(result):
    from IPython.display import display
    for step, path in sorted(result.comparison_paths.items()):
        print(f"step {step}: {path}"); display(Image.open(path))
