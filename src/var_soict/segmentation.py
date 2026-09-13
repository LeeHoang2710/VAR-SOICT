from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .plotting import image_tensor_to_pil


class PromptObjectMasker:
    """Prompt-guided foreground mask extraction with a lightweight CLIPSeg model."""

    def __init__(self, model_id: str = "CIDAS/clipseg-rd64-refined", device: str | None = None):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def segment_pil(self, image: Image.Image, object_prompt: str, threshold: float | None = None) -> torch.Tensor:
        inputs = self.processor(
            text=[object_prompt],
            images=[image.convert("RGB")],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.model(**inputs).logits
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)

        mask = torch.sigmoid(logits).unsqueeze(1)
        mask = F.interpolate(mask, size=(image.height, image.width), mode="bilinear", align_corners=False)
        mask = mask[0, 0].detach().float().cpu()
        mask = _normalize_mask(mask)
        if threshold is not None:
            mask = (mask >= float(threshold)).float()
        return mask

    def segment_image_tensor(self, image_01, object_prompt: str, threshold: float | None = None) -> torch.Tensor:
        return self.segment_pil(image_tensor_to_pil(image_01), object_prompt, threshold=threshold)


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    mask = mask - mask.min()
    max_value = mask.max().clamp_min(1e-8)
    return (mask / max_value).clamp(0, 1)


def save_mask_image(mask: torch.Tensor, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.clamp(0, 1).mul(255).byte().numpy(), mode="L").save(path)


def save_mask_overlay(image_01, mask: torch.Tensor, path, color=(255, 64, 64), alpha: float = 0.45) -> None:
    image = image_tensor_to_pil(image_01).convert("RGB")
    mask_image = Image.fromarray(mask.clamp(0, 1).mul(255).byte().numpy(), mode="L").resize(image.size)
    overlay = Image.new("RGB", image.size, color)
    blended = Image.composite(Image.blend(image, overlay, alpha), image, mask_image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(path)
