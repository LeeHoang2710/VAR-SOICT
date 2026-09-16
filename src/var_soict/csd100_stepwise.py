"""CSD100 case selection and prompt construction for the step-wise experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .stepwise_feature_experiment import run_stepwise_feature_experiment


DEFAULT_EXAMPLE_PAIR = ("fox+graffiti", "pen+artwork")
OBJECT_PROMPT_DESCRIPTORS = {
    "fox": "red fox",
    "mushroom": "golden mushroom",
    "bear": "cute bear",
    "piano": "grand piano",
    "duck": "rubber duck",
    "flower": "pink flower",
    "camera": "vintage camera",
    "car": "sports car",
    "bicycle": "blue bicycle",
    "rabbit": "white rabbit",
    "turtle": "green turtle",
}


def _find_csd100_dir(root: Path, explicit_dir: Path | None = None) -> Path:
    candidates = [explicit_dir, root / "csd100", Path("/content/VAR_SOICT/csd100")]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and any(candidate.glob("*+*/00.jpg")):
            return candidate.resolve()
    raise FileNotFoundError("Could not find CSD100 with *+*/00.jpg items")


def _parse_item(folder: Path):
    if "+" not in folder.name:
        raise ValueError(f"Invalid CSD100 item: {folder.name}")
    object_label, style_label = folder.name.split("+", 1)
    image_path = folder / "00.jpg"
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    clean = lambda value: value.replace("_", " ").replace("-", " ")
    return clean(object_label), clean(style_label), image_path


@dataclass(frozen=True)
class CSD100StepwiseCase:
    case_name: str
    content_item_id: str
    style_item_id: str
    content_prompt: str
    style_prompt: str
    content_reference_path: Path
    style_reference_path: Path
    style_label: str


def _style_suffix(style_label: str) -> str:
    label = " ".join(style_label.split())
    return label if label.lower().endswith("style") else f"{label} style"


def load_csd100_stepwise_case(
    var_soict_root: Path | str,
    *,
    content_item_id: str,
    style_item_id: str,
    csd100_dir: Path | str | None = None,
) -> CSD100StepwiseCase:
    """Load two CSD100 entries and build a controlled same-subject prompt pair."""
    root = Path(var_soict_root).resolve()
    csd_dir = _find_csd100_dir(root, None if csd100_dir is None else Path(csd100_dir))
    content_object, _, content_path = _parse_item(csd_dir / content_item_id)
    _, style_label, style_path = _parse_item(csd_dir / style_item_id)
    subject = OBJECT_PROMPT_DESCRIPTORS.get(content_object, content_object)
    content_prompt = f"a photo of {subject}"
    style_prompt = f"{content_prompt}, in {_style_suffix(style_label)}"
    return CSD100StepwiseCase(
        case_name=f"{content_item_id}__STYLE__{style_item_id}",
        content_item_id=content_item_id,
        style_item_id=style_item_id,
        content_prompt=content_prompt,
        style_prompt=style_prompt,
        content_reference_path=content_path,
        style_reference_path=style_path,
        style_label=style_label,
    )


def run_csd100_stepwise(
    bundle,
    config,
    *,
    var_soict_root: Path | str,
    content_item_id: str,
    style_item_id: str,
    output_root: Path | str,
    inject_steps=None,
    seed=None,
    sac=False,
    style_rank=1,
    content_rank=1,
    content_variance_threshold=None,
    strength=1.0,
    projection_strength=1.0,
    preserve_mean=False,
):
    """Load one CSD100 pairing, run both hypotheses, and save comparisons."""
    case = load_csd100_stepwise_case(
        var_soict_root,
        content_item_id=content_item_id,
        style_item_id=style_item_id,
    )
    output_dir = Path(output_root) / case.case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.open(case.content_reference_path).convert("RGB").save(output_dir / "content_reference.jpg")
    Image.open(case.style_reference_path).convert("RGB").save(output_dir / "style_reference.jpg")
    metadata = asdict(case)
    metadata["content_reference_path"] = str(metadata["content_reference_path"])
    metadata["style_reference_path"] = str(metadata["style_reference_path"])
    with (output_dir / "case.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    result = run_stepwise_feature_experiment(
        bundle,
        config,
        content_prompt=case.content_prompt,
        style_prompt=case.style_prompt,
        style_reference_path=case.style_reference_path,
        inject_steps=inject_steps,
        seed=seed,
        sac=sac,
        style_rank=style_rank,
        content_rank=content_rank,
        content_variance_threshold=content_variance_threshold,
        strength=strength,
        projection_strength=projection_strength,
        preserve_mean=preserve_mean,
        output_dir=output_dir,
        case_name=case.case_name,
    )
    return case, result


def run_csd100_example(
    bundle,
    config,
    *,
    var_soict_root: Path | str,
    output_root: Path | str,
    **experiment_kwargs,
):
    """Run the default curated CSD100 pair used for a quick smoke experiment."""
    content_item_id, style_item_id = DEFAULT_EXAMPLE_PAIR
    return run_csd100_stepwise(
        bundle,
        config,
        var_soict_root=var_soict_root,
        content_item_id=content_item_id,
        style_item_id=style_item_id,
        output_root=output_root,
        **experiment_kwargs,
    )
