from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEPENDENCY_PACKAGES = [
    "gguf",
    "gradio",
    "transformers",
    "sentencepiece",
    "easydict",
    "typed-argument-parser",
    "seaborn",
    "kornia",
    "gputil",
    "colorama",
    "omegaconf",
    "timm==0.9.6",
    "decord",
    "pytz",
    "imageio",
    "einops",
    "opencv-python",
    "accelerate",
]


@dataclass(frozen=True)
class ExperimentConfig:
    root: Path = Path("/content/notebook_09_infinity2b_gguf")
    official_repo: str = "https://github.com/FoundationVision/Infinity.git"
    gguf_repo: str = "kzopp/Infinity-2B-GGUF_UNOFFICIAL"
    model_pn: str = "0.25M"
    t5_device: str = "cuda"

    cfg: float = 1.0
    top_k: int = 600
    top_p: float = 0.95
    tau: float = 0.1
    seed: int = 42

    prompts_per_style: int = 3
    prompt_sample_seed: int = 42
    expected_eval_styles: int = 10

    paper_alpha: float = 1.0
    paper_pfb_feature_index: int = 2
    paper_sac_prediction_start: int = 2
    base_style_strength: float = 0.8

    multistep_feature_indices: list[int] = field(default_factory=lambda: [2, 4, 6, 8])
    multistep_style_strength: float = 1.0
    multistep_style_decay: float = 0.75

    top1_style_feature_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 9])
    top1_style_svd_rank: int = 1
    top1_style_strength: float = 1.0
    top1_style_decay: float = 0.75

    run_baseline: bool = True
    run_pfb_sac: bool = True
    run_multistep: bool = True
    run_top1_style_steps: bool = True
    run_aggregate: bool = True

    @property
    def expected_cases_per_variant(self) -> int:
        return self.expected_eval_styles * self.prompts_per_style


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    port_dir: Path
    official_dir: Path
    asset_dir: Path
    runtime_root: Path
    output_dir: Path
    baseline_dir: Path
    variant_dir: Path
    aggregate_dir: Path


@dataclass(frozen=True)
class ModelFiles:
    port_script: Path
    port_utils: Path
    patched_basic: Path
    patched_infinity: Path
    infinity_gguf: Path
    t5_gguf: Path
    vae_path: Path
    patched_loader: Path | None = None


@dataclass
class ModelBundle:
    device: str
    text_tokenizer: object
    text_encoder: object
    vae: object
    infinity_model: object
    scale_schedule: list[tuple[int, int, int]]

