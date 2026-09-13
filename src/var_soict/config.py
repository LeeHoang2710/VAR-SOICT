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
    output_run_name: str = "infinity2b_random_3_prompts_per_eval_style"
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
    top2_style_feature_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 9])
    top2_style_svd_rank: int = 2
    top2_style_strength: float = 1.0
    top2_style_decay: float = 0.75

    global_01369_feature_indices: list[int] = field(default_factory=lambda: [0, 1, 3, 6, 9])
    global_01369_svd_rank: int = 1
    global_01369_style_strength_by_step: dict[int, float] = field(
        default_factory=lambda: {0: 0.8, 1: 0.6, 3: 1.0, 6: 0.75, 9: 0.5}
    )

    object_mask_model_id: str = "CIDAS/clipseg-rd64-refined"
    object_mask_device: str = "cpu"
    object_mask_threshold: float | None = None
    style_reference_mask_prompt: str = "main object"
    style_reference_mask_threshold: float | None = None
    object_masked_feature_indices: list[int] = field(default_factory=lambda: [0, 1, 3, 6, 9])
    object_masked_mask_feature_indices: list[int] = field(default_factory=lambda: [3, 6, 9])
    object_masked_style_strength_by_step: dict[int, float] = field(
        default_factory=lambda: {0: 0.8, 1: 0.6, 3: 1.0, 6: 0.75, 9: 0.5}
    )
    object_masked_background_strength: float = 0.15
    object_masked_foreground_svd_rank: int = 1
    object_masked_background_svd_rank: int = 1
    object_masked_split_style_regions: bool = True

    run_baseline: bool = True
    run_pfb_sac: bool = True
    run_multistep: bool = True
    run_top1_style_steps: bool = True
    run_top2_style_steps: bool = False
    run_global_01369_style_steps: bool = False
    run_object_only_masked_style_steps: bool = False
    run_foreground_split_masked_style_steps: bool = False
    run_object_masked_style_steps: bool = False
    run_background_only_masked_style_steps: bool = False
    run_aggregate: bool = True
    save_scale_diagnostics: bool = True
    save_scale_diagnostics_for_all_variants: bool = False

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
