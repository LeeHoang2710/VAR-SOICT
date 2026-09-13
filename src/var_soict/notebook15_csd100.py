from __future__ import annotations

import gc
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
from PIL import Image
from tqdm.auto import tqdm

from .config import ExperimentConfig, RuntimePaths
from .plotting import image_tensor_to_pil, save_image_tensor
from .segmentation import PromptObjectMasker, save_mask_image, save_mask_overlay


EXAMPLE_PAIRS = [
    ("fox+graffiti", "pen+artwork"),
    ("mushroom+melting_golden_3D_rendering", "horse+rainbow_flowing_smoke_wave"),
    ("scarecrow+melting_golden_3D_rendering", "cat+glowing"),
    ("bear+glowing", "teapot+psychedelic"),
    ("piano+impressionism", "duck+blueprint"),
    ("duck+blueprint", "flower+mosaic"),
    ("flower+pixel", "brush+watercolor"),
    ("moose+origami", "cat+glowing"),
    ("bottle+drawing", "lantern+line_drawing_illustration_art"),
    ("muffin+drawing", "teacup+comic"),
    ("camera+artwork", "leopard+geometric"),
    ("camera+origami", "lollipop+origami"),
    ("car+rainbow_flowing_smoke_wave", "microphone+minimal_pastel_colors_art"),
    ("balloon+mosaic", "crown+art"),
    ("bicycle+blueprint", "saxophone+pop"),
    ("glass+watercolor_and_ink_wash", "robot+woodcut"),
    ("compass+flat_cartoon_illustration_art", "watermelon+papercut"),
    ("rabbit+sticker", "notebook+melting_golden_3D_rendering"),
    ("turtle+origami", "fox+minimal_pastel_colors_art"),
    ("umbrella+melting_golden_3D_rendering", "seagull+geometric"),
]


OBJECT_PROMPT_DESCRIPTORS = {
    "fox": "red fox",
    "mushroom": "golden mushroom",
    "scarecrow": "straw scarecrow",
    "bear": "cute bear",
    "piano": "grand piano",
    "duck": "rubber duck",
    "flower": "pink flower",
    "moose": "paper moose",
    "bottle": "glass bottle",
    "muffin": "frosted muffin",
    "camera": "vintage camera",
    "car": "sports car",
    "balloon": "hot air balloon",
    "bicycle": "blue bicycle",
    "glass": "clear glass",
    "compass": "brass compass",
    "rabbit": "white rabbit",
    "turtle": "green turtle",
    "umbrella": "red umbrella",
}


@dataclass(frozen=True)
class Notebook15OutputDirs:
    root: Path
    baseline: Path
    content_masks: Path
    content_mask_overlays: Path
    style_masks: Path
    style_mask_overlays: Path
    original: Path
    global_top2: Path
    object_masked: Path
    traces: Path
    diagnostics: Path
    galleries: Path
    evaluation: Path

    @property
    def all_dirs(self) -> tuple[Path, ...]:
        return (
            self.root,
            self.baseline,
            self.content_masks,
            self.content_mask_overlays,
            self.style_masks,
            self.style_mask_overlays,
            self.original,
            self.global_top2,
            self.object_masked,
            self.traces,
            self.diagnostics,
            self.galleries,
            self.evaluation,
        )


def find_var_soict_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    candidates = [start, *start.parents[:6], Path("/content/VAR_SOICT")]
    for candidate in candidates:
        if (candidate / "src" / "var_soict").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find VAR_SOICT root.")


def find_csd100_dir(var_soict_root: Path, explicit_dir: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_dir is not None:
        candidates.append(Path(explicit_dir))
    candidates.append(var_soict_root / "csd100")
    candidates.append(Path("/content/VAR_SOICT/csd100"))
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*+*/00.jpg")):
            return candidate.resolve()
    raise FileNotFoundError("Could not find CSD100. Expected it at VAR_SOICT/csd100.")


def notebook15_output_dirs(paths: RuntimePaths) -> Notebook15OutputDirs:
    root = paths.output_dir
    dirs = Notebook15OutputDirs(
        root=root,
        baseline=root / "baseline_content_stream",
        content_masks=root / "content_object_masks",
        content_mask_overlays=root / "content_object_mask_overlays",
        style_masks=root / "style_reference_object_masks",
        style_mask_overlays=root / "style_reference_object_mask_overlays",
        original=root / "original_pfb_sac_f3_full_rank",
        global_top2=root / "global_top2_pfb_sac_0_1_2_9",
        object_masked=root / "global012_object_masked_pfb_sac_3_6_9",
        traces=root / "traces",
        diagnostics=root / "scale_diagnostics",
        galleries=root / "galleries",
        evaluation=root / "evaluation",
    )
    for directory in dirs.all_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_csd100_item(folder: Path) -> dict[str, object]:
    if "+" not in folder.name:
        raise ValueError(f"CSD100 folder name must contain +: {folder.name}")
    object_label, style_label = folder.name.split("+", 1)
    image_path = folder / "00.jpg"
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    return {
        "folder": folder,
        "image_path": image_path,
        "object_label": object_label.replace("_", " ").replace("-", " "),
        "style_label": style_label.replace("_", " ").replace("-", " "),
        "item_id": folder.name,
    }


def detailed_object_phrase(object_label: str) -> str:
    return OBJECT_PROMPT_DESCRIPTORS.get(str(object_label), str(object_label))


def load_csd100_pair_rows(csd100_dir: Path, pairs: list[tuple[str, str]] | None = None) -> list[dict[str, object]]:
    pairs = EXAMPLE_PAIRS if pairs is None else pairs
    items = {
        path.name: parse_csd100_item(path)
        for path in sorted(Path(csd100_dir).iterdir())
        if path.is_dir() and (path / "00.jpg").exists()
    }
    missing = sorted({item_id for pair in pairs for item_id in pair} - set(items))
    if missing:
        raise FileNotFoundError("Missing CSD100 example item folders: " + ", ".join(missing))

    rows = []
    for pair_id, (content_item_id, style_item_id) in enumerate(pairs):
        content_item = items[content_item_id]
        style_item = items[style_item_id]
        pfb_prompt = f"a photo of {detailed_object_phrase(content_item['object_label'])}"
        rows.append(
            {
                "pair_id": pair_id,
                "case_id": f"{pair_id:02d}_{content_item_id}__STYLE__{style_item_id}",
                "content_id": content_item["item_id"],
                "content_object": content_item["object_label"],
                "content_style": content_item["style_label"],
                "content_path": content_item["image_path"],
                "style_id": style_item["item_id"],
                "style_object": style_item["object_label"],
                "style_label": style_item["style_label"],
                "style_path": style_item["image_path"],
                "pfb_prompt": pfb_prompt,
                "prompt": pfb_prompt,
            }
        )
    return rows


def pair_slug(row: dict[str, object]) -> str:
    return f"{int(row['pair_id']):02d}_{row['content_id']}__STYLE__{row['style_id']}"


def load_mask_tensor(mask_path: Path) -> torch.Tensor:
    return torchvision.transforms.functional.pil_to_tensor(Image.open(mask_path).convert("L")).float()[0] / 255.0


class Notebook15CSD100Experiment:
    variant_order = ["original_pfb_sac", "global_top2", "object_masked"]
    variant_labels = {
        "original_pfb_sac": "Original PFB+SAC\nF3 full-rank",
        "global_top2": "Global top-2\n[0,1,2,9]",
        "object_masked": "Global 0,1,2 + object\nmasked [3,6,9]",
    }
    variant_injected_indices = {
        "original_pfb_sac": [2],
        "global_top2": [0, 1, 2, 9],
        "object_masked": [0, 1, 2, 3, 6, 9],
    }

    def __init__(
        self,
        *,
        engine,
        pair_rows: list[dict[str, object]],
        config: ExperimentConfig,
        paths: RuntimePaths,
        output_dirs: Notebook15OutputDirs | None = None,
    ):
        self.engine = engine
        self.pair_rows = pair_rows
        self.config = config
        self.paths = paths
        self.dirs = notebook15_output_dirs(paths) if output_dirs is None else output_dirs
        self.baseline_results: dict[int, torch.Tensor] = {}
        self.content_masks: dict[int, torch.Tensor] = {}
        self.style_masks: dict[str, torch.Tensor] = {}
        self.original_results = {}
        self.global_top2_results = {}
        self.object_masked_results = {}

    def original_output_path(self, row):
        return self.dirs.original / f"{pair_slug(row)}_original_pfb_sac_f3_full_rank.png"

    def global_top2_output_path(self, row):
        return self.dirs.global_top2 / f"{pair_slug(row)}_global_top2_0_1_2_9.png"

    def object_masked_output_path(self, row):
        return self.dirs.object_masked / f"{pair_slug(row)}_global012_object_masked_3_6_9.png"

    def output_path_for_variant(self, row, variant: str) -> Path:
        if variant == "original_pfb_sac":
            return self.original_output_path(row)
        if variant == "global_top2":
            return self.global_top2_output_path(row)
        if variant == "object_masked":
            return self.object_masked_output_path(row)
        raise ValueError(f"Unknown variant: {variant}")

    def baseline_path(self, row):
        return self.dirs.baseline / f"{pair_slug(row)}_baseline.png"

    def content_mask_path(self, row):
        return self.dirs.content_masks / f"{pair_slug(row)}_content_mask.png"

    def content_mask_overlay_path(self, row):
        return self.dirs.content_mask_overlays / f"{pair_slug(row)}_content_mask_overlay.png"

    def style_mask_path(self, row):
        return self.dirs.style_masks / f"{pair_slug(row)}_style_mask.png"

    def style_mask_overlay_path(self, row):
        return self.dirs.style_mask_overlays / f"{pair_slug(row)}_style_mask_overlay.png"

    def trace_path(self, row, variant):
        return self.dirs.traces / f"{pair_slug(row)}_{variant}_trace.pt"

    def meta_path(self, row, variant):
        return self.dirs.traces / f"{pair_slug(row)}_{variant}_meta.json"

    def run_baselines_and_masks(self, *, force=False):
        masker = PromptObjectMasker(self.config.object_mask_model_id, device=self.config.object_mask_device)
        for row in tqdm(self.pair_rows, desc="Notebook 15 baselines + CLIPSeg masks"):
            pair_id = int(row["pair_id"])
            baseline_path = self.baseline_path(row)
            content_mask_path = self.content_mask_path(row)
            style_mask_path = self.style_mask_path(row)

            if baseline_path.exists() and content_mask_path.exists() and style_mask_path.exists() and not force:
                baseline = torchvision.transforms.functional.to_tensor(Image.open(baseline_path).convert("RGB")).unsqueeze(0)
                self.baseline_results[pair_id] = baseline
                self.content_masks[pair_id] = load_mask_tensor(content_mask_path)
                self.style_masks[str(row["style_path"])] = load_mask_tensor(style_mask_path)
                continue

            baseline = self.engine.generate_content_image(
                row["pfb_prompt"],
                seed=self.config.seed + int(row["pair_id"]),
            )
            save_image_tensor(baseline, baseline_path)
            self.baseline_results[pair_id] = baseline

            content_mask = masker.segment_image_tensor(
                baseline,
                row["content_object"],
                threshold=self.config.object_mask_threshold,
            )
            style_image = self.engine.get_style_image(row["style_path"])
            style_mask = masker.segment_image_tensor(
                style_image,
                self.config.style_reference_mask_prompt,
                threshold=self.config.style_reference_mask_threshold,
            )
            self.content_masks[pair_id] = content_mask
            self.style_masks[str(row["style_path"])] = style_mask

            save_mask_image(content_mask, content_mask_path)
            save_mask_overlay(baseline, content_mask, self.content_mask_overlay_path(row))
            save_mask_image(style_mask, style_mask_path)
            save_mask_overlay(style_image, style_mask, self.style_mask_overlay_path(row))

        print(f"baseline images available: {len(self.baseline_results)} / {len(self.pair_rows)}")
        print("mask overlays:", self.dirs.content_mask_overlays)
        print("style mask overlays:", self.dirs.style_mask_overlays)
        return self.baseline_results, self.content_masks, self.style_masks

    def _save_generation_artifacts(self, row, variant, result, output_path, variant_config):
        save_image_tensor(result["stylized_image_01"], output_path)
        trace_payload = {
            "content_features": [feature.detach().float().cpu() for feature in result["content_features"]],
            "style_features": [feature.detach().float().cpu() for feature in result["style_features"]],
            "generation_features": [feature.detach().float().cpu() for feature in result["generation_features"]],
        }
        torch.save(trace_payload, self.trace_path(row, variant))
        metadata = {
            "pair_id": int(row["pair_id"]),
            "content_id": row["content_id"],
            "style_id": row["style_id"],
            "content_object": row["content_object"],
            "style_object": row["style_object"],
            "style_label": row["style_label"],
            "pfb_prompt": row["pfb_prompt"],
            "variant": variant,
            "variant_label": self.variant_labels[variant],
            "variant_config": variant_config,
            "seed": self.config.seed + int(row["pair_id"]),
            "cfg": self.config.cfg,
            "tau": self.config.tau,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "pfb_relative_change_by_step": result.get("pfb_relative_change_by_step"),
            "sac_calls": result.get("sac_calls"),
            "max_q_copy_error": result.get("max_q_copy_error"),
            "max_k_copy_error": result.get("max_k_copy_error"),
        }
        with self.meta_path(row, variant).open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

    def _run_variant(self, variant: str, *, force=False, **kwargs):
        results = {}
        for row in tqdm(self.pair_rows, desc=self.variant_labels[variant].replace("\n", " ")):
            output_path = self.output_path_for_variant(row, variant)
            if output_path.exists() and self.trace_path(row, variant).exists() and not force:
                results[int(row["pair_id"])] = output_path
                continue

            result = self.engine.generate_variant_result(
                row["pfb_prompt"],
                row["style_path"],
                seed=self.config.seed + int(row["pair_id"]),
                **kwargs,
            )
            self._save_generation_artifacts(row, variant, result, output_path, kwargs)
            results[int(row["pair_id"])] = output_path
            del result
            gc.collect()
            torch.cuda.empty_cache()
        return results

    def run_original_pfb_sac(self, *, force=False):
        self.original_results = self._run_variant(
            "original_pfb_sac",
            force=force,
            pfb_feature_indices=[2],
            style_decay=1.0,
            style_strength=1.0,
            enable_sac=True,
            rank=None,
        )
        return self.original_results

    def run_global_top2(self, *, force=False):
        self.global_top2_results = self._run_variant(
            "global_top2",
            force=force,
            pfb_feature_indices=[0, 1, 2, 9],
            style_decay=0.75,
            style_strength=1.0,
            enable_sac=True,
            rank=2,
        )
        return self.global_top2_results

    def run_object_masked(self, *, force=False):
        if not self.content_masks or not self.style_masks:
            self.run_baselines_and_masks(force=False)
        masked_steps = [3, 6, 9]
        strength_by_step = {0: 0.8, 1: 0.64, 2: 0.512, 3: 1.0, 6: 0.75, 9: 0.5}
        results = {}
        for row in tqdm(self.pair_rows, desc=self.variant_labels["object_masked"].replace("\n", " ")):
            output_path = self.object_masked_output_path(row)
            if output_path.exists() and self.trace_path(row, "object_masked").exists() and not force:
                results[int(row["pair_id"])] = output_path
                continue

            content_mask = self.content_masks[int(row["pair_id"])]
            style_mask = self.style_masks[str(row["style_path"])]
            feature_masks_by_step = {step: content_mask for step in masked_steps}
            style_masks_by_step = {step: style_mask for step in masked_steps}
            kwargs = {
                "pfb_feature_indices": [0, 1, 2, 3, 6, 9],
                "style_decay": 1.0,
                "style_strength": 1.0,
                "enable_sac": True,
                "rank": 1,
                "style_strength_by_step": strength_by_step,
                "feature_masks_by_step": feature_masks_by_step,
                "style_masks_by_step": style_masks_by_step,
                "masked_background_strength": 0.0,
                "split_style_regions": True,
                "foreground_rank": 1,
                "background_rank": 1,
            }
            result = self.engine.generate_variant_result(
                row["pfb_prompt"],
                row["style_path"],
                seed=self.config.seed + int(row["pair_id"]),
                **kwargs,
            )
            serializable_kwargs = {
                **kwargs,
                "feature_masks_by_step": sorted(feature_masks_by_step),
                "style_masks_by_step": sorted(style_masks_by_step),
            }
            self._save_generation_artifacts(row, "object_masked", result, output_path, serializable_kwargs)
            results[int(row["pair_id"])] = output_path
            del result
            gc.collect()
            torch.cuda.empty_cache()
        self.object_masked_results = results
        return self.object_masked_results

    def run_all_variants(self, *, force=False):
        return {
            "original_pfb_sac": self.run_original_pfb_sac(force=force),
            "global_top2": self.run_global_top2(force=force),
            "object_masked": self.run_object_masked(force=force),
        }

    def _decode_trace_to_pil_images(self, trace):
        images = []
        with torch.no_grad():
            for cumulative_codes in trace:
                image_01 = self.engine._decode_summed_codes_to_image_01(cumulative_codes.to(self.engine.device))
                images.append(image_tensor_to_pil(image_01.detach().float().cpu()))
        return images

    def plot_scale_diagnostic(self, row, variant: str, *, save=True):
        import matplotlib.pyplot as plt

        trace_path = self.trace_path(row, variant)
        if not trace_path.exists():
            print("missing trace:", trace_path)
            return None
        trace_payload = torch.load(trace_path, map_location="cpu")
        traces = [
            trace_payload["content_features"],
            trace_payload["style_features"],
            trace_payload["generation_features"],
        ]
        decoded_rows = [self._decode_trace_to_pil_images(trace) for trace in traces]
        row_labels = [
            f"Content stream\n{row['pfb_prompt']}",
            f"Style reference\n{row['style_object']}\n({row['style_label']})",
            self.variant_labels[variant],
        ]
        num_scales = min(len(images) for images in decoded_rows)
        figure, axes = plt.subplots(3, num_scales + 1, figsize=(1.55 * (num_scales + 1), 5.2), squeeze=False)
        for axis in axes.reshape(-1):
            axis.axis("off")
        for row_id, (label, images) in enumerate(zip(row_labels, decoded_rows)):
            axes[row_id, 0].text(0.5, 0.5, label, ha="center", va="center", fontsize=9, wrap=True)
            for scale_id in range(num_scales):
                axes[row_id, scale_id + 1].imshow(images[scale_id])
                if row_id == 0:
                    marker = "*" if scale_id in self.variant_injected_indices[variant] else ""
                    axes[row_id, scale_id + 1].set_title(f"R{scale_id + 1}{marker}", fontsize=8)
        figure.suptitle(f"Pair {int(row['pair_id']):02d}: {row['content_id']} -> {row['style_id']}", fontsize=11)
        figure.tight_layout()
        save_path = self.dirs.diagnostics / f"{pair_slug(row)}_{variant}_scale_diagnostic.png"
        if save:
            figure.savefig(save_path, dpi=180, bbox_inches="tight")
            print("saved diagnostic:", save_path)
        plt.show()
        plt.close(figure)
        return save_path

    def plot_all_scale_diagnostics(self, variants: list[str] | None = None):
        variants = self.variant_order if variants is None else variants
        for row in tqdm(self.pair_rows, desc="Scale diagnostics"):
            for variant in variants:
                self.plot_scale_diagnostic(row, variant, save=True)

    def show_final_gallery(self, variants: list[str] | None = None):
        import matplotlib.pyplot as plt

        variants = self.variant_order if variants is None else variants
        columns = ["content source", "style reference", "baseline content stream"] + [
            self.variant_labels[variant] for variant in variants
        ]
        figure, axes = plt.subplots(
            len(self.pair_rows),
            len(columns),
            figsize=(3.2 * len(columns), 3.35 * len(self.pair_rows)),
            squeeze=False,
        )
        for row_id, row in enumerate(self.pair_rows):
            image_paths = [row["content_path"], row["style_path"], self.baseline_path(row)] + [
                self.output_path_for_variant(row, variant) for variant in variants
            ]
            titles = [
                f"content: {row['content_object']}\nsource style: {row['content_style']}",
                f"style: {row['style_label']}\nobject: {row['style_object']}",
                row["pfb_prompt"],
            ] + [self.variant_labels[variant] for variant in variants]
            for col_id, (path, title) in enumerate(zip(image_paths, titles)):
                axis = axes[row_id, col_id]
                if Path(path).exists():
                    axis.imshow(Image.open(path).convert("RGB"))
                else:
                    axis.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
                axis.axis("off")
                axis.set_title(title, fontsize=8)
        figure.suptitle("Notebook 15: Three PFB + SAC variants on CSD100 20 pairs", fontsize=14)
        figure.tight_layout()
        save_path = self.dirs.galleries / "notebook15_three_variant_csd100_20_pair_gallery.png"
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
        print("saved gallery:", save_path)
        plt.show()
        plt.close(figure)
        return save_path

    def package_outputs(self, *, include_traces=False) -> Path:
        zip_path = self.dirs.root / "notebook15_three_variant_outputs_and_metrics.zip"
        files_to_add: list[Path] = []
        for directory in (
            self.dirs.baseline,
            self.dirs.content_masks,
            self.dirs.content_mask_overlays,
            self.dirs.style_masks,
            self.dirs.style_mask_overlays,
            self.dirs.original,
            self.dirs.global_top2,
            self.dirs.object_masked,
            self.dirs.diagnostics,
            self.dirs.galleries,
            self.dirs.evaluation,
        ):
            for suffix in ("*.png", "*.jpg", "*.jpeg", "*.csv"):
                files_to_add.extend(sorted(directory.glob(suffix)))
        files_to_add.extend(sorted(self.dirs.traces.glob("*.json")))
        if include_traces:
            files_to_add.extend(sorted(self.dirs.traces.glob("*.pt")))
        files_to_add = sorted(set(files_to_add))
        if not files_to_add:
            raise FileNotFoundError("No notebook 15 outputs found yet.")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in files_to_add:
                archive.write(file_path, arcname=file_path.relative_to(self.dirs.root))
        try:
            from google.colab import files

            files.download(str(zip_path))
        except Exception:
            print("ZIP ready:", zip_path)
        return zip_path


def build_notebook15_experiment(
    *,
    engine,
    config: ExperimentConfig,
    paths: RuntimePaths,
    var_soict_root: Path | None = None,
    csd100_dir: Path | None = None,
) -> Notebook15CSD100Experiment:
    var_soict_root = find_var_soict_root() if var_soict_root is None else Path(var_soict_root)
    csd100_dir = find_csd100_dir(var_soict_root, explicit_dir=csd100_dir)
    pair_rows = load_csd100_pair_rows(csd100_dir)
    experiment = Notebook15CSD100Experiment(
        engine=engine,
        pair_rows=pair_rows,
        config=config,
        paths=paths,
    )
    print("CSD100 dir:", csd100_dir)
    print("Selected pairs:", len(pair_rows))
    print("Output dir:", experiment.dirs.root)
    return experiment
