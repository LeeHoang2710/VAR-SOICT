from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from .config import ExperimentConfig, RuntimePaths
from .plotting import download_output_zip, safe_name, save_image_tensor, show_final_aggregate_session, show_variant_session
from .segmentation import PromptObjectMasker, save_mask_image, save_mask_overlay


class Infinity2BExperiment:
    def __init__(
        self,
        *,
        engine,
        sessions: list[dict[str, object]],
        cases: list[dict[str, object]],
        config: ExperimentConfig,
        paths: RuntimePaths,
    ):
        self.engine = engine
        self.sessions = sessions
        self.cases = cases
        self.config = config
        self.paths = paths
        self.baseline_results = {}
        self.pfb_sac_results = {}
        self.multistep_results = {}
        self.top1_style_steps_results = {}
        self.object_masked_results = {}
        self.object_masks = {}
        self.style_reference_masks = {}

    def run_baseline(self):
        if self.config.run_baseline:
            for case in tqdm(self.cases, desc="Infinity-2B baseline content images"):
                image = self.engine.generate_content_image(case["prompt"])
                self.baseline_results[case["case_id"]] = image
                save_image_tensor(image, self.paths.baseline_dir / f"{case['case_id']}.png")

        print(f"baseline images available: {len(self.baseline_results)} / {len(self.cases)}")
        if self.baseline_results:
            print("Baseline galleries:")
            for session in self.sessions:
                gallery_path = self.paths.baseline_dir / f"session_{session['session_id'] + 1:02d}.png"
                show_variant_session(
                    session,
                    self.cases,
                    self.baseline_results,
                    "Infinity-2B baseline",
                    gallery_path,
                    self.engine.get_style_image,
                )
        return self.baseline_results

    def run_variant_section(
        self,
        variant_name,
        *,
        pfb_feature_indices,
        style_decay,
        style_strength,
        enable_sac,
        rank=None,
        style_strength_by_step=None,
        feature_masks_by_case=None,
        style_masks_by_case=None,
        masked_background_strength=0.0,
        split_style_regions=False,
        foreground_rank=None,
        background_rank=None,
    ):
        section_dir = self.paths.variant_dir / safe_name(variant_name)
        section_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for case in tqdm(self.cases, desc=variant_name):
            image = self.engine.generate_variant_image(
                case["prompt"],
                case["style_path"],
                pfb_feature_indices=pfb_feature_indices,
                style_decay=style_decay,
                style_strength=style_strength,
                enable_sac=enable_sac,
                rank=rank,
                style_strength_by_step=style_strength_by_step,
                feature_masks_by_step=(feature_masks_by_case or {}).get(case["case_id"]),
                style_masks_by_step=(style_masks_by_case or {}).get(case["case_id"]),
                masked_background_strength=masked_background_strength,
                split_style_regions=split_style_regions,
                foreground_rank=foreground_rank,
                background_rank=background_rank,
            )
            results[case["case_id"]] = image
            save_image_tensor(image, section_dir / f"{case['case_id']}.png")

        for session in self.sessions:
            gallery_path = section_dir / f"session_{session['session_id'] + 1:02d}.png"
            show_variant_session(session, self.cases, results, variant_name, gallery_path, self.engine.get_style_image)
        return results

    def run_pfb_sac(self):
        if self.config.run_pfb_sac:
            self.pfb_sac_results = self.run_variant_section(
                "PFB + SAC",
                pfb_feature_indices=[self.config.paper_pfb_feature_index],
                style_decay=1.0,
                style_strength=self.config.base_style_strength,
                enable_sac=True,
            )
        print(f"PFB + SAC images available: {len(self.pfb_sac_results)} / {len(self.cases)}")
        return self.pfb_sac_results

    def run_multistep_decay(self):
        if self.config.run_multistep:
            self.multistep_results = self.run_variant_section(
                "Multi-step PFB + SAC (steps 2,4,6,8; strength=1.0, decay=0.75)",
                pfb_feature_indices=self.config.multistep_feature_indices,
                style_decay=self.config.multistep_style_decay,
                style_strength=self.config.multistep_style_strength,
                enable_sac=True,
            )
        print(f"multi-step decay images available: {len(self.multistep_results)} / {len(self.cases)}")
        return self.multistep_results

    def run_top1_style_steps(self):
        if max(self.config.top1_style_feature_indices) >= len(self.engine.scale_schedule):
            raise RuntimeError(
                f"This variant requires at least {max(self.config.top1_style_feature_indices) + 1} Infinity scales; "
                f"got {len(self.engine.scale_schedule)}."
            )
        if self.config.run_top1_style_steps:
            self.top1_style_steps_results = self.run_variant_section(
                "Top-1 SVD PFB + SAC (style steps 0,1,2,9; strength=1.0, decay=0.75)",
                pfb_feature_indices=self.config.top1_style_feature_indices,
                style_decay=self.config.top1_style_decay,
                style_strength=self.config.top1_style_strength,
                enable_sac=True,
                rank=self.config.top1_style_svd_rank,
            )
        print(f"top-1 style-step images available: {len(self.top1_style_steps_results)} / {len(self.cases)}")
        return self.top1_style_steps_results

    def build_prompt_object_masks(self):
        mask_dir = self.paths.output_dir / "object_masks"
        overlay_dir = self.paths.output_dir / "object_mask_overlays"
        masker = PromptObjectMasker(self.config.object_mask_model_id, device=self.config.object_mask_device)
        masks = {}

        for case in tqdm(self.cases, desc="CLIPSeg prompt object masks"):
            content_image = self.baseline_results.get(case["case_id"])
            if content_image is None:
                content_image = self.engine.generate_content_image(case["prompt"])
                self.baseline_results[case["case_id"]] = content_image
                save_image_tensor(content_image, self.paths.baseline_dir / f"{case['case_id']}.png")

            mask = masker.segment_image_tensor(
                content_image,
                case["content_prompt"],
                threshold=self.config.object_mask_threshold,
            )
            masks[case["case_id"]] = mask
            save_mask_image(mask, mask_dir / f"{case['case_id']}_mask.png")
            save_mask_overlay(content_image, mask, overlay_dir / f"{case['case_id']}_overlay.png")

        self.object_masks = masks
        print(f"object masks available: {len(self.object_masks)} / {len(self.cases)}")
        print("mask previews:", overlay_dir)
        return self.object_masks

    def build_style_reference_masks(self):
        mask_dir = self.paths.output_dir / "style_reference_object_masks"
        overlay_dir = self.paths.output_dir / "style_reference_object_mask_overlays"
        masker = PromptObjectMasker(self.config.object_mask_model_id, device=self.config.object_mask_device)
        masks = {}

        for session in tqdm(self.sessions, desc="CLIPSeg style-reference object masks"):
            style_image = self.engine.get_style_image(session["style_path"])
            mask = masker.segment_image_tensor(
                style_image,
                self.config.style_reference_mask_prompt,
                threshold=self.config.style_reference_mask_threshold,
            )
            masks[str(session["style_path"])] = mask
            save_mask_image(mask, mask_dir / f"{session['style_id']}_style_object_mask.png")
            save_mask_overlay(style_image, mask, overlay_dir / f"{session['style_id']}_style_object_overlay.png")

        self.style_reference_masks = masks
        print(f"style reference masks available: {len(self.style_reference_masks)} / {len(self.sessions)}")
        print("style mask previews:", overlay_dir)
        return self.style_reference_masks

    def run_object_masked_style_steps(self):
        if self.config.run_object_masked_style_steps:
            if not self.object_masks:
                self.build_prompt_object_masks()
            if self.config.object_masked_split_style_regions and not self.style_reference_masks:
                self.build_style_reference_masks()

            masked_steps = set(self.config.object_masked_mask_feature_indices)
            feature_masks_by_case = {
                case_id: {step: mask for step in masked_steps}
                for case_id, mask in self.object_masks.items()
            }
            style_masks_by_case = {}
            if self.config.object_masked_split_style_regions:
                for case in self.cases:
                    style_mask = self.style_reference_masks[str(case["style_path"])]
                    style_masks_by_case[case["case_id"]] = {step: style_mask for step in masked_steps}

            self.object_masked_results = self.run_variant_section(
                "Object-aware PFB + SAC (global 0,1; fg/bg split 3,6,9)",
                pfb_feature_indices=self.config.object_masked_feature_indices,
                style_decay=1.0,
                style_strength=1.0,
                enable_sac=True,
                style_strength_by_step=self.config.object_masked_style_strength_by_step,
                feature_masks_by_case=feature_masks_by_case,
                style_masks_by_case=style_masks_by_case,
                masked_background_strength=self.config.object_masked_background_strength,
                split_style_regions=self.config.object_masked_split_style_regions,
                foreground_rank=self.config.object_masked_foreground_svd_rank,
                background_rank=self.config.object_masked_background_svd_rank,
            )
        print(f"object-masked style-step images available: {len(self.object_masked_results)} / {len(self.cases)}")
        return self.object_masked_results

    def final_aggregate_columns(self):
        columns = [
            ("style reference", None),
        ]
        optional_columns = [
            ("baseline", self.baseline_results),
            ("PFB + SAC", self.pfb_sac_results),
            ("multi-step decay", self.multistep_results),
            ("top-1 style steps", self.top1_style_steps_results),
        ]
        columns.extend((label, results) for label, results in optional_columns if results)
        if self.object_masked_results:
            columns.append(("object-masked style", self.object_masked_results))
        return columns

    def plot_final_aggregate(self):
        if not self.config.run_aggregate:
            return
        expected = len(self.cases)
        available = {label: len(results) for label, results in self.final_aggregate_columns() if results is not None}
        if not available:
            raise RuntimeError("Run at least one generation variant before final aggregate comparison.")
        if any(count != expected for count in available.values()):
            raise RuntimeError(
                f"Run each selected generation variant before final aggregate comparison: "
                f"{available}, expected {expected} each."
            )
        for session in self.sessions:
            show_final_aggregate_session(
                session,
                self.cases,
                self.final_aggregate_columns(),
                self.paths.aggregate_dir / f"session_{session['session_id'] + 1:02d}_final_aggregate.png",
                self.engine.get_style_image,
            )

    def download_outputs(self, zip_path: Path | None = None):
        return download_output_zip(self.paths.output_dir, zip_path=zip_path)
