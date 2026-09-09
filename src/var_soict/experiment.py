from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from .config import ExperimentConfig, RuntimePaths
from .plotting import download_output_zip, safe_name, save_image_tensor, show_final_aggregate_session, show_variant_session


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

    def final_aggregate_columns(self):
        return [
            ("style reference", None),
            ("baseline", self.baseline_results),
            ("PFB + SAC", self.pfb_sac_results),
            ("multi-step decay", self.multistep_results),
            ("top-1 style steps", self.top1_style_steps_results),
        ]

    def plot_final_aggregate(self):
        if not self.config.run_aggregate:
            return
        expected = len(self.cases)
        available = {label: len(results) for label, results in self.final_aggregate_columns() if results is not None}
        if any(count != expected for count in available.values()):
            raise RuntimeError(
                f"Run baseline and all three style variants before final aggregate comparison: "
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

