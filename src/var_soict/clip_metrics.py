from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

try:
    from IPython.display import display
except Exception:  # pragma: no cover - only used outside notebooks.
    def display(value):
        print(value)


class CLIPMetricsEvaluator:
    def __init__(self, *, output_dir: Path, model_id: str = "openai/clip-vit-base-patch32", device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.image_cache: dict[str, torch.Tensor] = {}
        self.text_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _feature_tensor(output, projection_layer=None):
        if torch.is_tensor(output):
            return output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            pooled = output.pooler_output
            if projection_layer is not None and pooled.shape[-1] == projection_layer.in_features:
                return projection_layer(pooled)
            return pooled
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        raise TypeError(f"Could not convert CLIP output to a feature tensor: {type(output)}")

    @torch.no_grad()
    def image_embedding(self, image_path) -> torch.Tensor:
        image_path = Path(image_path)
        key = str(image_path.resolve())
        if key in self.image_cache:
            return self.image_cache[key]

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        output = self.model.get_image_features(**inputs)
        embedding = self._feature_tensor(output, getattr(self.model, "visual_projection", None)).float()
        embedding = F.normalize(embedding, dim=-1)
        self.image_cache[key] = embedding.detach().cpu()
        return self.image_cache[key]

    @torch.no_grad()
    def text_embedding(self, text: str) -> torch.Tensor:
        key = str(text)
        if key in self.text_cache:
            return self.text_cache[key]

        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        output = self.model.get_text_features(**inputs)
        embedding = self._feature_tensor(output, getattr(self.model, "text_projection", None)).float()
        embedding = F.normalize(embedding, dim=-1)
        self.text_cache[key] = embedding.detach().cpu()
        return self.text_cache[key]

    @staticmethod
    def cosine_score(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> float:
        score = F.cosine_similarity(embedding_a.float(), embedding_b.float(), dim=-1).item()
        return float(score)

    @staticmethod
    def harmonic_score(s_txt: float, s_img: float) -> float:
        denominator = float(s_txt) + float(s_img)
        return (2.0 * float(s_txt) * float(s_img) / denominator) if denominator != 0.0 else 0.0

    @staticmethod
    def style_text_prompt(row: dict[str, object]) -> str:
        return f"a {row['style_label']} style image"

    @staticmethod
    def content_text_prompt(row: dict[str, object]) -> str:
        return str(row["pfb_prompt"])

    @staticmethod
    def style_object_text_prompt(row: dict[str, object]) -> str:
        return f"a photo of {row['style_object']}"

    def compute_style_metrics(self, experiment, variants: list[str] | None = None) -> pd.DataFrame:
        variants = experiment.variant_order if variants is None else variants
        rows = []
        for row in tqdm(experiment.pair_rows, desc="CLIP style metrics"):
            for variant in variants:
                output_path = experiment.output_path_for_variant(row, variant)
                if not output_path.exists():
                    continue
                generated = self.image_embedding(output_path)
                style_image = self.image_embedding(row["style_path"])
                evaluation_prompt = self.content_text_prompt(row)
                prompt_text = self.text_embedding(evaluation_prompt)
                s_txt = self.cosine_score(generated, prompt_text)
                s_img = self.cosine_score(generated, style_image)
                s_harmonic = self.harmonic_score(s_txt, s_img)
                rows.append(
                    {
                        "pair_id": int(row["pair_id"]),
                        "content_id": row["content_id"],
                        "style_id": row["style_id"],
                        "content_object": row["content_object"],
                        "style_object": row["style_object"],
                        "style_label": row["style_label"],
                        "variant": variant,
                        "variant_label": experiment.variant_labels[variant].replace("\n", " "),
                        "evaluation_prompt": evaluation_prompt,
                        "S_txt": s_txt,
                        "S_img": s_img,
                        "S_harmonic": s_harmonic,
                        "output_path": str(output_path),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def _evaluation_prompt(content: dict[str, str], style: dict[str, str]) -> str:
        descriptor = style["style_descriptor"].strip()
        suffix = descriptor if descriptor.lower().endswith("style") else f"{descriptor} style"
        content_prompt = content["content_prompt"]
        superclass = content["superclass"]
        return f"{content_prompt}, {superclass}, in {suffix}"

    def compute_content_ortho_metrics(
        self,
        *,
        project_root: Path,
        generated_root: Path,
        prompts_csv: Path | None = None,
        styles_csv: Path | None = None,
    ) -> pd.DataFrame:
        """Compute FineStyle S_txt, S_img, and their harmonic mean."""
        project_root = Path(project_root).resolve()
        generated_root = Path(generated_root).resolve()
        prompts_csv = prompts_csv or project_root / "prompts/content_prompts_190.csv"
        styles_csv = styles_csv or project_root / "styles/quantitative_eval_styles_10.csv"
        contents = self._read_csv(prompts_csv)
        styles = self._read_csv(styles_csv)
        rows = []
        total = len(contents) * len(styles)

        for style in tqdm(styles, desc="CLIP metrics by style", unit="style"):
            style_path = (project_root / style["style_reference_image"]).resolve()
            if not style_path.is_file():
                raise FileNotFoundError(style_path)
            style_embedding = self.image_embedding(style_path)
            for content in tqdm(contents, desc=style["style_id"], unit="image", leave=False):
                case_dir = generated_root / style["style_id"] / content["content_id"]
                generated_path = case_dir / "generated.png"
                metadata_path = case_dir / "metadata.json"
                if not generated_path.is_file():
                    continue
                prompt = self._evaluation_prompt(content, style)
                generated_embedding = self.image_embedding(generated_path)
                s_txt = self.cosine_score(generated_embedding, self.text_embedding(prompt))
                s_img = self.cosine_score(generated_embedding, style_embedding)
                metadata = {}
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "content_id": content["content_id"],
                        "style_id": style["style_id"],
                        "content_prompt": content["content_prompt"],
                        "evaluation_prompt": prompt,
                        "style_name": style["style_name"],
                        "S_txt": s_txt,
                        "S_img": s_img,
                        "S_harmonic": self.harmonic_score(s_txt, s_img),
                        "generated_path": str(generated_path),
                        "style_reference_path": str(style_path),
                        "seed": metadata.get("seed"),
                        "inject_step": metadata.get("inject_step"),
                    }
                )
        print(f"Found {len(rows)} / {total} generated images")
        return pd.DataFrame(rows)

    def save_content_ortho_metrics(self, metrics_df: pd.DataFrame):
        if metrics_df.empty:
            raise ValueError("No generated.png files found for CLIP evaluation")
        detail_path = self.output_dir / "content_ortho_clip_metrics.csv"
        metrics_df.to_csv(detail_path, index=False)
        metric_columns = ["S_txt", "S_img", "S_harmonic"]
        by_style = metrics_df.groupby(["style_id", "style_name"], as_index=False)[metric_columns].mean()
        by_style.insert(2, "num_images", metrics_df.groupby(["style_id", "style_name"]).size().values)
        by_style_path = self.output_dir / "content_ortho_clip_metrics_by_style.csv"
        by_style.to_csv(by_style_path, index=False)
        overall = pd.DataFrame([{
            "num_images": len(metrics_df),
            **{name: float(metrics_df[name].mean()) for name in metric_columns},
        }])
        overall_path = self.output_dir / "content_ortho_clip_metrics_overall.csv"
        overall.to_csv(overall_path, index=False)
        print("saved metrics:", detail_path)
        print("saved style summary:", by_style_path)
        print("saved overall summary:", overall_path)
        return metrics_df, by_style, overall

    def compute_content_leakage_metrics(self, experiment, variants: list[str] | None = None) -> pd.DataFrame:
        variants = experiment.variant_order if variants is None else variants
        rows = []
        for row in tqdm(experiment.pair_rows, desc="CLIP content/leakage metrics"):
            for variant in variants:
                output_path = experiment.output_path_for_variant(row, variant)
                if not output_path.exists():
                    continue
                generated = self.image_embedding(output_path)
                content_text = self.text_embedding(self.content_text_prompt(row))
                style_object = self.text_embedding(self.style_object_text_prompt(row))
                c_txt = self.cosine_score(generated, content_text)
                leak = self.cosine_score(generated, style_object)
                rows.append(
                    {
                        "pair_id": int(row["pair_id"]),
                        "content_id": row["content_id"],
                        "style_id": row["style_id"],
                        "content_object": row["content_object"],
                        "style_object": row["style_object"],
                        "style_label": row["style_label"],
                        "variant": variant,
                        "variant_label": experiment.variant_labels[variant].replace("\n", " "),
                        "content_text_prompt": self.content_text_prompt(row),
                        "style_object_text_prompt": self.style_object_text_prompt(row),
                        "C_txt": c_txt,
                        "Leak": leak,
                        "ContentMargin": c_txt - leak,
                        "output_path": str(output_path),
                    }
                )
        return pd.DataFrame(rows)

    def save_style_metrics(self, metrics_df: pd.DataFrame):
        return self._save_metrics(
            metrics_df,
            csv_name="notebook15_style_metrics.csv",
            summary_name="notebook15_style_metrics_summary.csv",
            plot_name="notebook15_style_metrics_summary.png",
            value_columns=["S_txt", "S_img", "S_harmonic"],
            title="Notebook 15 style metrics by variant",
        )

    def save_content_leakage_metrics(self, metrics_df: pd.DataFrame):
        return self._save_metrics(
            metrics_df,
            csv_name="notebook15_content_leakage_metrics.csv",
            summary_name="notebook15_content_leakage_metrics_summary.csv",
            plot_name="notebook15_content_leakage_metrics_summary.png",
            value_columns=["C_txt", "Leak", "ContentMargin"],
            title="Notebook 15 content preservation and style-object leakage",
            zero_line=True,
        )

    def _save_metrics(
        self,
        metrics_df: pd.DataFrame,
        *,
        csv_name: str,
        summary_name: str,
        plot_name: str,
        value_columns: list[str],
        title: str,
        zero_line: bool = False,
    ):
        import matplotlib.pyplot as plt
        import numpy as np

        if metrics_df.empty:
            print("No generated outputs found for metric evaluation.")
            return metrics_df, pd.DataFrame()

        csv_path = self.output_dir / csv_name
        metrics_df.to_csv(csv_path, index=False)
        print("saved metrics:", csv_path)

        summary = metrics_df.groupby(["variant", "variant_label"], as_index=False)[value_columns].mean()
        summary_path = self.output_dir / summary_name
        summary.to_csv(summary_path, index=False)
        print("saved summary:", summary_path)

        display(metrics_df)
        display(summary)

        figure, axis = plt.subplots(figsize=(9, 4.6))
        x = np.arange(len(summary))
        width = 0.8 / max(1, len(value_columns))
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(value_columns))
        for offset, metric_name in zip(offsets, value_columns):
            axis.bar(x + offset, summary[metric_name], width=width, label=metric_name)
        if zero_line:
            axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xticks(x)
        axis.set_xticklabels(summary["variant_label"], rotation=12, ha="right")
        y_min = min(-0.05 if zero_line else 0.0, float(summary[value_columns].min().min()) * 1.15)
        y_max = max(0.35, float(summary[value_columns].max().max()) * 1.15)
        axis.set_ylim(y_min, y_max)
        axis.set_ylabel("CLIP cosine similarity / margin")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        plot_path = self.output_dir / plot_name
        figure.savefig(plot_path, dpi=180, bbox_inches="tight")
        print("saved metric plot:", plot_path)
        plt.show()
        plt.close(figure)
        return metrics_df, summary
