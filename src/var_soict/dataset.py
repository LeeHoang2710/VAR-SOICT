from __future__ import annotations

import csv
import random
from pathlib import Path

from .config import ExperimentConfig


def find_var_soict_root(runtime_root: Path | None = None, start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    runtime_root = Path("/content") if runtime_root is None and Path("/content").exists() else runtime_root
    candidates = [start, *start.parents[:4]]
    if runtime_root is not None:
        runtime_root = Path(runtime_root)
        candidates.extend([runtime_root / "VAR_SOICT", runtime_root / "Image Generation" / "VAR_SOICT"])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "styles" / "quantitative_eval_styles_10.csv").exists()
            and (candidate / "prompts" / "content_prompts_190.csv").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find VAR_SOICT. Run from the VAR_SOICT folder, "
        "or copy VAR_SOICT to /content before running the notebook."
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def styled_prompt(content: dict[str, str], style: dict[str, str]) -> str:
    descriptor = style["style_descriptor"].strip()
    if descriptor.lower().endswith("style"):
        return f"{content['content_prompt']}, {content['superclass']}, in {descriptor}"
    return f"{content['content_prompt']}, {content['superclass']}, in {descriptor} style"


def load_random_eval_cases(
    *,
    var_soict_root: Path,
    output_dir: Path,
    config: ExperimentConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Path]:
    var_soict_root = Path(var_soict_root)
    quant_style_csv = var_soict_root / "styles" / "quantitative_eval_styles_10.csv"
    content_prompt_csv = var_soict_root / "prompts" / "content_prompts_190.csv"
    style_rows = read_csv_rows(quant_style_csv)
    content_rows = read_csv_rows(content_prompt_csv)

    if len(style_rows) != config.expected_eval_styles:
        raise RuntimeError(f"Expected {config.expected_eval_styles} quantitative styles, found {len(style_rows)}.")
    if len(content_rows) != 190:
        raise RuntimeError(f"Expected 190 Parti content prompts, found {len(content_rows)}.")
    if len(content_rows) < config.prompts_per_style:
        raise RuntimeError(f"Need at least {config.prompts_per_style} prompts to sample per style.")

    sessions: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    rng = random.Random(config.prompt_sample_seed)

    for session_id, style in enumerate(style_rows):
        style_path = var_soict_root / style["style_reference_image"]
        if not style_path.exists():
            raise FileNotFoundError(style_path)

        sampled_prompts = rng.sample(content_rows, k=config.prompts_per_style)
        session_cases = []
        session = {
            "session_id": session_id,
            "name": f"Figure 10 style {int(style['figure10_index']):02d}: {style['style_name']}",
            "style_path": style_path,
            "style_label": style["style_name"],
            "style_id": style["style_id"],
            "figure10_index": int(style["figure10_index"]),
            "prompts": [],
        }

        for prompt_id, content in enumerate(sampled_prompts):
            prompt = styled_prompt(content, style)
            case = {
                "case_id": f"{style['style_id']}_p{prompt_id + 1:02d}",
                "session_id": session_id,
                "session_name": session["name"],
                "style_path": style_path,
                "style_label": style["style_name"],
                "style_id": style["style_id"],
                "figure10_index": int(style["figure10_index"]),
                "content_id": content["content_id"],
                "content_prompt": content["content_prompt"],
                "category": content["category"],
                "superclass": content["superclass"],
                "prompt": prompt,
            }
            session["prompts"].append(prompt)
            session_cases.append(case)
            cases.append(case)
            manifest_rows.append(
                {
                    "case_id": case["case_id"],
                    "style_id": case["style_id"],
                    "figure10_index": case["figure10_index"],
                    "style_name": case["style_label"],
                    "content_id": case["content_id"],
                    "content_prompt": case["content_prompt"],
                    "superclass": case["superclass"],
                    "prompt": case["prompt"],
                    "style_reference_image": style["style_reference_image"],
                }
            )

        session["cases"] = session_cases
        sessions.append(session)

    if len(cases) != config.expected_cases_per_variant:
        raise RuntimeError(f"Expected {config.expected_cases_per_variant} cases, found {len(cases)}.")

    selected_cases_csv = Path(output_dir) / "selected_cases_30.csv"
    selected_cases_csv.parent.mkdir(parents=True, exist_ok=True)
    with selected_cases_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    return sessions, cases, selected_cases_csv

