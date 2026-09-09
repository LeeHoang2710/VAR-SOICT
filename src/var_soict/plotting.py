from __future__ import annotations

import shutil
import textwrap
from pathlib import Path


def safe_name(text) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(text)).strip("_").lower()


def image_tensor_to_pil(image):
    import torch
    from PIL import Image

    tensor = image.detach().float().cpu() if torch.is_tensor(image) else image
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()
    return Image.fromarray(array)


def save_image_tensor(image, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_tensor_to_pil(image).save(path)


def plot_image(axis, image) -> None:
    axis.imshow(image[0].detach().float().clamp(0, 1).permute(1, 2, 0).numpy())
    axis.axis("off")


def show_variant_session(session, cases, case_images, variant_name, save_path, get_style_image) -> None:
    import matplotlib.pyplot as plt

    columns = ["style reference"] + [case["prompt"] for case in cases if case["session_id"] == session["session_id"]]
    figure, axes = plt.subplots(1, len(columns), figsize=(3.2 * len(columns), 4.2), squeeze=False)
    axes = axes[0]
    plot_image(axes[0], get_style_image(session["style_path"]))
    axes[0].set_title(f"style reference\n{session['style_label']}", fontsize=10)
    session_cases = [case for case in cases if case["session_id"] == session["session_id"]]
    for column_id, case in enumerate(session_cases, start=1):
        plot_image(axes[column_id], case_images[case["case_id"]])
        axes[column_id].set_title(f'"{textwrap.fill(case["prompt"], width=26)}"', fontsize=9)
    figure.suptitle(f"{variant_name} | {session['name']}", fontsize=14)
    figure.tight_layout()
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    print("saved:", save_path)
    plt.show()
    plt.close(figure)


def show_final_aggregate_session(session, cases, aggregate_columns, save_path, get_style_image) -> None:
    import matplotlib.pyplot as plt

    session_cases = [case for case in cases if case["session_id"] == session["session_id"]]
    figure, axes = plt.subplots(
        len(session_cases),
        len(aggregate_columns),
        figsize=(3.35 * len(aggregate_columns), 3.75 * len(session_cases)),
        squeeze=False,
    )
    for row_id, case in enumerate(session_cases):
        for column_id, (label, results) in enumerate(aggregate_columns):
            axis = axes[row_id, column_id]
            if results is None:
                plot_image(axis, get_style_image(session["style_path"]))
                axis.set_title(label, fontsize=11, pad=10)
                axis.text(
                    0.5,
                    -0.10,
                    f'"{textwrap.fill(case["prompt"], width=36)}"',
                    ha="center",
                    va="top",
                    transform=axis.transAxes,
                    fontsize=10,
                    clip_on=False,
                )
            else:
                plot_image(axis, results[case["case_id"]])
                axis.set_title(label, fontsize=11, pad=10)

    figure.suptitle(f"Infinity-2B final aggregate comparison | {session['name']}", fontsize=14, y=0.995)
    figure.subplots_adjust(left=0.02, right=0.995, top=0.92, bottom=0.08, wspace=0.18, hspace=0.52)
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    print("saved:", save_path)
    plt.show()
    plt.close(figure)


def download_output_zip(output_dir, zip_path=None) -> Path:
    output_dir = Path(output_dir)
    zip_path = Path(zip_path) if zip_path is not None else output_dir.with_suffix(".zip")
    if not output_dir.exists():
        raise FileNotFoundError(f"Output folder not found: {output_dir}")

    shutil.make_archive(
        str(zip_path.with_suffix("")),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    try:
        from google.colab import files

        files.download(str(zip_path))
    except Exception:
        print("ZIP ready:", zip_path)
    return zip_path

