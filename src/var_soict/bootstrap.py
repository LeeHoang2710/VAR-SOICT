from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .config import DEPENDENCY_PACKAGES, ExperimentConfig, ModelBundle, ModelFiles, RuntimePaths
from .t5_streaming import load_t5_encoder_streaming


def build_runtime_paths(config: ExperimentConfig) -> RuntimePaths:
    runtime_root = Path("/content") if Path("/content").exists() else Path.cwd()
    root = Path(config.root)
    if not Path("/content").exists() and str(root).startswith("/content/"):
        root = runtime_root / ".runtime" / root.name
    port_dir = root / "gguf_port"
    official_dir = port_dir / "Infinity"
    asset_dir = root / "assets"
    output_dir = runtime_root / "Infinity_outputs" / config.output_run_name
    paths = RuntimePaths(
        root=root,
        port_dir=port_dir,
        official_dir=official_dir,
        asset_dir=asset_dir,
        runtime_root=runtime_root,
        output_dir=output_dir,
        baseline_dir=output_dir / "baseline",
        variant_dir=output_dir / "variants",
        aggregate_dir=output_dir / "aggregate",
    )
    for path in (
        paths.port_dir,
        paths.asset_dir,
        paths.output_dir,
        paths.baseline_dir,
        paths.variant_dir,
        paths.aggregate_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def check_torch_runtime(*, require_cuda: bool = True) -> str:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU:", props.name)
        print("VRAM GiB:", round(props.total_memory / 2**30, 2))
    elif require_cuda:
        raise RuntimeError("A CUDA GPU is required for practical inference. Select a GPU runtime and rerun.")
    else:
        print("WARNING: no GPU detected; inference will be extremely slow.")
    return "cuda" if torch.cuda.is_available() else "cpu"


def install_dependencies(packages: list[str] | None = None) -> None:
    packages = DEPENDENCY_PACKAGES if packages is None else packages
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    print("Dependency installation finished.")


def _download_hf_file(repo_id: str, filename: str, target_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    target_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(target_dir)))


def prepare_infinity_sources(config: ExperimentConfig, paths: RuntimePaths) -> ModelFiles:
    if not paths.official_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", config.official_repo, str(paths.official_dir)], check=True)
    else:
        print("Official Infinity source already exists:", paths.official_dir)

    port_script = _download_hf_file(config.gguf_repo, "generate_image_2b_q8_gguf.py", paths.port_dir)
    port_utils = _download_hf_file(config.gguf_repo, "infinity_gguf_utils.py", paths.port_dir)
    patch_dir = paths.root / "gguf_patched_source"
    patched_basic = _download_hf_file(config.gguf_repo, "Infinity/infinity/models/basic.py", patch_dir)
    patched_infinity = _download_hf_file(config.gguf_repo, "Infinity/infinity/models/infinity.py", patch_dir)

    official_basic = paths.official_dir / "infinity" / "models" / "basic.py"
    official_infinity = paths.official_dir / "infinity" / "models" / "infinity.py"
    shutil.copy2(patched_basic, official_basic)
    shutil.copy2(patched_infinity, official_infinity)

    infinity_source = official_infinity.read_text()
    old_attention_guard = (
        "customized_kernel_installed = any('Infinity' in arg_name for arg_name in "
        "flash_attn_func.__code__.co_varnames)"
    )
    new_attention_guard = (
        "customized_kernel_installed = flash_attn_func is not None and any('Infinity' in "
        "arg_name for arg_name in flash_attn_func.__code__.co_varnames)"
    )
    if old_attention_guard not in infinity_source and new_attention_guard not in infinity_source:
        raise RuntimeError("Expected optional-attention guard was not found in patched infinity.py")
    if old_attention_guard in infinity_source:
        official_infinity.write_text(infinity_source.replace(old_attention_guard, new_attention_guard, 1))

    infinity_gguf = _download_hf_file(config.gguf_repo, "infinity_2b_reg_Q8_0.gguf", paths.asset_dir)
    t5_gguf = _download_hf_file(config.gguf_repo, "flan-t5-xl-encoder-Q8_0.gguf", paths.asset_dir)
    vae_path = _download_hf_file(config.gguf_repo, "Infinity/infinity_vae_d32_reg.pth", paths.asset_dir)
    files = ModelFiles(port_script, port_utils, patched_basic, patched_infinity, infinity_gguf, t5_gguf, vae_path)

    print("GGUF model:", files.infinity_gguf)
    print("T5 encoder:", files.t5_gguf)
    print("VAE:", files.vae_path)
    print("Loader:", files.port_script)
    print("Loader utility:", files.port_utils)
    return files


def verify_model_files(paths: RuntimePaths, files: ModelFiles) -> None:
    required_files = [
        files.port_script,
        files.port_utils,
        files.patched_basic,
        files.patched_infinity,
        files.infinity_gguf,
        files.t5_gguf,
        files.vae_path,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    for path in required_files:
        print(f"{path.name:40s} {path.stat().st_size / 2**30:.3f} GiB")
    assert (paths.official_dir / "infinity" / "models" / "infinity.py").exists(), "Official Infinity source is incomplete."
    print("All GGUF, VAE, and official source files are present.")


def import_gguf_loader(paths: RuntimePaths, files: ModelFiles):
    sys.path.insert(0, str(paths.port_dir))
    sys.path.insert(0, str(paths.official_dir))

    loader_source = files.port_script.read_text()
    compat_pattern = r"\n    # Apply NumPy 2\.0 compatibility patch.*?\n    # Load GGUF state dict"
    loader_source, replacements = re.subn(
        compat_pattern,
        "\n    # NumPy compatibility is handled by the installed gguf package.\n    # Load GGUF state dict",
        loader_source,
        count=1,
        flags=re.S,
    )
    print("Removed obsolete NumPy compatibility block:", replacements == 1)

    patched_loader = paths.port_dir / "generate_image_2b_q8_gguf_colab.py"
    patched_loader.write_text(loader_source)
    spec = importlib.util.spec_from_file_location("infinity_gguf_colab_loader", patched_loader)
    gguf_loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gguf_loader
    spec.loader.exec_module(gguf_loader)
    print("Custom GGUF loader imported successfully.")
    return gguf_loader


def load_model_bundle(config: ExperimentConfig, files: ModelFiles, gguf_loader) -> ModelBundle:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("A CUDA GPU is required for practical inference. Select a GPU runtime and rerun.")

    print("[1/4] Loading T5 tokenizer...")
    text_tokenizer = gguf_loader.load_t5_tokenizer_from_gguf(str(files.t5_gguf))

    print(f"[2/4] Streaming quantized T5 encoder to {config.t5_device}...")
    text_encoder = load_t5_encoder_streaming(
        files.t5_gguf,
        device=config.t5_device,
        gguf_loader=gguf_loader,
        torch_module=torch,
    )

    print("[3/4] Loading VAE on GPU...")
    vae = gguf_loader.load_vae(str(files.vae_path), vae_type=32, device=device)

    print("[4/4] Loading quantized Infinity-2B transformer on GPU...")
    infinity_model = gguf_loader.load_infinity_from_gguf(
        str(files.infinity_gguf),
        vae=vae,
        device=device,
        model_type="infinity_2b",
        text_channels=2048,
        pn=config.model_pn,
    )

    infinity_model.eval()
    vae.eval()
    print("All components loaded successfully.")
    return ModelBundle(
        device=device,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        infinity_model=infinity_model,
        scale_schedule=[],
    )


def build_scale_schedule(model_pn: str, *, aspect_ratio: float = 1.0) -> list[tuple[int, int, int]]:
    import numpy as np
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - aspect_ratio))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][model_pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    print("Aspect ratio:", h_div_w_template)
    print("Preset:", model_pn)
    print("Scale schedule:", scale_schedule)
    return scale_schedule
