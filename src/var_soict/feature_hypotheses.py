"""Feature-level hypotheses used by the Infinity style-transfer experiments.

All functions operate on cumulative VAE features shaped ``[B, C, T, H, W]``
(or ``[B, C, H, W]``).  Keeping these operations outside the sampler makes the
inference code responsible only for *when* an intervention is applied.
"""

from __future__ import annotations

import torch


def _feature_matrix(feature: torch.Tensor) -> torch.Tensor:
    return feature.detach().float().reshape(feature.shape[0], feature.shape[1], -1)


def truncated_svd(feature: torch.Tensor, *, rank: int | None = None, alpha: float = 1.0) -> torch.Tensor:
    """Return the exponentially weighted low-rank reconstruction of a feature."""
    original_dtype = feature.dtype
    matrices = _feature_matrix(feature)
    outputs = []
    for matrix in matrices:
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        used_rank = singular_values.numel() if rank is None else min(int(rank), singular_values.numel())
        weights = torch.exp(-float(alpha) * torch.arange(used_rank, device=matrix.device, dtype=matrix.dtype))
        reconstruction = (u[:, :used_rank] * (singular_values[:used_rank] * weights).unsqueeze(0)) @ vh[:used_rank]
        outputs.append(reconstruction.reshape(feature.shape[1:]))
    return torch.stack(outputs).to(dtype=original_dtype)


def principal_feature_blend(
    generation_feature: torch.Tensor,
    style_feature: torch.Tensor,
    *,
    rank: int | None = None,
    alpha: float = 1.0,
    strength: float = 1.0,
) -> torch.Tensor:
    """Replace the selected principal component of generation with style."""
    if generation_feature.shape != style_feature.shape:
        raise ValueError(f"Feature shape mismatch: {generation_feature.shape} vs {style_feature.shape}")
    style_feature = style_feature.to(generation_feature)
    delta = truncated_svd(style_feature, rank=rank, alpha=alpha) - truncated_svd(
        generation_feature, rank=rank, alpha=alpha
    )
    return generation_feature + float(strength) * delta


def remove_content_subspace(
    style_feature: torch.Tensor,
    content_feature: torch.Tensor,
    *,
    content_rank: int = 1,
    projection_strength: float = 1.0,
    preserve_mean: bool = True,
) -> torch.Tensor:
    """Remove content channel directions from a paired style-content residual.

    The basis is estimated from the spatially centered content feature.  The
    style/content mean difference is retained by default because global colour
    and tone are useful style signals.
    """
    if style_feature.shape != content_feature.shape:
        raise ValueError(f"Feature shape mismatch: {style_feature.shape} vs {content_feature.shape}")
    if content_rank < 1:
        raise ValueError("content_rank must be positive")
    if not 0.0 <= projection_strength <= 1.0:
        raise ValueError("projection_strength must be in [0, 1]")

    original_dtype = style_feature.dtype
    style = _feature_matrix(style_feature)
    content = _feature_matrix(content_feature.to(style_feature))
    outputs = []
    for style_matrix, content_matrix in zip(style, content):
        style_mean = style_matrix.mean(dim=1, keepdim=True)
        content_mean = content_matrix.mean(dim=1, keepdim=True)
        centered_content = content_matrix - content_mean
        delta = (style_matrix - style_mean) - centered_content
        u, _, _ = torch.linalg.svd(centered_content, full_matrices=False)
        used_rank = min(int(content_rank), u.shape[1])
        basis = u[:, :used_rank]
        content_projection = basis @ (basis.transpose(0, 1) @ delta)
        residual = delta - float(projection_strength) * content_projection
        if preserve_mean:
            residual = residual + (style_mean - content_mean)
        outputs.append(residual.reshape(style_feature.shape[1:]))
    return torch.stack(outputs).to(dtype=original_dtype)


def content_orthogonal_feature_blend(
    generation_feature: torch.Tensor,
    style_feature: torch.Tensor,
    content_feature: torch.Tensor,
    *,
    style_rank: int | None = 1,
    content_rank: int = 1,
    alpha: float = 1.0,
    strength: float = 1.0,
    projection_strength: float = 1.0,
    preserve_mean: bool = True,
) -> torch.Tensor:
    """Inject the low-rank part of style that is orthogonal to content."""
    residual = remove_content_subspace(
        style_feature,
        content_feature,
        content_rank=content_rank,
        projection_strength=projection_strength,
        preserve_mean=preserve_mean,
    )
    residual = truncated_svd(residual, rank=style_rank, alpha=alpha)
    return generation_feature + float(strength) * residual.to(generation_feature)


def estimate_content_basis(
    content_feature: torch.Tensor,
    *,
    content_rank: int | None = 1,
    variance_threshold: float | None = None,
) -> tuple[list[torch.Tensor], list[dict[str, float | int]]]:
    """Estimate per-sample channel bases from centered content activations.

    ``content_rank`` gives a fixed-rank baseline.  When ``variance_threshold``
    is supplied, the smallest rank reaching that cumulative energy is used,
    capped by ``content_rank`` when both are supplied.  Degenerate one-token
    features return an empty basis rather than an arbitrary SVD direction.
    """
    if content_rank is not None and content_rank < 1:
        raise ValueError("content_rank must be positive or None")
    if variance_threshold is not None and not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")
    if content_rank is None and variance_threshold is None:
        raise ValueError("Specify content_rank, variance_threshold, or both")

    bases, diagnostics = [], []
    for content_matrix in _feature_matrix(content_feature):
        centered = content_matrix - content_matrix.mean(dim=1, keepdim=True)
        u, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
        energy = singular_values.square()
        total_energy = energy.sum()
        if centered.shape[1] <= 1 or float(total_energy) <= torch.finfo(centered.dtype).eps:
            used_rank = 0
        elif variance_threshold is None:
            used_rank = min(int(content_rank), singular_values.numel())
        else:
            cumulative = energy.cumsum(0) / total_energy
            threshold_rank = int(torch.searchsorted(cumulative, variance_threshold).item()) + 1
            used_rank = threshold_rank if content_rank is None else min(threshold_rank, int(content_rank))
        basis = u[:, :used_rank]
        explained = 0.0 if used_rank == 0 else float(energy[:used_rank].sum() / total_energy)
        gap = 0.0
        if 0 < used_rank < singular_values.numel():
            gap = float(singular_values[used_rank - 1] / singular_values[used_rank].clamp_min(1e-12))
        bases.append(basis)
        diagnostics.append({"rank": used_rank, "explained_variance": explained, "spectral_gap": gap})
    return bases, diagnostics


def projected_pfb_content_blend(
    generation_feature: torch.Tensor,
    style_feature: torch.Tensor,
    content_feature: torch.Tensor,
    *,
    style_rank: int | None = 1,
    content_rank: int | None = 1,
    content_variance_threshold: float | None = None,
    alpha: float = 1.0,
    strength: float = 1.0,
    projection_strength: float = 1.0,
    preserve_mean: bool = False,
    return_diagnostics: bool = False,
):
    """Apply PFB after suppressing its component in an estimated content subspace.

    Implements ``Fg + strength * (I - eta Pc) [Phi(Fs) - Phi(Fg)]``.
    The projector ``Pc`` is estimated only from centered ``Fc``.  With the
    default ``preserve_mean=False`` no separate mean/palette term is added, so
    ``projection_strength=0`` exactly recovers ordinary PFB.  The optional
    mean term is passed through the same projector to retain orthogonality.
    """
    if generation_feature.shape != style_feature.shape or generation_feature.shape != content_feature.shape:
        raise ValueError(
            f"Feature shape mismatch: generation={generation_feature.shape}, "
            f"style={style_feature.shape}, content={content_feature.shape}"
        )
    if not 0.0 <= projection_strength <= 1.0:
        raise ValueError("projection_strength must be in [0, 1]")

    style_feature = style_feature.to(generation_feature)
    content_feature = content_feature.to(generation_feature)
    delta = truncated_svd(style_feature, rank=style_rank, alpha=alpha) - truncated_svd(
        generation_feature, rank=style_rank, alpha=alpha
    )
    if preserve_mean:
        raw_delta = _feature_matrix(style_feature - generation_feature)
        mean_delta = raw_delta.mean(dim=2, keepdim=True).expand_as(raw_delta)
        delta = delta + mean_delta.reshape_as(delta).to(delta)
    delta_matrices = _feature_matrix(delta)
    bases, diagnostics = estimate_content_basis(
        content_feature,
        content_rank=content_rank,
        variance_threshold=content_variance_threshold,
    )
    projected = []
    for delta_matrix, basis, diagnostic in zip(delta_matrices, bases, diagnostics):
        content_component = basis @ (basis.transpose(0, 1) @ delta_matrix)
        edit = delta_matrix - float(projection_strength) * content_component
        input_norm = delta_matrix.norm().clamp_min(1e-12)
        diagnostic["removed_fraction"] = float(content_component.norm() / input_norm)
        diagnostic["orthogonality_residual"] = (
            0.0 if basis.shape[1] == 0 else float((basis.transpose(0, 1) @ edit).norm() / input_norm)
        )
        projected.append(edit.reshape(generation_feature.shape[1:]))
    projected_delta = torch.stack(projected).to(generation_feature)
    output = generation_feature + float(strength) * projected_delta
    return (output, diagnostics) if return_diagnostics else output


def select_feature(features: list[torch.Tensor] | tuple[torch.Tensor, ...], step: int, name: str) -> torch.Tensor:
    if not 0 <= int(step) < len(features):
        raise ValueError(f"{name} does not contain step {step}; received {len(features)} steps")
    return features[int(step)]
