import torch

from var_soict.feature_hypotheses import (
    principal_feature_blend,
    projected_pfb_content_blend,
)


def _features():
    generator = torch.Generator().manual_seed(7)
    shape = (2, 8, 1, 4, 4)
    return tuple(torch.randn(shape, generator=generator) for _ in range(3))


def test_projection_is_orthogonal_and_shape_preserving():
    generation, style, content = _features()
    output, diagnostics = projected_pfb_content_blend(
        generation,
        style,
        content,
        style_rank=3,
        content_rank=2,
        projection_strength=1.0,
        preserve_mean=False,
        return_diagnostics=True,
    )
    assert output.shape == generation.shape
    assert all(item["orthogonality_residual"] < 1e-5 for item in diagnostics)


def test_zero_projection_strength_recovers_pfb():
    generation, style, content = _features()
    pfb = principal_feature_blend(generation, style, rank=2, alpha=0.7)
    actual = projected_pfb_content_blend(
        generation,
        style,
        content,
        style_rank=2,
        alpha=0.7,
        projection_strength=0.0,
        preserve_mean=False,
    )
    torch.testing.assert_close(actual, pfb)


def test_degenerate_single_token_content_uses_empty_basis():
    generation = torch.randn(1, 8, 1, 1, 1)
    style = torch.randn_like(generation)
    output, diagnostics = projected_pfb_content_blend(
        generation,
        style,
        generation,
        content_rank=3,
        return_diagnostics=True,
    )
    assert torch.isfinite(output).all()
    assert diagnostics[0]["rank"] == 0


def test_variance_threshold_selects_data_dependent_rank():
    generation, style, content = _features()
    _, diagnostics = projected_pfb_content_blend(
        generation,
        style,
        content,
        content_rank=None,
        content_variance_threshold=0.8,
        return_diagnostics=True,
    )
    assert all(item["rank"] >= 1 for item in diagnostics)
    assert all(item["explained_variance"] >= 0.8 for item in diagnostics)
