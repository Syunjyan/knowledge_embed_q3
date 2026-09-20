import torch

from embed.official_physics import (
    apply_dirichlet,
    darcy_pde_match,
    dynamic_weight,
    ns_mean_conservation,
)


def test_dirichlet_projection_zeroes_boundary() -> None:
    field = torch.ones(2, 25)
    projected = apply_dirichlet(field, 5).reshape(2, 5, 5)

    assert torch.count_nonzero(projected[:, 0, :]) == 0
    assert torch.count_nonzero(projected[:, -1, :]) == 0
    assert torch.count_nonzero(projected[:, :, 0]) == 0
    assert torch.count_nonzero(projected[:, :, -1]) == 0
    assert torch.all(projected[:, 1:-1, 1:-1] == 1)


def test_matching_identical_discrete_residuals_is_zero() -> None:
    field = torch.randn(2, 25)
    coefficient = torch.ones(2, 25)
    loss = darcy_pde_match(field, field, coefficient, dx=0.25, s=5)
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_mean_conservation_is_zero_for_identical_fields() -> None:
    field = torch.randn(2, 16)
    loss = ns_mean_conservation(field, field)
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_dynamic_weight_is_capped() -> None:
    weight = dynamic_weight(torch.tensor(10.0), torch.tensor(0.001), cap=0.1)
    assert 0.0 <= float(weight) <= 0.1 + 1e-7
