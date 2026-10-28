# SPDX-License-Identifier: LGPL-3.0-or-later
"""CUDA fitting values and derivatives across activation tails."""

import pytest
import torch

import deepmd.pt  # noqa: F401 - register the native operator library

from deepmd.pt_expt.kernels.graph_fitting import (
    ensure_registered,
    op_available,
)

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or not op_available(),
    reason="CUDA and the compiled fitting operators are required",
)


@_CUDA
@pytest.mark.parametrize("activation", [0, 1])
@pytest.mark.parametrize("tile", [0, 3])
def test_fitting_activation_tails_match_double_reference(
    activation: int, tile: int
) -> None:
    """A one-layer fitting network retains small negative SiLU tails.

    Tight relative checks on negative inputs catch cancellation that an
    absolute tolerance over a random network can hide. The same reference
    covers the separate forward/backward and fused inference entry points.
    """
    ensure_registered()
    values = [-30.0, -20.0, -18.0, -16.0, -10.0, -2.0, 0.0, 2.0, 10.0, 20.0, 30.0]
    x = torch.tensor(values, device="cuda", dtype=torch.float32).reshape(-1, 1)
    atype = torch.zeros(len(values), device="cuda", dtype=torch.int64)
    weights = [torch.ones((1, 4), device="cuda")]
    biases = [torch.zeros(4, device="cuda")]
    head = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
    head_bias = torch.zeros(1, device="cuda")
    atom_bias = torch.zeros(1, device="cuda", dtype=torch.float64)
    seed = torch.ones(len(values), device="cuda", dtype=torch.float64)

    reference_x = x.double().requires_grad_(True)
    reference = (
        torch.tanh(reference_x)
        if activation == 0
        else torch.nn.functional.silu(reference_x)
    )
    (reference_gradient,) = torch.autograd.grad(reference.sum(), reference_x)
    energy, saved = torch.ops.deepmd.graph_fitting(
        x, atype, weights, biases, [0], head, head_bias, atom_bias, activation
    )
    gradient = torch.ops.deepmd.graph_fitting_backward(
        seed[:, None], saved, weights, biases, [0], head, activation
    )
    fused_gradient = x.clone()
    fused_energy = torch.ops.deepmd.graph_fitting_energy_gradient(
        fused_gradient,
        atype,
        weights,
        biases,
        [0],
        head,
        head_bias,
        atom_bias,
        activation,
        seed,
        tile,
    )
    torch.testing.assert_close(energy, fused_energy, atol=0, rtol=0)
    torch.testing.assert_close(gradient, fused_gradient, atol=0, rtol=0)
    torch.testing.assert_close(energy, reference.detach(), atol=1e-12, rtol=3e-6)
    # tanh's derivative can round to zero in float32 saturation; SiLU's
    # negative tail must retain its nonzero derivative.
    gradient_atol = 1e-7 if activation == 0 else 1e-12
    torch.testing.assert_close(
        gradient.double(), reference_gradient, atol=gradient_atol, rtol=3e-6
    )
