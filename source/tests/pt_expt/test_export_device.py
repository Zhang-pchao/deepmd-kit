# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph export must select operators for the destination, not the trace host."""

import os

import pytest
import torch

from deepmd.pt_expt.utils import env, serialization


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("level", [None, "0", "1", "2"])
@pytest.mark.parametrize("lower_kind", ["graph", "dpa1_canonical", "dpa4c_canonical"])
def test_graph_export_uses_target_device(monkeypatch, device, level, lower_kind):
    monkeypatch.setattr(env, "DEVICE", torch.device(device))
    monkeypatch.setattr(
        serialization, "_resolve_target_lower_kind", lambda *args: args[-1]
    )
    monkeypatch.setattr(
        serialization, "_uses_dpa4_kernel_defaults", lambda *args: False
    )
    if level is None:
        monkeypatch.delenv("DP_CUDA_INFER", raising=False)
    else:
        monkeypatch.setenv("DP_CUDA_INFER", level)
    observed = []

    def trace(*args, **kwargs):
        observed.append(os.environ.get("DP_CUDA_INFER"))

    monkeypatch.setattr(serialization, "_deserialize_to_file_pt2", trace)
    serialization.deserialize_to_file(
        "unused.pt2", {"model": {}}, lower_kind=lower_kind
    )
    assert observed == [level if device == "cpu" else "2"]
    assert os.environ.get("DP_CUDA_INFER") == level


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("level", [None, "0", "2"])
def test_graph_export_restores_level_after_error(monkeypatch, device, level):
    monkeypatch.setattr(env, "DEVICE", torch.device(device))
    monkeypatch.setattr(
        serialization, "_resolve_target_lower_kind", lambda *args: args[-1]
    )
    monkeypatch.setattr(
        serialization, "_uses_dpa4_kernel_defaults", lambda *args: False
    )
    if level is None:
        monkeypatch.delenv("DP_CUDA_INFER", raising=False)
    else:
        monkeypatch.setenv("DP_CUDA_INFER", level)

    def trace(*args, **kwargs):
        raise RuntimeError("trace failed")

    monkeypatch.setattr(serialization, "_deserialize_to_file_pt2", trace)
    with pytest.raises(RuntimeError, match="trace failed"):
        serialization.deserialize_to_file(
            "unused.pt2", {"model": {}}, lower_kind="graph"
        )
    assert os.environ.get("DP_CUDA_INFER") == level


@pytest.mark.parametrize("level", [None, "0", "1", "2"])
def test_dense_export_preserves_requested_level(monkeypatch, level):
    if level is None:
        monkeypatch.delenv("DP_CUDA_INFER", raising=False)
    else:
        monkeypatch.setenv("DP_CUDA_INFER", level)
    observed = []

    def trace(*args, **kwargs):
        observed.append(os.environ.get("DP_CUDA_INFER"))

    monkeypatch.setattr(serialization, "_deserialize_to_file_pt2", trace)
    serialization.deserialize_to_file("unused.pt2", {"model": {}}, lower_kind="nlist")
    assert observed == [level]
    assert os.environ.get("DP_CUDA_INFER") == level
