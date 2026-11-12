# SPDX-License-Identifier: LGPL-3.0-or-later
"""Independent many-body energy and full-coordinate gradient checks."""

from dataclasses import (
    replace,
)

import pytest
import torch

from deepmd.pt_expt.utils.canonical_graph import (
    CanonicalGraph,
    canonical_index_bytes,
    canonical_index_words,
)
from deepmd.pt_expt.utils.selected_energy import (
    affected_one_hop_energy_owners,
    select_one_hop_energy_graph,
)


def graph_from_positions(x, cutoff=2.1, box=None):
    delta = x[None, :, :] - x[:, None, :]
    if box is not None:
        delta = delta - box * torch.round(delta / box)
    active = torch.sum(delta**2, dim=-1) < cutoff**2
    active.fill_diagonal_(False)
    dest, src = torch.nonzero(active, as_tuple=True)
    e, n = src.numel(), x.shape[0]
    source = torch.zeros(max(e, 2), dtype=torch.uint32)
    vec = torch.zeros(max(e, 2), 3, dtype=torch.float32)
    source[:e] = src.to(torch.uint32)
    vec[:e] = delta[dest, src].to(torch.float32)
    order = torch.arange(max(e, 2), dtype=torch.int64)
    order[:e] = torch.argsort(src, stable=True)
    return CanonicalGraph(
        n_node=torch.tensor([n]),
        n_local=torch.tensor([n]),
        source=source,
        edge_vec=vec,
        destination_row_ptr=torch.cat(
            (
                torch.zeros(1, dtype=torch.int64),
                torch.bincount(dest, minlength=n).cumsum(0),
            )
        ),
        source_row_ptr=torch.cat(
            (
                torch.zeros(1, dtype=torch.int64),
                torch.bincount(src, minlength=n).cumsum(0),
            )
        ),
        source_order=order,
    )


def graph_energy_force(graph):
    n, local = int(graph.n_node[0]), int(graph.n_local[0])
    e = int(graph.destination_row_ptr[-1])
    vec = graph.edge_vec[:e].double().detach().requires_grad_()
    dest = torch.repeat_interleave(
        torch.arange(n), torch.diff(graph.destination_row_ptr)
    )
    moments = torch.zeros(n, dtype=torch.float64).index_add(
        0, dest, torch.sum(vec**2, dim=1)
    )
    energy = (0.7 * moments[:local] ** 2 + 1.3).sum()
    edge_gradient = torch.autograd.grad(energy, vec)[0]
    force = torch.zeros(n, 3, dtype=torch.float64)
    force.index_add_(0, dest, edge_gradient)
    force.index_add_(0, canonical_index_words(graph.source)[:e].long(), -edge_gradient)
    return energy.detach(), force


def dense_reference(x, owners, cutoff=2.1, box=None):
    coords = x.double().detach().requires_grad_()
    energy = coords.sum() * 0.0
    for center in owners:
        moment = coords.sum() * 0.0
        for neighbor in range(len(coords)):
            if center == neighbor:
                continue
            delta = coords[neighbor] - coords[center]
            if box is not None:
                delta = delta - box * torch.round(delta / box)
            if float(torch.sum(delta.detach() ** 2)) < cutoff**2:
                moment = moment + torch.sum(delta**2)
        energy = energy + 0.7 * moment**2 + 1.3
    return energy.detach(), -torch.autograd.grad(energy, coords)[0]


@pytest.mark.parametrize("owners", [[], [0], [1, 3], [3, 2, 1, 0]])
@pytest.mark.parametrize("periodic", [False, True])
def test_selected_manybody_energy_and_all_node_forces(owners, periodic):
    # Binary fractions avoid FP32 geometry noise in this independent oracle.
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.5, 0.0, 0.25], [4.0, 0.5, 0.0]]
    )
    box = torch.tensor([5.0, 6.0, 7.0]) if periodic else None
    graph = graph_from_positions(x, box=box)
    selected = select_one_hop_energy_graph(
        graph, torch.tensor(owners, dtype=torch.int64)
    )
    ref_energy, ref_force = dense_reference(x, owners, box=box)
    if not owners:
        assert selected.graph is None
        force = selected.scatter(torch.zeros(0, 3, dtype=torch.float64))
        torch.testing.assert_close(force, ref_force, rtol=0, atol=0)
        assert ref_energy == 0
        return
    energy, force = graph_energy_force(selected.graph)
    torch.testing.assert_close(energy, ref_energy, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        selected.scatter(force), ref_force, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        force.sum(0), torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-12
    )
    if len(owners) == 1:
        assert torch.count_nonzero(force[len(owners) :]) > 0
    e = selected.edge_indices.numel()
    assert torch.equal(
        selected.graph.edge_vec[:e], graph.edge_vec[selected.edge_indices]
    )


def test_partitioned_energy_and_force_sum():
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0], [3.0, 0.0, 0.0]]
    )
    graph = graph_from_positions(x)
    total_e, total_f = graph_energy_force(graph)
    parts = [
        select_one_hop_energy_graph(graph, torch.tensor(o)) for o in ([2, 0], [3, 1])
    ]
    es, fs = [], []
    for part in parts:
        en, force = graph_energy_force(part.graph)
        es.append(en)
        fs.append(part.scatter(force))
    torch.testing.assert_close(sum(es), total_e, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(sum(fs), total_f, rtol=1e-12, atol=1e-12)


def test_isolated_owner_and_storage_guards():
    x = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    selected = select_one_hop_energy_graph(graph_from_positions(x), torch.tensor([1]))
    assert selected.node_indices.tolist() == [1]
    assert selected.graph.source.numel() == 2
    assert selected.graph.destination_row_ptr.tolist() == [0, 0]
    en, force = graph_energy_force(selected.graph)
    assert en == 1.3
    assert torch.count_nonzero(force) == 0


@pytest.mark.parametrize("owners", [[-1], [4], [0, 0]])
def test_invalid_owner_selection_rejected(owners):
    graph = graph_from_positions(torch.arange(12).reshape(4, 3).float())
    with pytest.raises(ValueError):
        select_one_hop_energy_graph(graph, torch.tensor(owners))


def test_ghosts_are_neighbors_but_cannot_own_selected_energy():
    graph = graph_from_positions(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    graph = replace(graph, n_local=torch.tensor([1]))
    selected = select_one_hop_energy_graph(graph, torch.tensor([0]))
    assert selected.node_indices.tolist() == [0, 1]
    with pytest.raises(ValueError, match="owned prefix"):
        select_one_hop_energy_graph(graph, torch.tensor([1]))


def test_incomplete_ownership_and_bad_force_shape_rejected():
    graph = graph_from_positions(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="one complete frame"):
        select_one_hop_energy_graph(
            replace(graph, n_node=torch.tensor([1])), torch.tensor([0])
        )
    selection = select_one_hop_energy_graph(graph, torch.tensor([0]))
    with pytest.raises(ValueError, match="leading row"):
        selection.scatter(torch.zeros(1, 3))


def test_dependency_union_covers_cutoff_crossing_and_zero_displacement():
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.25, 0.0, 0.0], [9.0, 0.0, 0.0]]
    )
    y = x.clone()
    y[1, 0] = 2.25
    graphs = (graph_from_positions(x), graph_from_positions(y))
    # Node 3 does not move, but its nonzero quantum amplitude is support.
    owners = affected_one_hop_energy_owners(graphs, torch.tensor([1, 3]))
    assert owners.tolist() == [0, 1, 2, 3]
    assert (
        affected_one_hop_energy_owners(
            graphs, torch.tensor([], dtype=torch.int64)
        ).numel()
        == 0
    )
    with pytest.raises(ValueError, match="at least one"):
        affected_one_hop_energy_owners((), torch.tensor([0]))
    with pytest.raises(ValueError, match="same node axis"):
        affected_one_hop_energy_owners(
            (graphs[0], replace(graphs[1], n_local=torch.tensor([3]))),
            torch.tensor([0]),
        )


@pytest.mark.parametrize("beads", [1, 2, 4])
@pytest.mark.parametrize("region", ["classical", "quantum", "transition"])
@pytest.mark.parametrize("moving_anchor", [False, True])
def test_reference_local_identity_and_auxiliary_pullback(beads, region, moving_anchor):
    # Independent FP64 many-body potential; no production map/pullback helper.
    c0 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.25, 0.25, 0.0],
            [2.5, 0.0, 0.0],
            [4.0, 0.25, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    rng = torch.Generator().manual_seed(72)
    offsets = 0.03 * torch.randn(beads, 5, 3, generator=rng, dtype=torch.float64)
    offsets -= offsets.mean(0)
    y = (c0 + offsets).requires_grad_()
    c = y.mean(0)
    anchor = c[0] if moving_anchor else torch.zeros(3, dtype=torch.float64)
    if region == "transition":
        radius = torch.linalg.vector_norm(
            c - anchor + torch.tensor([0.0, 0.125, 0.0]), dim=1
        )
        t = ((radius - 0.5) / 2.0).clamp(0, 1)
        a = 1 - 10 * t**3 + 15 * t**4 - 6 * t**5
        a = torch.where(radius >= 2.5, 0.0, a)
    else:
        a = torch.full((5,), float(region == "quantum"), dtype=torch.float64)
    x = c[None] + a[None, :, None] * (y - c[None])
    frames = (c, *x.unbind())
    graphs = tuple(graph_from_positions(frame.detach().float()) for frame in frames)
    support = torch.nonzero(a.detach() > 0).flatten()
    owners = affected_one_hop_energy_owners(graphs, support).tolist()

    def energy(coords, selected):
        delta = coords[None] - coords[:, None]
        r2 = (delta**2).sum(-1)
        mask = (r2.detach() < 2.1**2) & ~torch.eye(5, dtype=torch.bool)
        moment = (((1 - r2 / (2.1**2)).clamp(min=0) ** 3) * mask).sum(1)
        atomic = 0.7 * moment**2 + 1.3
        return atomic[selected].sum() + coords.sum() * 0

    total = sum(energy(frame, list(range(5))) for frame in x) / beads
    reference = energy(c, list(range(5))) - energy(c, owners)
    local_terms = [energy(frame, owners) for frame in x]
    split = reference + sum(local_terms) / beads
    f_full = -torch.autograd.grad(total, y, retain_graph=True)[0]
    f_split = -torch.autograd.grad(split, y, retain_graph=True)[0]
    torch.testing.assert_close(split, total, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(f_split, f_full, rtol=1e-10, atol=1e-10)
    residual = -torch.autograd.grad(reference, c, retain_graph=True)[0]
    assembled = torch.zeros_like(y)
    for frame in x.unbind():
        # Recompute using this tensor view so autograd observes the exact input.
        local = -torch.autograd.grad(energy(frame, owners), frame, retain_graph=True)[0]
        assembled += (
            torch.autograd.grad(
                frame, y, grad_outputs=local + residual, retain_graph=True
            )[0]
            / beads
        )
    torch.testing.assert_close(assembled, f_full, rtol=1e-10, atol=1e-10)
    if beads > 1 and region == "transition":
        # A separate negative control targets only the shared residual. Native
        # P*R(C) and dC/dY=1/P cancel: dividing this force again must fail.
        wrong_residual = torch.zeros_like(y)
        for frame in x.unbind():
            local = -torch.autograd.grad(
                energy(frame, owners), frame, retain_graph=True
            )[0]
            wrong_residual += (
                torch.autograd.grad(
                    frame, y, grad_outputs=local + residual / beads, retain_graph=True
                )[0]
                / beads
            )
        assert torch.max(torch.abs(wrong_residual - f_full)) > 1e-4
    # The analytic equality includes the moving-anchor reaction.
    assert torch.isfinite(assembled).all()

    def independently_mapped_energy(aux):
        center = aux.mean(0)
        origin = center[0] if moving_anchor else torch.zeros(3, dtype=torch.float64)
        if region == "transition":
            rad = torch.linalg.vector_norm(
                center - origin + torch.tensor([0.0, 0.125, 0.0]), dim=1
            )
            u = ((rad - 0.5) / 2.0).clamp(0, 1)
            amp = torch.where(rad >= 2.5, 0.0, 1 - 10 * u**3 + 15 * u**4 - 6 * u**5)
        else:
            amp = torch.full((5,), float(region == "quantum"), dtype=torch.float64)
        physical = center + amp[None, :, None] * (aux - center)
        return sum(energy(frame, list(range(5))) for frame in physical) / beads

    direction = torch.randn(y.shape, generator=rng, dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    derivative = -(f_full * direction).sum().item()
    errors = []
    for step in (1e-3, 2e-4, 4e-5):
        fd = (
            independently_mapped_energy(y.detach() + step * direction)
            - independently_mapped_energy(y.detach() - step * direction)
        ) / (2 * step)
        errors.append(abs(fd.item() - derivative))
    assert errors[-1] < 1e-8, errors
    assert errors[-1] < errors[0] + 1e-9, errors
    if beads > 1 and region == "transition":
        assert torch.max(torch.abs(beads * assembled - f_full)) > 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("owners", [[], [0], [1, 3], [3, 2, 1, 0]])
def test_cuda_topology_selection_and_scatter_matches_cpu(owners):
    from dataclasses import fields

    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.5, 0.0, 0.25], [4.0, 0.5, 0.0]]
    )
    cpu = graph_from_positions(x)
    gpu = replace(
        cpu, **{field.name: getattr(cpu, field.name).cuda() for field in fields(cpu)}
    )
    expected = select_one_hop_energy_graph(cpu, torch.tensor(owners, dtype=torch.int64))
    actual = select_one_hop_energy_graph(
        gpu, torch.tensor(owners, dtype=torch.int64, device="cuda")
    )
    assert actual.node_indices.cpu().tolist() == expected.node_indices.tolist()
    torch.testing.assert_close(
        actual.edge_indices.cpu(), expected.edge_indices, rtol=0, atol=0
    )
    vals = torch.arange(len(actual.node_indices) * 3, dtype=torch.float64).reshape(
        -1, 3
    )
    torch.testing.assert_close(
        actual.scatter(vals.cuda()).cpu(), expected.scatter(vals), rtol=0, atol=0
    )
    support = torch.tensor([1], dtype=torch.int64)
    torch.testing.assert_close(
        affected_one_hop_energy_owners((gpu,), support.cuda()).cpu(),
        affected_one_hop_energy_owners((cpu,), support),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("index_kind", ["uint32", "int64", "bytes4", "bytes8"])
def test_selected_graph_accepts_canonical_index_storage(index_kind):
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=torch.float64
    )
    graph = graph_from_positions(x)
    owners = torch.tensor([0], dtype=torch.int64)
    reference = select_one_hop_energy_graph(graph, owners)
    words = graph.source.to(
        torch.int64 if index_kind in ("int64", "bytes8") else torch.uint32
    )
    source = canonical_index_bytes(words) if index_kind.startswith("bytes") else words
    candidate_graph = replace(graph, source=source)
    candidate = select_one_hop_energy_graph(candidate_graph, owners)
    assert torch.equal(candidate.node_indices, reference.node_indices)
    assert torch.equal(candidate.edge_indices, reference.edge_indices)
    assert torch.equal(candidate.graph.edge_vec, reference.graph.edge_vec)
    support = torch.tensor([1], dtype=torch.int64)
    assert torch.equal(
        affected_one_hop_energy_owners((candidate_graph,), support),
        affected_one_hop_energy_owners((graph,), support),
    )
