# SPDX-License-Identifier: LGPL-3.0-or-later
"""Select atomic-energy owners from a complete one-hop canonical graph.

This module changes computational ownership, not the potential or nuclear
resolution. The caller must establish that its model is one-hop local and has
no unhandled global/conditional contributions. A multi-block graph must not be
trimmed with this operation.
"""

from dataclasses import (
    dataclass,
)

import torch

from deepmd.pt_expt.utils.canonical_graph import (
    CanonicalGraph,
    canonical_index_dtype,
    canonical_index_words,
    validate_canonical_graph_shapes,
)


@dataclass(frozen=True)
class SelectedEnergyGraph:
    """A selected-owner graph and its mapping to the original node axis.

    The graph is None for an empty selection, whose energy and gradient are
    identically zero. Node indices are unique original indices, with energy
    owners first and read-only halo nodes after them. All returned node gradients,
    including halo gradients, must be scattered before any ghost reduction.
    Periodic edge vectors are retained without reconstructing image coordinates.
    """

    graph: CanonicalGraph | None
    node_indices: torch.Tensor
    edge_indices: torch.Tensor
    original_node_count: int

    def scatter(self, values: torch.Tensor) -> torch.Tensor:
        """Restore node values, including halo derivatives, to the original axis."""
        if values.ndim < 1 or values.shape[0] != self.node_indices.numel():
            raise ValueError("values must have one leading row per selected graph node")
        if values.device != self.node_indices.device:
            raise ValueError("values and node indices must be on the same device")
        result = values.new_zeros((self.original_node_count, *values.shape[1:]))
        return result.index_copy(0, self.node_indices, values)


def select_one_hop_energy_graph(
    graph: CanonicalGraph,
    owners: torch.Tensor,
) -> SelectedEnergyGraph:
    """Retain selected energy centers and their complete source neighborhoods.

    Parameters
    ----------
    graph
        A complete single-frame canonical graph. Its owned prefix may exclude
        spatial-decomposition ghosts, which can still be read as neighbors.
    owners
        Unique int64 indices in the graph's owned prefix, on the graph device.
        Their order defines the new owned prefix.

    Returns
    -------
    SelectedEnergyGraph
        A compact graph containing only selected destination rows and every
        source node needed by those rows. Gradients are defined on all these
        nodes; they must not be truncated to the owned prefix.

    Notes
    -----
    This topology transformation is an eager utility, not an exportable model
    operation. It retains exact edge vectors and does not search neighbors,
    choose a quantum region, or certify a model's receptive field. Selection
    updates require complete reference/bead dependency coverage in the caller.
    """
    n = graph.destination_row_ptr.numel() - 1
    validate_canonical_graph_shapes(graph, n)
    if graph.n_node.numel() != 1 or int(graph.n_node[0]) != n:
        raise ValueError("selected energy graphs require one complete frame")
    if owners.ndim != 1 or owners.dtype != torch.int64:
        raise ValueError("owners must be a one-dimensional int64 tensor")
    if owners.device != graph.source.device:
        raise ValueError("owners and graph must be on the same device")
    nlocal = int(graph.n_local[0])
    if nlocal < 0 or nlocal > n:
        raise ValueError("owned node count must lie within the node axis")
    if owners.numel() and (int(owners.min()) < 0 or int(owners.max()) >= nlocal):
        raise ValueError("energy owners must belong to the original owned prefix")
    if torch.unique(owners).numel() != owners.numel():
        raise ValueError("energy owners must not contain duplicates")
    ptr = graph.destination_row_ptr
    if int(ptr[0]) != 0 or bool(torch.any(ptr[1:] < ptr[:-1])):
        raise ValueError("destination row pointers must start at zero and be ordered")
    count = int(ptr[-1])
    source_words = canonical_index_words(graph.source)
    if count < 0 or count > source_words.numel():
        raise ValueError("physical edge count exceeds storage")
    if int(graph.source_row_ptr[-1]) != count:
        raise ValueError("source and destination physical edge counts must agree")
    source = source_words[:count].to(torch.int64)
    if source.numel() and (int(source.min()) < 0 or int(source.max()) >= n):
        raise ValueError("source node index is outside the original node axis")
    empty = owners.new_empty(0)
    if not owners.numel():
        return SelectedEnergyGraph(None, empty, empty, n)

    destinations = torch.repeat_interleave(
        torch.arange(n, device=owners.device), ptr[1:] - ptr[:-1]
    )
    owner_rank = owners.new_full((n,), -1)
    owner_rank[owners] = torch.arange(owners.numel(), device=owners.device)
    edges = torch.nonzero(owner_rank[destinations] >= 0).flatten()
    edges = edges[torch.argsort(owner_rank[destinations[edges]], stable=True)]
    neighbor_nodes = torch.unique(source[edges], sorted=True)
    halo = neighbor_nodes[owner_rank[neighbor_nodes] < 0]
    nodes = torch.cat((owners, halo))
    reverse = owners.new_full((n,), -1)
    reverse[nodes] = torch.arange(nodes.numel(), device=nodes.device)
    new_sources = reverse[source[edges]]
    new_destinations = owner_rank[destinations[edges]]

    edge_count = edges.numel()
    storage = max(edge_count, 2)
    compact_sources = torch.zeros(
        storage, dtype=canonical_index_dtype(nodes.numel()), device=nodes.device
    )
    vectors = graph.edge_vec.new_zeros((storage, 3))
    source_order = torch.arange(storage, device=nodes.device, dtype=torch.int64)
    if edge_count:
        compact_sources[:edge_count] = new_sources.to(compact_sources.dtype)
        vectors[:edge_count] = graph.edge_vec[edges]
        source_order[:edge_count] = torch.argsort(new_sources, stable=True)
    zero = owners.new_zeros(1)
    destination_ptr = torch.cat(
        (zero, torch.bincount(new_destinations, minlength=nodes.numel()).cumsum(0))
    )
    source_ptr = torch.cat(
        (zero, torch.bincount(new_sources, minlength=nodes.numel()).cumsum(0))
    )
    compact = CanonicalGraph(
        n_node=owners.new_tensor([nodes.numel()]),
        n_local=owners.new_tensor([owners.numel()]),
        source=compact_sources,
        edge_vec=vectors,
        destination_row_ptr=destination_ptr,
        source_row_ptr=source_ptr,
        source_order=source_order,
    )
    validate_canonical_graph_shapes(compact, nodes.numel())
    return SelectedEnergyGraph(compact, nodes, edges, n)


def affected_one_hop_energy_owners(
    graphs: tuple[CanonicalGraph, ...], support: torch.Tensor
) -> torch.Tensor:
    """Find all owned centers that can depend on a changing node support.

    Each graph must use the same node identity/order and owned prefix. The
    caller supplies every reference and candidate geometry, including all
    periodic-image edges. Support is a capability set, not a displacement
    threshold: a node with zero current displacement can still have derivatives.
    The returned conservative union is valid for the supplied geometries only;
    it is not a cache or a neighbor-list skin policy.
    """
    if not graphs:
        raise ValueError("at least one complete graph is required")
    n = graphs[0].destination_row_ptr.numel() - 1
    nlocal = int(graphs[0].n_local[0])
    if support.ndim != 1 or support.dtype != torch.int64:
        raise ValueError("support must be a one-dimensional int64 tensor")
    if support.device != graphs[0].source.device:
        raise ValueError("support and graph must be on the same device")
    if support.numel() and (int(support.min()) < 0 or int(support.max()) >= n):
        raise ValueError("support index is outside the graph node axis")
    active = torch.zeros(n, dtype=torch.bool, device=support.device)
    active[support] = True
    affected = active.clone()
    for graph in graphs:
        # Reuse structural validation; empty selection performs no graph work.
        select_one_hop_energy_graph(graph, support.new_empty(0))
        if (
            int(graph.n_node[0]) != n
            or int(graph.n_local[0]) != nlocal
            or graph.source.device != support.device
        ):
            raise ValueError("graphs must have the same node axis and owned prefix")
        ptr = graph.destination_row_ptr
        count = int(ptr[-1])
        destinations = torch.repeat_interleave(
            torch.arange(n, device=support.device), ptr[1:] - ptr[:-1]
        )
        changed = active[canonical_index_words(graph.source)[:count].long()]
        affected[destinations[changed]] = True
    return torch.nonzero(affected[:nlocal]).flatten()


__all__ = [
    "SelectedEnergyGraph",
    "affected_one_hop_energy_owners",
    "select_one_hop_energy_graph",
]
