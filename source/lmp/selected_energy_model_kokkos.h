// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef DEEPMD_LMP_SELECTED_ENERGY_MODEL_KOKKOS_H_
#define DEEPMD_LMP_SELECTED_ENERGY_MODEL_KOKKOS_H_

#include <Kokkos_Core.hpp>
#include <chrono>
#include <string>

#include "canonical_graph.h"
#include "selected_energy_graph.h"
#include "selected_energy_graph_kokkos.h"

namespace deepmd {
namespace selected_energy {

struct Evaluation {
  double energy = 0.0;
  std::vector<double> force;
  std::array<double, 9> virial{};
  int owners = 0, nodes = 0;
  std::int64_t edges = 0;
};

// The model is owned by the caller and is loaded once. This adapter stages the
// compact topology, invokes the existing canonical API and scatters ALL node
// derivatives. No PIMD, bias history, force-cache or MPI policy lives here.
// Persistent work buffers may grow; coordinate-dependent results never persist.
template <class DeviceType>
class ModelEvaluator {
 public:
  void use_device_graph(bool enabled) { device_graph_enabled_ = enabled; }
  std::vector<int> affected_owners(const Neighborhood& geometry,
                                   const std::vector<int>& support) {
    if (device_graph_enabled_) {
      return device_graph_.affected_owners(geometry, support);
    }
    return deepmd::selected_energy::affected_owners({geometry}, support);
  }

  // Nested per-rank timings: graph, upload, inference, download/scatter.
  const std::array<double, 4>& profile() const { return profile_; }
  void enable_graph_profile(bool enabled) {
    device_graph_.enable_profile(enabled);
    reference_graph_.enable_profile(enabled);
  }
  const std::array<double, 8>& graph_profile() {
    for (std::size_t i = 0; i < graph_profile_.size(); ++i) {
      graph_profile_[i] =
          device_graph_.profile()[i] + reference_graph_.profile()[i];
    }
    return graph_profile_;
  }
  void reset_profile() {
    profile_.fill(0.0);
    device_graph_.reset_profile();
    reference_graph_.reset_profile();
  }
  template <class Model>
  Evaluation evaluate(Model& model,
                      const Neighborhood& geometry,
                      const std::vector<int>& owners,
                      const std::vector<int>& types) {
    return evaluate_impl(model, geometry, owners, types, device_graph_,
                         nullptr);
  }

  // The reference geometry/types are immutable within this call. The selected
  // graph extracts complete rows from this call's full graph, never a previous
  // force call's topology. Both model evaluations and all halo forces remain.
  template <class Model>
  std::array<Evaluation, 2> evaluate_reference_pair(
      Model& model,
      const Neighborhood& geometry,
      const std::vector<int>& owners,
      const std::vector<int>& types) {
    std::vector<int> all(geometry.size());
    std::iota(all.begin(), all.end(), 0);
    std::array<Evaluation, 2> result;
    result[0] =
        evaluate_impl(model, geometry, all, types, reference_graph_, nullptr);
    result[1] =
        evaluate_impl(model, geometry, owners, types, device_graph_,
                      device_graph_enabled_ ? &reference_graph_ : nullptr);
    return result;
  }

 private:
  template <class Model>
  Evaluation evaluate_impl(Model& model,
                           const Neighborhood& geometry,
                           const std::vector<int>& owners,
                           const std::vector<int>& types,
                           DeviceSelection<DeviceType>& active_graph,
                           const DeviceSelection<DeviceType>* full) {
    if (!model.supports_selected_energy()) {
      throw std::invalid_argument(
          "Model does not support one-hop selected atomic energies");
    }
    if (static_cast<int>(types.size()) != geometry.size()) {
      throw std::invalid_argument(
          "Selected model types and geometry have different node axes");
    }
    auto stamp = std::chrono::steady_clock::now();
    auto mark = [&](int phase) {
      const auto now = std::chrono::steady_clock::now();
      profile_[phase] += std::chrono::duration<double>(now - stamp).count();
      stamp = now;
    };
    CanonicalSelection graph;
    Evaluation result;
    result.force.assign(3 * static_cast<std::size_t>(geometry.size()), 0.0);
    std::int64_t storage = 0;
    for (const int type : types) {
      if (type < 0 || type >= model.numb_types()) {
        throw std::invalid_argument(
            "Selected model type is outside the model type map");
      }
    }
    if (device_graph_enabled_) {
      active_graph.build_impl(geometry, owners, types, false, full);
      graph.owned = active_graph.owned;
      graph.nodes = download(active_graph.nodes, active_graph.node_count);
      result.edges = active_graph.edge_count;
      storage = active_graph.storage;
      atype_ = active_graph.types;
      source_ = active_graph.source;
      edge_vec_ = active_graph.edge_vec;
      destination_ptr_ = active_graph.destination_ptr;
      source_ptr_ = active_graph.source_ptr;
      source_order_ = active_graph.source_order;
    } else {
      graph = build(geometry, owners);
      result.edges = graph.destination_ptr.back();
      storage = static_cast<std::int64_t>(graph.source.size());
    }
    result.owners = graph.owned;
    result.nodes = static_cast<int>(graph.nodes.size());
    mark(0);
    if (owners.empty()) {
      return result;
    }
    if (!device_graph_enabled_) {
      std::vector<std::int64_t> selected_types;
      for (const int node : graph.nodes) {
        selected_types.push_back(types[node]);
      }
      upload(atype_, selected_types, "selected:type");
      upload(source_, graph.source, "selected:source");
      upload(edge_vec_, graph.edge_vec, "selected:edge_vec");
      upload(destination_ptr_, graph.destination_ptr,
             "selected:destination_ptr");
      upload(source_ptr_, graph.source_ptr, "selected:source_ptr");
      upload(source_order_, graph.source_order, "selected:source_order");
    }
    reserve(energy_, owners.size(), "selected:energy");
    reserve(force_, 3 * graph.nodes.size(), "selected:force");
    reserve(virial_, 9, "selected:virial");
    Kokkos::fence();
    mark(1);
    model.compute_canonical_graph_gpu(
        energy_.data(), force_.data(), virial_.data(), nullptr, atype_.data(),
        source_.data(), DP_CANONICAL_INDEX32, edge_vec_.data(),
        destination_ptr_.data(), source_ptr_.data(), source_order_.data(),
        nullptr, nullptr, nullptr, graph.owned, result.nodes, storage);
    Kokkos::fence();
    mark(2);
    const auto energy = download(energy_, owners.size());
    const auto force = download(force_, 3 * graph.nodes.size());
    const auto virial = download(virial_, 9);
    for (const double value : energy) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("Non-finite selected atomic energy");
      }
      result.energy += value;
    }
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
      for (int d = 0; d < 3; ++d) {
        const double value = force[3 * i + d];
        if (!std::isfinite(value)) {
          throw std::runtime_error("Non-finite selected node force");
        }
        result.force[3 * static_cast<std::size_t>(graph.nodes[i]) + d] = value;
      }
    }
    for (int d = 0; d < 9; ++d) {
      if (!std::isfinite(virial[d])) {
        throw std::runtime_error("Non-finite selected graph virial");
      }
      result.virial[d] = virial[d];
    }
    mark(3);
    return result;
  }

 private:
  bool device_graph_enabled_ = false;
  DeviceSelection<DeviceType> device_graph_, reference_graph_;
  std::array<double, 8> graph_profile_{};
  std::array<double, 4> profile_{};
  template <class T>
  using Buffer = Kokkos::View<T*, DeviceType>;
  template <class T>
  static void reserve(Buffer<T>& target, std::size_t size, const char* label) {
    if (target.extent(0) < size) {
      target = Buffer<T>(std::string(label), size);
    }
  }
  template <class T>
  static void upload(Buffer<T>& target,
                     const std::vector<T>& values,
                     const char* label) {
    reserve(target, values.size(), label);
    const Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        host(values.data(), values.size());
    Kokkos::deep_copy(
        Kokkos::subview(target, std::make_pair(std::size_t(0), values.size())),
        host);
  }
  template <class T>
  static std::vector<T> download(const Buffer<T>& source, std::size_t size) {
    std::vector<T> values(size);
    const Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host(
        values.data(), size);
    Kokkos::deep_copy(
        host, Kokkos::subview(source, std::make_pair(std::size_t(0), size)));
    return values;
  }
  Buffer<std::int64_t> atype_, destination_ptr_, source_ptr_, source_order_;
  Buffer<std::uint32_t> source_;
  Buffer<float> edge_vec_;
  Buffer<double> energy_, force_, virial_;
};

}  // namespace selected_energy
}  // namespace deepmd
#endif
