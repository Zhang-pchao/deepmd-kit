// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef DEEPMD_LMP_SELECTED_ENERGY_GRAPH_H_
#define DEEPMD_LMP_SELECTED_ENERGY_GRAPH_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace deepmd {
namespace selected_energy {

using Position = std::array<double, 3>;

// A complete one-hop neighbor search over a shared, stable global node axis.
// This host implementation avoids an N*N temporary and can query only selected
// centers. It supports orthorhombic boxes with at most one periodic image per
// node pair. Unsupported cells must use the caller's full provider instead.
class Neighborhood {
 public:
  Neighborhood(const std::vector<Position>& positions,
               const Position& box,
               const std::array<bool, 3>& periodic,
               double cutoff)
      : positions_(positions), box_(box), periodic_(periodic), cutoff_(cutoff) {
    if (!std::isfinite(cutoff) || cutoff <= 0.0 ||
        positions.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
          "Invalid selected-energy cutoff or node count");
    }
    for (int d = 0; d < 3; ++d) {
      cells_[d] = 0;
      width_[d] = cutoff;
      if (periodic[d]) {
        if (!std::isfinite(box[d]) || box[d] <= 2 * cutoff ||
            box[d] / cutoff > std::numeric_limits<int>::max() - 2) {
          throw std::invalid_argument(
              "Selected-energy periodic length must exceed twice cutoff");
        }
        cells_[d] = static_cast<int>(std::floor(box[d] / cutoff));
        width_[d] = box[d] / cells_[d];
      }
    }
    for (int i = 0; i < size(); ++i) {
      bins_[key(positions_[i])].push_back(i);
    }
  }

  int size() const { return static_cast<int>(positions_.size()); }
  const std::vector<Position>& positions() const { return positions_; }
  const Position& box() const { return box_; }
  const std::array<bool, 3>& periodic() const { return periodic_; }
  double cutoff() const { return cutoff_; }

  std::vector<std::pair<int, Position>> neighbors(int center) const {
    if (center < 0 || center >= size()) {
      throw std::out_of_range("Invalid energy center");
    }
    const auto origin = key(positions_[center]);
    std::array<std::array<int, 3>, 27> visited;
    int visited_count = 0;
    std::vector<std::pair<int, Position>> result;
    for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          std::array<int, 3> bin = {origin[0] + i, origin[1] + j,
                                    origin[2] + k};
          for (int d = 0; d < 3; ++d) {
            if (periodic_[d]) {
              bin[d] = (bin[d] + cells_[d]) % cells_[d];
            }
          }
          if (std::find(visited.begin(), visited.begin() + visited_count,
                        bin) != visited.begin() + visited_count) {
            continue;
          }
          visited[visited_count++] = bin;
          const auto found = bins_.find(bin);
          if (found == bins_.end()) {
            continue;
          }
          for (const int node : found->second) {
            if (node == center) {
              continue;
            }
            Position delta;
            double r2 = 0.0;
            for (int d = 0; d < 3; ++d) {
              delta[d] = positions_[node][d] - positions_[center][d];
              if (periodic_[d]) {
                delta[d] -= box_[d] * std::round(delta[d] / box_[d]);
              }
              r2 += delta[d] * delta[d];
            }
            if (r2 < cutoff_ * cutoff_) {
              result.emplace_back(node, delta);
            }
          }
        }
      }
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return result;
  }

 private:
  std::array<int, 3> key(const Position& x) const {
    std::array<int, 3> result;
    for (int d = 0; d < 3; ++d) {
      double value = x[d];
      if (!std::isfinite(value)) {
        throw std::invalid_argument("Non-finite selected-energy coordinate");
      }
      if (periodic_[d]) {
        value = std::fmod(value, box_[d]);
        if (value < 0) {
          value += box_[d];
        }
      }
      const double cell = std::floor(value / width_[d]);
      if (cell < std::numeric_limits<int>::min() + 2.0 ||
          cell > std::numeric_limits<int>::max() - 2.0) {
        throw std::invalid_argument(
            "Selected-energy coordinate exceeds cell-index range");
      }
      result[d] = static_cast<int>(cell);
      if (periodic_[d]) {
        result[d] = std::min(result[d], cells_[d] - 1);
      }
    }
    return result;
  }
  std::vector<Position> positions_;
  Position box_, width_;
  std::array<bool, 3> periodic_;
  std::array<int, 3> cells_;
  double cutoff_;
  std::map<std::array<int, 3>, std::vector<int>> bins_;
};

// Union reverse dependencies for all supplied geometries. Support must include
// every node capable of changing, not only currently displaced nodes.
inline std::vector<int> affected_owners(const std::vector<Neighborhood>& frames,
                                        const std::vector<int>& support) {
  if (frames.empty()) {
    throw std::invalid_argument("At least one geometry is required");
  }
  const int n = frames.front().size();
  std::vector<bool> selected(n, false);
  for (const auto& frame : frames) {
    if (frame.size() != n) {
      throw std::invalid_argument("Geometry node axes differ");
    }
    for (const int node : support) {
      if (node < 0 || node >= n) {
        throw std::out_of_range("Invalid quantum support index");
      }
      selected[node] = true;
      for (const auto& neighbor : frame.neighbors(node)) {
        selected[neighbor.first] = true;
      }
    }
  }
  std::vector<int> owners;
  for (int i = 0; i < n; ++i) {
    if (selected[i]) {
      owners.push_back(i);
    }
  }
  return owners;
}

struct CanonicalSelection {
  int owned = 0;
  std::vector<int> nodes;
  std::vector<std::uint32_t> source;
  std::vector<std::int64_t> source_order;
  std::vector<float> edge_vec;
  std::vector<std::int64_t> destination_ptr, source_ptr;
};

inline CanonicalSelection build(const Neighborhood& geometry,
                                const std::vector<int>& owners) {
  CanonicalSelection out;
  const int n = geometry.size();
  std::vector<int> index(n, -1);
  if (owners.size() > static_cast<std::size_t>(n)) {
    throw std::invalid_argument("More selected owners than graph nodes");
  }
  out.nodes = owners;
  out.owned = static_cast<int>(owners.size());
  for (int i = 0; i < out.owned; ++i) {
    const int node = owners[i];
    if (node < 0 || node >= n || index[node] >= 0) {
      throw std::invalid_argument(
          "Selected energy owners must be unique valid indices");
    }
    index[node] = i;
  }
  std::vector<std::vector<std::pair<int, Position>>> rows;
  std::vector<unsigned char> halo(n, 0);
  std::int64_t count = 0;
  for (const int owner : owners) {
    rows.push_back(geometry.neighbors(owner));
    count += static_cast<std::int64_t>(rows.back().size());
    for (const auto& entry : rows.back()) {
      if (index[entry.first] < 0) {
        halo[entry.first] = 1;
      }
    }
  }
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("Selected energy edges exceed uint32 range");
  }
  for (int node = 0; node < n; ++node) {
    if (halo[node]) {
      index[node] = static_cast<int>(out.nodes.size());
      out.nodes.push_back(node);
    }
  }
  const auto storage =
      static_cast<std::size_t>(std::max<std::int64_t>(count, 2));
  out.source.resize(storage, 0);
  out.source_order.resize(storage, 0);
  out.edge_vec.resize(3 * storage, 0.0f);
  out.destination_ptr.assign(out.nodes.size() + 1, 0);
  out.source_ptr.assign(out.nodes.size() + 1, 0);
  std::uint32_t edge = 0;
  for (int row = 0; row < out.owned; ++row) {
    for (const auto& entry : rows[row]) {
      const int src = index[entry.first];
      out.source[edge] = static_cast<std::uint32_t>(src);
      ++out.source_ptr[src + 1];
      for (int d = 0; d < 3; ++d) {
        out.edge_vec[3 * static_cast<std::size_t>(edge) + d] =
            static_cast<float>(entry.second[d]);
      }
      ++edge;
    }
    out.destination_ptr[row + 1] = edge;
  }
  for (std::size_t row = out.owned; row < out.nodes.size(); ++row) {
    out.destination_ptr[row + 1] = edge;
  }
  std::partial_sum(out.source_ptr.begin(), out.source_ptr.end(),
                   out.source_ptr.begin());
  std::iota(out.source_order.begin(), out.source_order.end(), 0);
  // Stable counting scatter has the same order as stable_sort(source), but
  // linear work. source_ptr already contains the necessary bucket offsets.
  auto cursor = out.source_ptr;
  for (std::int64_t e = 0; e < count; ++e) {
    out.source_order[cursor[out.source[e]]++] = e;
  }
  return out;
}

}  // namespace selected_energy
}  // namespace deepmd
#endif
