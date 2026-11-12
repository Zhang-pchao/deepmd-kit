// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef DEEPMD_SELECTED_ENERGY_GRAPH_KOKKOS_H_
#define DEEPMD_SELECTED_ENERGY_GRAPH_KOKKOS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <chrono>
#include <type_traits>

#include "selected_energy_graph.h"

namespace deepmd {
namespace selected_energy {

// Canonical dual-CSR contract shared with CompactCanonicalGraphKokkos. The
// native LAMMPS neighbor list describes X, not the independent centroid C.
// This adapter bins the supplied geometry with Kokkos primitives, without
// mutating LAMMPS atoms or borrowing an uncovered neighbor list. Storage is
// O(N+E); bins are capped by N even for a large vacuum box. No force is cached.
template <class DeviceType>
class DeviceSelection {
 public:
  template <class T>
  using View = Kokkos::View<T*, DeviceType>;
  View<std::int64_t> types, destination_ptr, source_ptr;
  View<std::uint32_t> source;
  View<std::int64_t> source_order;
  View<float> edge_vec;
  View<int> nodes;
  void enable_profile(bool value) { profile_enabled_ = value; }
  const std::array<double, 8>& profile() const { return profile_; }
  void reset_profile() { profile_.fill(0.0); }
  int owned = 0, node_count = 0;
  std::int64_t edge_count = 0, storage = 0;

  void build(const Neighborhood& geometry,
             const std::vector<int>& owners,
             const std::vector<int>& atom_types) {
    build_impl(geometry, owners, atom_types, false);
  }

  // Reverse one-hop dependencies use the identical cutoff/PBC traversal but
  // need only membership, not sorted edge arrays or model inputs.
  std::vector<int> affected_owners(const Neighborhood& geometry,
                                   const std::vector<int>& support) {
    build_impl(geometry, support, {}, true);
    std::vector<int> selected(geometry.size());
    Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host(
        selected.data(), selected.size());
    Kokkos::deep_copy(
        host, Kokkos::subview(halo_,
                              std::make_pair(std::size_t(0), selected.size())));
    for (const int node : support) {
      selected[node] = 1;
    }
    std::vector<int> result;
    for (int i = 0; i < geometry.size(); ++i) {
      if (selected[i]) {
        result.push_back(i);
      }
    }
    return result;
  }

  // Public only because CUDA extended lambdas require a public host function.
  // Consumers use build() or affected_owners(); no geometry survives a call.
  void build_impl(const Neighborhood& geometry,
                  const std::vector<int>& owners,
                  const std::vector<int>& atom_types,
                  bool membership_only,
                  const DeviceSelection* full = nullptr) {
    auto stamp = std::chrono::steady_clock::now();
    auto mark = [&](int phase) {
      if (!profile_enabled_) {
        return;
      }
      Kokkos::fence();  // Diagnostic-only; never enabled in production timing.
      const auto now = std::chrono::steady_clock::now();
      profile_[phase] += std::chrono::duration<double>(now - stamp).count();
      stamp = now;
    };
    const int n = geometry.size();
    if (!membership_only && atom_types.size() != static_cast<std::size_t>(n)) {
      throw std::invalid_argument(
          "Selected graph type axis differs from coordinates");
    }
    std::vector<int> host_map(n, -1);
    for (std::size_t i = 0; i < owners.size(); ++i) {
      if (owners[i] < 0 || owners[i] >= n || host_map[owners[i]] >= 0) {
        throw std::invalid_argument(
            "Selected graph owners must be unique valid indices");
      }
      host_map[owners[i]] = static_cast<int>(i);
    }
    if (full && (full == this || !full->is_full_ || full->node_count != n)) {
      throw std::invalid_argument(
          "Subset requires a distinct full canonical graph");
    }
    is_full_ = !membership_only && owners.size() == static_cast<std::size_t>(n);
    for (int i = 0; is_full_ && i < n; ++i) {
      is_full_ = owners[i] == i;
    }
    owned = static_cast<int>(owners.size());
    upload(owner_, owners, "selected_device:owners");
    upload(map_, host_map, "selected_device:map");
    if (!membership_only) {
      upload(atom_types_, atom_types, "selected_device:atom_types");
    }
    std::vector<double> host_x;
    host_x.reserve(3 * static_cast<std::size_t>(n));
    for (const auto& p : geometry.positions()) {
      host_x.insert(host_x.end(), p.begin(), p.end());
    }
    upload(x_, host_x, "selected_device:x");
    reserve(nodes, n, "selected_device:nodes");
    reserve(halo_, n, "selected_device:halo");
    reserve(next_, n, "selected_device:next");
    reserve(row_, owned + 1, "selected_device:row");
    Kokkos::deep_copy(halo_, 0);
    Kokkos::deep_copy(row_, std::int64_t{0});
    const auto x = x_;
    const auto owner = owner_;
    const auto map = map_;
    const auto halo = halo_;
    const auto row = row_;
    const auto node_ids = nodes;
    const int m = owned;
    const double cutoff2 = geometry.cutoff() * geometry.cutoff();
    Kokkos::Array<int, 3> counts;
    Kokkos::Array<double, 3> width, origin, box;
    Kokkos::Array<bool, 3> periodic;
    // A cap per dimension bounds empty-vacuum storage while retaining cells
    // at least as wide as the cutoff. Finite coordinate checks are inherited
    // from Neighborhood; clamp the dimension before converting to int.
    const int cap = std::max(1, static_cast<int>(std::cbrt(std::max(1, n))));
    int bins = 1;
    for (int d = 0; d < 3; ++d) {
      box[d] = geometry.box()[d];
      periodic[d] = geometry.periodic()[d];
      double lo = 0, hi = 0;
      if (!periodic[d] && n) {
        lo = hi = geometry.positions()[0][d];
        for (const auto& p : geometry.positions()) {
          lo = std::min(lo, p[d]);
          hi = std::max(hi, p[d]);
        }
      }
      origin[d] = periodic[d] ? 0 : lo;
      const double length =
          periodic[d] ? box[d] : std::max(geometry.cutoff(), hi - lo);
      counts[d] = static_cast<int>(std::max(
          1.0, std::min(double(cap), std::floor(length / geometry.cutoff()))));
      width[d] = std::max(geometry.cutoff(), length / counts[d]);
      bins *= counts[d];
    }
    reserve(head_, bins, "selected_device:head");
    reserve(bin_ptr_, bins + 1, "selected_device:bin_ptr");
    reserve(bin_nodes_, n, "selected_device:bin_nodes");
    reserve(row_cursor_, m, "selected_device:row_cursor");
    Kokkos::deep_copy(head_, 0);
    Kokkos::deep_copy(bin_ptr_, 0);
    const auto bin_ptr = bin_ptr_;
    const auto bin_nodes = bin_nodes_;
    const auto row_cursor = row_cursor_;
    const auto head = head_;
    const auto next = next_;
    mark(0);
    if (!full) {
      Kokkos::parallel_for(
          "selected_device:bin", Kokkos::RangePolicy<DeviceType>(0, n),
          KOKKOS_LAMBDA(const int i) {
            int cell[3];
            for (int d = 0; d < 3; ++d) {
              double v = x(3 * i + d) - origin[d];
              if (periodic[d]) {
                v -= box[d] * floor(v / box[d]);
              }
              cell[d] = int(floor(v / width[d]));
              if (cell[d] >= counts[d]) {
                cell[d] = counts[d] - 1;
              }
              if (cell[d] < 0) {
                cell[d] = 0;
              }
            }
            const int key =
                (cell[0] * counts[1] + cell[1]) * counts[2] + cell[2];
            next(i) = key;
            Kokkos::atomic_fetch_add(&head(key), 1);
          });
      // Compact bins avoid dependent pointer chasing. Candidate atoms inside a
      // bin are contiguous; a warp can inspect them concurrently.
      Kokkos::parallel_scan(
          "selected_device:bin_scan", Kokkos::RangePolicy<DeviceType>(0, bins),
          KOKKOS_LAMBDA(const int k, int& update, const bool final) {
            const int count = head(k);
            if (final) {
              bin_ptr(k) = update;
              head(k) = update;
            }
            update += count;
            if (final && k == bins - 1) {
              bin_ptr(bins) = update;
            }
          });
      Kokkos::parallel_for(
          "selected_device:bin_scatter", Kokkos::RangePolicy<DeviceType>(0, n),
          KOKKOS_LAMBDA(const int i) {
            const int pos = Kokkos::atomic_fetch_add(&head(next(i)), 1);
            bin_nodes(pos) = i;
          });
    }
    mark(1);
    // Count and fill only selected destinations, retaining every halo source.
    // Traversal is repeated once; no N*N distances or candidate allocation.
    using TeamPolicy = Kokkos::TeamPolicy<DeviceType>;
    using Member = typename TeamPolicy::member_type;
    constexpr int lanes = std::is_same<typename DeviceType::execution_space,
                                       Kokkos::DefaultHostExecutionSpace>::value
                              ? 1
                              : 32;
    const bool from_full = full != nullptr;
    const auto full_ptr = full ? full->destination_ptr : View<std::int64_t>();
    const auto full_source = full ? full->source : View<std::uint32_t>();
    const auto full_nodes = full ? full->nodes : View<int>();
    for (int pass = 0; pass < 2; ++pass) {
      const auto keys = edge_keys_;
      Kokkos::parallel_for(
          "selected_device:neighbors", TeamPolicy(m, lanes),
          KOKKOS_LAMBDA(const Member& team) {
            const int k = team.league_rank();
            const int i = owner(k);
            int cell[3];
            for (int d = 0; d < 3; ++d) {
              double v = x(3 * i + d) - origin[d];
              if (periodic[d]) {
                v -= box[d] * floor(v / box[d]);
              }
              cell[d] = int(floor(v / width[d]));
              if (cell[d] >= counts[d]) {
                cell[d] = counts[d] - 1;
              }
              if (cell[d] < 0) {
                cell[d] = 0;
              }
            }
            int visited[27], nv = 0;
            std::int64_t size = 0;
            if (from_full) {
              const auto begin = full_ptr(i), end = full_ptr(i + 1);
              for (auto e = begin + team.team_rank(); e < end;
                   e += team.team_size()) {
                const int j = full_nodes(full_source(e));
                if (pass == 0) {
                  Kokkos::atomic_exchange(&halo(j), 1);
                } else {
                  keys(row(k) + e - begin) =
                      std::uint64_t(k) * std::uint64_t(n) + std::uint64_t(j);
                }
                ++size;
              }
            } else {
              for (int a = -1; a <= 1; ++a) {
                for (int b = -1; b <= 1; ++b) {
                  for (int c = -1; c <= 1; ++c) {
                    int near[3] = {cell[0] + a, cell[1] + b, cell[2] + c};
                    bool valid = true;
                    for (int d = 0; d < 3; ++d) {
                      if (periodic[d]) {
                        near[d] = (near[d] + counts[d]) % counts[d];
                      } else if (near[d] < 0 || near[d] >= counts[d]) {
                        valid = false;
                      }
                    }
                    if (!valid) {
                      continue;
                    }
                    const int key =
                        (near[0] * counts[1] + near[1]) * counts[2] + near[2];
                    bool seen = false;
                    for (int v = 0; v < nv; ++v) {
                      if (visited[v] == key) {
                        seen = true;
                      }
                    }
                    if (seen) {
                      continue;
                    }
                    visited[nv++] = key;
                    for (int pos = bin_ptr(key) + team.team_rank();
                         pos < bin_ptr(key + 1); pos += team.team_size()) {
                      const int j = bin_nodes(pos);
                      if (i == j) {
                        continue;
                      }
                      double r2 = 0;
                      for (int d = 0; d < 3; ++d) {
                        double v = x(3 * j + d) - x(3 * i + d);
                        if (periodic[d]) {
                          v -= box[d] * round(v / box[d]);
                        }
                        r2 += v * v;
                      }
                      if (r2 >= cutoff2) {
                        continue;
                      }
                      if (pass == 0) {
                        Kokkos::atomic_exchange(&halo(j), 1);
                      } else {
                        keys(Kokkos::atomic_fetch_add(&row_cursor(k),
                                                      std::int64_t{1})) =
                            std::uint64_t(k) * std::uint64_t(n) +
                            std::uint64_t(j);
                      }
                      ++size;
                    }
                  }
                }
              }
            }
            if (pass == 0) {
              std::int64_t total = 0;
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(team, team.team_size()),
                  [&](const int, std::int64_t& value) { value += size; },
                  total);
              if (team.team_rank() == 0) {
                row(k) = total;
              }
            }
          });
      if (pass == 0 && membership_only) {
        mark(2);
        return;
      }
      if (pass == 0) {
        Kokkos::parallel_scan(
            "selected_device:rows", Kokkos::RangePolicy<DeviceType>(0, m),
            KOKKOS_LAMBDA(const int k, std::int64_t& update, const bool final) {
              const auto count = row(k);
              if (final) {
                row(k) = update;
              }
              update += count;
              if (final && k == m - 1) {
                row(m) = update;
              }
            });
        Kokkos::deep_copy(edge_count, Kokkos::subview(row_, m));
        storage = std::max<std::int64_t>(2, edge_count);
        if (storage > std::numeric_limits<std::uint32_t>::max()) {
          throw std::overflow_error(
              "Selected device graph exceeds uint32 edge range");
        }
        reserve(edge_keys_, storage, "selected_device:edge_keys");
        Kokkos::parallel_for(
            "selected_device:row_cursor", Kokkos::RangePolicy<DeviceType>(0, m),
            KOKKOS_LAMBDA(const int k) { row_cursor(k) = row(k); });
      }
    }
    mark(2);
    auto keys = Kokkos::subview(
        edge_keys_, std::make_pair(std::size_t(0), std::size_t(edge_count)));
    // Each spatial row is short. Sort within rows to preserve the host oracle's
    // global-node ordering without a device-wide sparse-key bin sort.
    if (!from_full) {
      Kokkos::parallel_for(
          "selected_device:sort_rows", TeamPolicy(m, lanes),
          KOKKOS_LAMBDA(const Member& team) {
            const int k = team.league_rank();
            auto values = Kokkos::subview(
                keys,
                Kokkos::pair<std::size_t, std::size_t>(row(k), row(k + 1)));
            Kokkos::Experimental::sort_team(team, values);
          });
    }
    mark(3);
    int nhalo = 0;
    Kokkos::parallel_scan(
        "selected_device:compact", Kokkos::RangePolicy<DeviceType>(0, n),
        KOKKOS_LAMBDA(const int i, int& update, const bool final) {
          const bool keep = map(i) < 0 && halo(i);
          if (final) {
            if (keep) {
              map(i) = m + update;
            }
            if (map(i) >= 0) {
              node_ids(map(i)) = i;
            }
          }
          update += keep;
        },
        nhalo);
    node_count = m + nhalo;
    const int nn = node_count;
    reserve(types, nn, "selected_device:types");
    reserve(destination_ptr, nn + 1, "selected_device:destination_ptr");
    reserve(source_ptr, nn + 1, "selected_device:source_ptr");
    reserve(source_counts_, nn, "selected_device:source_counts");
    reserve(source, storage, "selected_device:source");
    reserve(edge_vec, 3 * storage, "selected_device:edge_vec");
    reserve(source_order, storage, "selected_device:source_order");
    reserve(source_cursor_, nn, "selected_device:source_cursor");
    Kokkos::deep_copy(source_counts_, std::int64_t{0});
    const auto type = types;
    const auto atom_type = atom_types_;
    const auto dptr = destination_ptr;
    const auto sptr = source_ptr;
    const auto source_counts = source_counts_;
    const auto src = source;
    const auto vec = edge_vec;
    const auto order = source_order;
    const auto cursor = source_cursor_;
    const auto edges = edge_count;
    Kokkos::parallel_for(
        "selected_device:node_arrays",
        Kokkos::RangePolicy<DeviceType>(0, nn + 1), KOKKOS_LAMBDA(const int k) {
          dptr(k) = k < m ? row(k) : edges;
          if (k < nn) {
            type(k) = atom_type(node_ids(k));
          }
        });
    mark(4);
    Kokkos::parallel_for(
        "selected_device:edge_arrays",
        Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(0,
                                                                         edges),
        KOKKOS_LAMBDA(const std::int64_t e) {
          const int k = int(keys(e) / std::uint64_t(n));
          const int j = int(keys(e) % std::uint64_t(n));
          const int i = owner(k);
          const int mj = map(j);
          src(e) = std::uint32_t(mj);
          Kokkos::atomic_fetch_add(&source_counts(mj), std::int64_t{1});
          for (int d = 0; d < 3; ++d) {
            double v = x(3 * j + d) - x(3 * i + d);
            if (periodic[d]) {
              v -= box[d] * round(v / box[d]);
            }
            vec(3 * e + d) = float(v);
          }
        });
    mark(5);
    Kokkos::deep_copy(sptr, std::int64_t{0});
    Kokkos::parallel_scan(
        "selected_device:source_scan", Kokkos::RangePolicy<DeviceType>(0, nn),
        KOKKOS_LAMBDA(const int k, std::int64_t& update, const bool final) {
          const auto count = source_counts(k);
          if (final) {
            sptr(k) = update;
          }
          update += count;
          if (final && k == nn - 1) {
            sptr(nn) = update;
          }
        });
    // Same linear source-CSR scatter as CompactCanonicalGraphKokkos, followed
    // by a row-local ordering pass for reproducible accumulation order.
    Kokkos::parallel_for(
        "selected_device:source_cursor", Kokkos::RangePolicy<DeviceType>(0, nn),
        KOKKOS_LAMBDA(const int k) { cursor(k) = std::uint32_t(sptr(k)); });
    Kokkos::parallel_for(
        "selected_device:source_scatter",
        Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(0,
                                                                         edges),
        KOKKOS_LAMBDA(const std::int64_t e) {
          const auto position =
              Kokkos::atomic_fetch_add(&cursor(src(e)), std::uint32_t{1});
          order(position) = e;
        });
    Kokkos::parallel_for(
        "selected_device:sort_sources", TeamPolicy(nn, lanes),
        KOKKOS_LAMBDA(const Member& team) {
          const int k = team.league_rank();
          auto values = Kokkos::subview(
              order,
              Kokkos::pair<std::size_t, std::size_t>(sptr(k), sptr(k + 1)));
          Kokkos::Experimental::sort_team(team, values);
        });
    mark(6);
    Kokkos::parallel_for(
        "selected_device:guards",
        Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(
            edges, storage),
        KOKKOS_LAMBDA(const std::int64_t e) {
          src(e) = 0;
          order(e) = std::uint32_t(e);
          for (int d = 0; d < 3; ++d) {
            vec(3 * e + d) = 0;
          }
        });
    mark(7);
  }

 private:
  bool profile_enabled_ = false;
  bool is_full_ = false;
  std::array<double, 8> profile_{};
  template <class T>
  static void reserve(View<T>& v, std::size_t n, const char* label) {
    if (v.extent(0) < n) {
      v = View<T>(std::string(label), n + n / 50 + 8);
    }
  }
  template <class T>
  static void upload(View<T>& v, const std::vector<T>& h, const char* label) {
    reserve(v, h.size(), label);
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host(
        h.data(), h.size());
    Kokkos::deep_copy(
        Kokkos::subview(v, std::make_pair(std::size_t(0), h.size())), host);
  }
  View<int> owner_, map_, halo_, next_, head_, atom_types_, bin_ptr_,
      bin_nodes_;
  View<double> x_;
  View<std::int64_t> row_, source_counts_, row_cursor_;
  View<std::uint64_t> edge_keys_;
  View<std::uint32_t> source_cursor_;
};
}  // namespace selected_energy
}  // namespace deepmd
#endif
