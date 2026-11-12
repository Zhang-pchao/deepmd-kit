// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <random>

#include "../../../selected_energy_graph_kokkos.h"

using namespace deepmd::selected_energy;
using Device = Kokkos::DefaultExecutionSpace;

template <class V>
auto host(const V& v, std::size_t size) {
  using T = typename V::non_const_value_type;
  std::vector<T> values(size);
  Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> mirror(
      values.data(), size);
  Kokkos::deep_copy(mirror,
                    Kokkos::subview(v, std::make_pair(std::size_t(0), size)));
  return values;
}

void compare(DeviceSelection<Device>& candidate,
             const Neighborhood& geometry,
             const std::vector<int>& owners,
             const DeviceSelection<Device>* full = nullptr) {
  std::vector<int> types(geometry.size());
  for (int i = 0; i < geometry.size(); ++i) {
    types[i] = i % 2;
  }
  const auto reference = build(geometry, owners);
  EXPECT_EQ(candidate.affected_owners(geometry, owners),
            affected_owners({geometry}, owners));
  candidate.build_impl(geometry, owners, types, false, full);
  ASSERT_EQ(candidate.owned, reference.owned);
  ASSERT_EQ(candidate.node_count, reference.nodes.size());
  ASSERT_EQ(candidate.edge_count, reference.destination_ptr.back());
  ASSERT_EQ(candidate.storage, reference.source.size());
  EXPECT_EQ(host(candidate.nodes, candidate.node_count), reference.nodes);
  EXPECT_EQ(host(candidate.source, candidate.storage), reference.source);
  EXPECT_EQ(host(candidate.source_order, candidate.storage),
            reference.source_order);
  EXPECT_EQ(host(candidate.destination_ptr, candidate.node_count + 1),
            reference.destination_ptr);
  EXPECT_EQ(host(candidate.source_ptr, candidate.node_count + 1),
            reference.source_ptr);
  const auto vectors = host(candidate.edge_vec, 3 * candidate.storage);
  for (std::size_t i = 0; i < vectors.size(); ++i) {
    EXPECT_NEAR(vectors[i], reference.edge_vec[i], 1e-6);
  }
  const auto got_types = host(candidate.types, candidate.node_count);
  for (std::size_t i = 0; i < got_types.size(); ++i) {
    EXPECT_EQ(got_types[i], types[reference.nodes[i]]);
  }
}

TEST(SelectedDeviceGraph, RandomPeriodicMixedAndOpen) {
  std::mt19937 random(9173);
  std::uniform_real_distribution<double> draw(-30, 30);
  DeviceSelection<Device> graph;
  for (int sample = 0; sample < 6; ++sample) {
    std::vector<Position> x(174);
    for (auto& p : x) {
      for (double& d : p) {
        d = draw(random);
      }
    }
    for (const auto& pbc : {std::array<bool, 3>{true, true, true},
                            std::array<bool, 3>{true, true, false},
                            std::array<bool, 3>{false, false, false}}) {
      Neighborhood geometry(x, {13.1, 17, 19}, pbc, 6);
      compare(graph, geometry, {173, 0, 3, 8, 45, 90});
      std::vector<int> all(174);
      std::iota(all.begin(), all.end(), 0);
      compare(graph, geometry, all);
      compare(graph, geometry, {});
    }
  }
}
TEST(SelectedDeviceGraph, CutoffAndPeriodicCrossing) {
  DeviceSelection<Device> graph;
  for (double step : {-1e-7, 0., 1e-7}) {
    Neighborhood geometry(
        {{0, 0, 0}, {6 + step, 0, 0}, {19.999, 0, 0}, {20.001, 0, 0}},
        {20, 20, 20}, {true, true, true}, 6);
    compare(graph, geometry, {0, 1});
  }
}
TEST(SelectedDeviceGraph, GrowthEmptyAndVacuum) {
  DeviceSelection<Device> graph;
  compare(graph, Neighborhood({}, {20, 20, 20}, {true, true, true}, 6), {});
  compare(graph, Neighborhood({{0, 0, 0}}, {20, 20, 20}, {true, true, true}, 6),
          {0});
  std::vector<Position> x(174);
  for (int i = 0; i < 174; ++i) {
    x[i] = {double(i % 6), double((i / 6) % 6), double(i / 36)};
  }
  compare(graph, Neighborhood(x, {1e8, 1e8, 1e8}, {true, true, true}, 6),
          {0, 50, 173});
  compare(graph,
          Neighborhood({{1, 1, 1}, {2, 2, 2}}, {20, 20, 20},
                       {false, false, false}, 6),
          {1});
}
TEST(SelectedDeviceGraph, SameCallFullReferenceRows) {
  DeviceSelection<Device> full, subset;
  std::mt19937 random(17391);
  std::uniform_real_distribution<double> draw(-15, 15);
  std::vector<Position> x(174);
  std::vector<int> all(174), types(174);
  std::iota(all.begin(), all.end(), 0);
  for (int i = 0; i < 174; ++i) {
    types[i] = i % 2;
  }
  for (int frame = 0; frame < 8; ++frame) {
    for (auto& p : x) {
      for (double& d : p) {
        d = draw(random);
      }
    }
    Neighborhood geometry(x, {20, 21, 23}, {true, true, frame % 2 == 0}, 6);
    full.build(geometry, all, types);
    compare(subset, geometry, {173, 5, 23, 8, 100}, &full);
    compare(subset, geometry, {}, &full);
    compare(subset, geometry, all, &full);
  }
}

TEST(SelectedDeviceGraph, InvalidOwnerAndTypeAxes) {
  DeviceSelection<Device> graph;
  Neighborhood geometry({{0, 0, 0}, {1, 0, 0}}, {20, 20, 20},
                        {true, true, true}, 6);
  EXPECT_THROW(graph.build(geometry, {0, 0}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(graph.build(geometry, {-1}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(graph.build(geometry, {2}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(graph.build(geometry, {0}, {0}), std::invalid_argument);
}
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  const int status = RUN_ALL_TESTS();
  Kokkos::finalize();
  return status;
}
