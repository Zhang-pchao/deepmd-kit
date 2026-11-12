// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <random>

#include "../../lmp/selected_energy_graph.h"

using deepmd::selected_energy::affected_owners;
using deepmd::selected_energy::build;
using deepmd::selected_energy::Neighborhood;
using deepmd::selected_energy::Position;

TEST(SelectedEnergyGraph, IndependentAllPairsAndPeriodicImages) {
  std::mt19937 generator(873);
  std::uniform_real_distribution<double> random(-15, 15);
  for (const bool pbc : {false, true}) {
    std::vector<Position> x(174);
    for (auto& v : x) {
      for (auto& c : v) {
        c = random(generator);
      }
    }
    const Position box = {13.1, 15., 17.};
    Neighborhood search(x, box, {pbc, pbc, pbc}, 6.0);
    for (int center = 0; center < 174; ++center) {
      const auto result = search.neighbors(center);
      std::vector<int> reference;
      for (int node = 0; node < 174; ++node) {
        if (center == node) {
          continue;
        }
        double r2 = 0;
        for (int d = 0; d < 3; ++d) {
          double delta = x[node][d] - x[center][d];
          if (pbc) {
            delta -= box[d] * std::nearbyint(delta / box[d]);
          }
          r2 += delta * delta;
        }
        if (r2 < 36) {
          reference.push_back(node);
        }
      }
      ASSERT_EQ(result.size(), reference.size());
      for (std::size_t i = 0; i < reference.size(); ++i) {
        EXPECT_EQ(result[i].first, reference[i]);
      }
    }
  }
}

TEST(SelectedEnergyGraph, CompleteHaloAndDualCsr) {
  std::vector<Position> x = {{0, 0, 0}, {1, 0, 0}, {2.5, 0, 0}, {4, 0, 0}};
  Neighborhood search(x, {20, 20, 20}, {true, true, true}, 2.1);
  const auto graph = build(search, {2, 0});
  EXPECT_EQ(graph.owned, 2);
  EXPECT_EQ(graph.nodes, (std::vector<int>{2, 0, 1, 3}));
  EXPECT_EQ(graph.destination_ptr, (std::vector<std::int64_t>{0, 2, 3, 3, 3}));
  EXPECT_EQ(graph.source_ptr.back(), 3);
  for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
    for (auto e = graph.source_ptr[node]; e < graph.source_ptr[node + 1]; ++e) {
      EXPECT_EQ(graph.source[graph.source_order[e]], node);
    }
  }
  EXPECT_THROW(build(search, {0, 0}), std::invalid_argument);
  EXPECT_THROW(build(search, {-1}), std::invalid_argument);
  const auto empty = build(search, {});
  EXPECT_TRUE(empty.nodes.empty());
  EXPECT_EQ(empty.source.size(), 2u);
  EXPECT_EQ(empty.destination_ptr, (std::vector<std::int64_t>{0}));
}

TEST(SelectedEnergyGraph, UnionCrossingAndUnsupportedCell) {
  std::vector<Position> x = {{0, 0, 0}, {2, 0, 0}, {4.25, 0, 0}, {9, 0, 0}};
  auto y = x;
  y[1][0] = 2.25;
  std::vector<Neighborhood> frames;
  frames.emplace_back(x, Position{20, 20, 20},
                      std::array<bool, 3>{false, false, false}, 2.1);
  frames.emplace_back(y, Position{20, 20, 20},
                      std::array<bool, 3>{false, false, false}, 2.1);
  EXPECT_EQ(affected_owners(frames, {1, 3}), (std::vector<int>{0, 1, 2, 3}));
  EXPECT_TRUE(affected_owners(frames, {}).empty());
  EXPECT_THROW(Neighborhood(x, (Position{4, 20, 20}),
                            (std::array<bool, 3>{true, true, true}), 2.1),
               std::invalid_argument);
  EXPECT_THROW(affected_owners(frames, {4}), std::out_of_range);
  x[0][0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(Neighborhood(x, (Position{20, 20, 20}),
                            (std::array<bool, 3>{false, false, false}), 2.1),
               std::invalid_argument);
}

TEST(SelectedEnergyGraph, StableSourceOrderMatchesIndependentSorting) {
  std::mt19937 generator(2107);
  std::uniform_real_distribution<double> random(-9., 9.);
  for (const int n : {0, 1, 2, 24, 174}) {
    std::vector<Position> x(n);
    for (auto& v : x) {
      for (auto& c : v) {
        c = random(generator);
      }
    }
    Neighborhood search(x, {12.016, 12.016, 12.016}, {true, true, true}, 6.);
    std::vector<int> owners(n);
    std::iota(owners.begin(), owners.end(), 0);
    std::shuffle(owners.begin(), owners.end(), generator);
    for (const bool subset : {false, true}) {
      if (subset) {
        owners.resize(owners.size() / 2);
      }
      const auto graph = build(search, owners);
      auto expected = graph.source_order;
      std::iota(expected.begin(), expected.end(), 0);
      const auto edges = graph.destination_ptr.back();
      std::stable_sort(expected.begin(), expected.begin() + edges,
                       [&](std::uint32_t a, std::uint32_t b) {
                         return graph.source[a] < graph.source[b];
                       });
      EXPECT_EQ(graph.source_order, expected);
      EXPECT_TRUE(
          std::is_sorted(graph.nodes.begin() + graph.owned, graph.nodes.end()));
      for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
        for (auto e = graph.source_ptr[node]; e < graph.source_ptr[node + 1];
             ++e) {
          EXPECT_EQ(graph.source[graph.source_order[e]], node);
        }
      }
    }
  }
}
