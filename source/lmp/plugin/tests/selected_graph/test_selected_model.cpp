// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include "../../../selected_energy_model_kokkos.h"
#include "c_api.h"

namespace {
using Device = Kokkos::DefaultExecutionSpace;
using deepmd::selected_energy::ModelEvaluator;
using deepmd::selected_energy::Neighborhood;

struct CanonicalModel {
  bool supports_selected_energy() const { return true; }
  int numb_types() const { return 2; }
  int calls = 0;
  void compute_canonical_graph_gpu(double* energy,
                                   double* force,
                                   double* virial,
                                   double* atom_virial,
                                   const std::int64_t*,
                                   const void*,
                                   DP_CanonicalIndexWidth index_width,
                                   const float*,
                                   const std::int64_t*,
                                   const std::int64_t*,
                                   const std::int64_t* source_order,
                                   const double* coordinates,
                                   void* world,
                                   const double* box,
                                   std::int64_t owners,
                                   std::int64_t nodes,
                                   std::int64_t,
                                   void* stream = nullptr) {
    ++calls;
    EXPECT_EQ(index_width, DP_CANONICAL_INDEX32);
    EXPECT_NE(source_order, nullptr);
    EXPECT_EQ(atom_virial, nullptr);
    EXPECT_EQ(coordinates, nullptr);
    EXPECT_EQ(world, nullptr);
    EXPECT_EQ(box, nullptr);
    EXPECT_EQ(stream, nullptr);
    Kokkos::parallel_for(
        "selected_model:energy", Kokkos::RangePolicy<Device>(0, owners),
        KOKKOS_LAMBDA(int i) { energy[i] = 2.0; });
    Kokkos::parallel_for(
        "selected_model:force", Kokkos::RangePolicy<Device>(0, 3 * nodes),
        KOKKOS_LAMBDA(int i) { force[i] = i + 0.25; });
    Kokkos::parallel_for(
        "selected_model:virial", Kokkos::RangePolicy<Device>(0, 9),
        KOKKOS_LAMBDA(int i) { virial[i] = 10.0 + i; });
  }
};

TEST(SelectedModel, CanonicalInterfacePreservesGraphVirialAndHaloForces) {
  Neighborhood geometry({{0, 0, 0}, {1, 0, 0}, {2.5, 0, 0}}, {20, 20, 20},
                        {true, true, true}, 1.6);
  ModelEvaluator<Device> evaluator;
  CanonicalModel model;
  const auto result = evaluator.evaluate(model, geometry, {0, 2}, {0, 1, 0});
  EXPECT_EQ(model.calls, 1);
  EXPECT_EQ(result.owners, 2);
  EXPECT_EQ(result.nodes, 3);
  EXPECT_DOUBLE_EQ(result.energy, 4.0);
  for (int d = 0; d < 3; ++d) {
    EXPECT_DOUBLE_EQ(result.force[d], d + 0.25);
    EXPECT_DOUBLE_EQ(result.force[6 + d], 3 + d + 0.25);
    EXPECT_DOUBLE_EQ(result.force[3 + d], 6 + d + 0.25);
  }
  for (int d = 0; d < 9; ++d) {
    EXPECT_DOUBLE_EQ(result.virial[d], 10.0 + d);
  }
}
}  // namespace
