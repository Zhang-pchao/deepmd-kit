// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepPotPTExpt.h"
#include "test_utils.h"
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
#include <gtest/gtest.h>

#include "../src/commonPTExpt.h"
#include "../src/selected_energy_capability.h"

using deepmd::ptexpt::parse_json;
using deepmd::ptexpt::selected_energy_capability;

TEST(SelectedEnergyCapability, RejectsUnknownOrNonlocalModels) {
  const auto meta = parse_json(R"({
    "lower_input_kind":"dpa4c_canonical","graph_edge_dtype":"float32",
    "canonical_index_dtype":"auto","canonical_index_storage":"byte_words","has_message_passing":false,
    "is_spin":false,"has_chg_spin_ebd":false,"dim_fparam":0,
    "dim_aparam":0,"dim_chg_spin":0,"pair_exclude_types":[],
    "do_atomic_virial":true,"fitting_output_defs":[{"name":"energy",
    "reducible":true,"atomic":true,"intensive":false,
    "r_differentiable":true,"c_differentiable":true}]})");
  const auto model = parse_json(R"({"type":"standard","bridging_method":"None",
    "pair_exclude_types":[],"atom_exclude_types":[],
    "descriptor":{"type":"dpa4c"},"fitting_net":{"type":"ener"}})");
  EXPECT_TRUE(selected_energy_capability(meta, model));
  auto nonlocal_model = model;
  nonlocal_model.obj_val["descriptor"].obj_val["n_blocks"] = parse_json("2");
  EXPECT_FALSE(selected_energy_capability(meta, nonlocal_model));
  auto long_range_meta = meta;
  long_range_meta.obj_val["has_long_range"] = parse_json("true");
  EXPECT_FALSE(selected_energy_capability(long_range_meta, model));
  auto legacy_meta = meta;
  legacy_meta.obj_val["canonical_index_dtype"] = parse_json("\"uint32\"");
  EXPECT_FALSE(selected_energy_capability(legacy_meta, model));
  auto changed = meta;
  changed.obj_val.erase("has_message_passing");
  EXPECT_FALSE(selected_energy_capability(changed, model));
  changed = meta;
  changed.obj_val["has_message_passing"] = parse_json("true");
  EXPECT_FALSE(selected_energy_capability(changed, model));
  changed = meta;
  changed.obj_val["dim_fparam"] = parse_json("1");
  EXPECT_FALSE(selected_energy_capability(changed, model));
  changed = meta;
  changed.obj_val["is_spin"] = parse_json("true");
  EXPECT_FALSE(selected_energy_capability(changed, model));
  changed = meta;
  changed.obj_val["lower_input_kind"] = parse_json("\"dpa1_canonical\"");
  EXPECT_FALSE(selected_energy_capability(changed, model));
  changed = meta;
  changed.obj_val["fitting_output_defs"].arr_val[0].obj_val["intensive"] =
      parse_json("true");
  EXPECT_FALSE(selected_energy_capability(changed, model));
  changed = model;
  changed.obj_val["long_range"] = parse_json("{}");
  EXPECT_FALSE(selected_energy_capability(meta, changed));
  changed = model;
  changed.obj_val["bridging_method"] = parse_json("\"switch\"");
  EXPECT_FALSE(selected_energy_capability(meta, changed));
  EXPECT_FALSE(selected_energy_capability(parse_json("{}"), model));
}
#else
TEST(SelectedEnergyCapability, BackendUnavailable) {
  GTEST_SKIP() << "PyTorch exportable backend is required";
}
#endif
