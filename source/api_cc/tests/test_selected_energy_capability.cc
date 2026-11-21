// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepPotPTExpt.h"
#include "test_utils.h"
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
#include <gtest/gtest.h>

#include "../src/commonPTExpt.h"
#include "../src/selected_energy_capability.h"

using deepmd::ptexpt::parse_json;
using deepmd::ptexpt::selected_energy_capability;

TEST(SelectedEnergyCapability, AcceptsAuditedModelsAndRejectsUnsupportedMetadata) {
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

  auto conditioned = meta;
  conditioned.obj_val["has_chg_spin_ebd"] = parse_json("true");
  conditioned.obj_val["has_default_chg_spin"] = parse_json("true");
  conditioned.obj_val["default_chg_spin"] = parse_json("[0,1]");
  conditioned.obj_val["charge_state_constants"] =
      parse_json(R"(["type_table","pair_table","","embedding"])");
  auto conditioned_model = model;
  conditioned_model.obj_val["descriptor"].obj_val["add_chg_spin_ebd"] =
      parse_json("true");
  conditioned_model.obj_val["descriptor"].obj_val["default_chg_spin"] =
      parse_json("[0,1]");
  EXPECT_TRUE(selected_energy_capability(conditioned, conditioned_model));
  EXPECT_FALSE(selected_energy_capability(meta, conditioned_model));
  EXPECT_FALSE(selected_energy_capability(conditioned, model));
  for (const auto* key : {"has_default_chg_spin", "default_chg_spin",
                          "charge_state_constants"}) {
    changed = conditioned;
    changed.obj_val.erase(key);
    EXPECT_FALSE(selected_energy_capability(changed, conditioned_model)) << key;
  }
  for (const auto* value : {"[]", "[0]", "[0,3]", "[1,1]", "[0,1,0]",
                            R"(["0",1])"}) {
    changed = conditioned;
    changed.obj_val["default_chg_spin"] = parse_json(value);
    EXPECT_FALSE(selected_energy_capability(changed, conditioned_model)) << value;
  }
  changed = conditioned;
  changed.obj_val["dim_chg_spin"] = parse_json("2");
  EXPECT_FALSE(selected_energy_capability(changed, conditioned_model));
  changed = conditioned;
  changed.obj_val["charge_state_constants"] =
      parse_json(R"(["","pair_table","","embedding"])");
  EXPECT_FALSE(selected_energy_capability(changed, conditioned_model));
  changed = conditioned_model;
  changed.obj_val["descriptor"].obj_val["default_chg_spin"] = parse_json("[1,1]");
  EXPECT_FALSE(selected_energy_capability(conditioned, changed));
}
#else
TEST(SelectedEnergyCapability, BackendUnavailable) {
  GTEST_SKIP() << "PyTorch exportable backend is required";
}
#endif
