// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <exception>
#include <set>
#include <string>

namespace deepmd {
namespace ptexpt {

// Fail closed for unknown/missing metadata. The operation uses the existing
// canonical inference API; this query certifies only the audited one-hop model
// family. It does not select spatial support or certify graph coverage.
template <class Json>
bool selected_energy_capability(const Json& meta, const Json& model) {
  try {
    const auto text = [](const Json& obj, const char* key, const char* value) {
      return obj.has(key) && obj[key].type == Json::String &&
             obj[key].as_string() == value;
    };
    const auto flag = [](const Json& obj, const char* key, bool value) {
      return obj.has(key) && obj[key].type == Json::Bool &&
             obj[key].as_bool() == value;
    };
    const auto zero = [](const Json& obj, const char* key) {
      return obj.has(key) && obj[key].type == Json::Number &&
             obj[key].as_double() == 0.0;
    };
    const auto empty = [](const Json& obj, const char* key) {
      return obj.has(key) && obj[key].type == Json::Array &&
             obj[key].as_array().empty();
    };
    if (!text(meta, "lower_input_kind", "dpa4c_canonical") ||
        !text(meta, "graph_edge_dtype", "float32") ||
        (!text(meta, "canonical_index_dtype", "auto") ||
         !text(meta, "canonical_index_storage", "byte_words")) ||
        !flag(meta, "has_message_passing", false) ||
        !flag(meta, "is_spin", false) ||
        !flag(meta, "has_chg_spin_ebd", false) || !zero(meta, "dim_fparam") ||
        !zero(meta, "dim_aparam") || !zero(meta, "dim_chg_spin") ||
        !empty(meta, "pair_exclude_types") ||
        !flag(meta, "do_atomic_virial", true)) {
      return false;
    }
    if (!text(model, "type", "standard") ||
        !text(model, "bridging_method", "None") ||
        !empty(model, "pair_exclude_types") ||
        !empty(model, "atom_exclude_types") ||
        !text(model["descriptor"], "type", "dpa4c") ||
        !text(model["fitting_net"], "type", "ener")) {
      return false;
    }
    const std::set<std::string> allowed = {"type_map",
                                           "descriptor",
                                           "fitting_net",
                                           "data_stat_nbatch",
                                           "data_stat_full",
                                           "data_stat_protect",
                                           "data_bias_nsample",
                                           "pair_exclude_types",
                                           "atom_exclude_types",
                                           "preset_out_bias",
                                           "srtab_add_bias",
                                           "type",
                                           "model_branch_alias",
                                           "info",
                                           "bridging_method",
                                           "bridging_r_inner",
                                           "bridging_r_outer",
                                           "bridging_fraction_inner",
                                           "bridging_fraction_outer"};
    for (const auto& item : model.as_object()) {
      if (!allowed.count(item.first)) {
        return false;
      }
    }
    const auto& descriptor = model["descriptor"];
    if ((descriptor.has("n_blocks") &&
         descriptor["n_blocks"].as_double() != 1.0) ||
        (meta.has("has_long_range") && !flag(meta, "has_long_range", false)) ||
        model.has("long_range")) {
      return false;
    }
    bool energy = false;
    for (const auto& output : meta["fitting_output_defs"].as_array()) {
      if (text(output, "name", "mask")) {
        continue;
      }
      if (energy || !text(output, "name", "energy") ||
          !flag(output, "reducible", true) || !flag(output, "atomic", true) ||
          !flag(output, "intensive", false) ||
          !flag(output, "r_differentiable", true) ||
          !flag(output, "c_differentiable", true)) {
        return false;
      }
      energy = true;
    }
    return energy;
  } catch (const std::exception&) {
    return false;
  }
}
}  // namespace ptexpt
}  // namespace deepmd
