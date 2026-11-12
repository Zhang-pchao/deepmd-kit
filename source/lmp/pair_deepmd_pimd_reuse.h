// SPDX-License-Identifier: LGPL-3.0-or-later
// Template implementation included by pair_deepmd_kokkos.cpp only.
// The graph and inference adapters contain no integrator or bias policy.
// This layer owns collective snapshots and scatters complete PES derivatives.
template <class DeviceType>
void PairDeepMDKokkos<DeviceType>::settings(int narg, char** arg) {
  reuse_fix_id.clear();
  reuse_mode = 0;
  reuse_stats.fill(0.0);
  reuse_calls.fill(0.0);
  reuse_profile.fill(0.0);
  selected_evaluator.reset_profile();
  reuse_fallback_reason.clear();
  bool device_graph = false, graph_option_seen = false;
  bool graph_profile = false, profile_option_seen = false;
  std::vector<char*> ordinary;
  for (int i = 0; i < narg; ++i) {
    if (std::string(arg[i]) == "pimd_graph_profile") {
      if (profile_option_seen || i + 1 >= narg) {
        error->all(FLERR, "deepmd/kk expects one pimd_graph_profile yes|no");
      }
      profile_option_seen = true;
      const std::string value = arg[++i];
      if (value != "yes" && value != "no") {
        error->all(FLERR, "deepmd/kk pimd_graph_profile must be yes or no");
      }
      graph_profile = value == "yes";
      continue;
    }
    if (std::string(arg[i]) == "pimd_graph") {
      if (graph_option_seen || i + 1 >= narg) {
        error->all(FLERR, "deepmd/kk expects one pimd_graph host|device");
      }
      graph_option_seen = true;
      const std::string value = arg[++i];
      if (value != "host" && value != "device") {
        error->all(FLERR, "deepmd/kk pimd_graph must be host or device");
      }
      device_graph = value == "device";
      continue;
    }
    if (std::string(arg[i]) != "pimd_reuse") {
      ordinary.push_back(arg[i]);
      continue;
    }
    if (!reuse_fix_id.empty() || i + 2 >= narg) {
      error->all(FLERR,
                 "deepmd/kk expects one pimd_reuse FIX full|verify|fallback");
    }
    reuse_fix_id = arg[++i];
    const std::string mode = arg[++i];
    if (mode == "verify") {
      reuse_mode = 1;
    } else if (mode == "fallback") {
      reuse_mode = 2;
    } else if (mode != "full") {
      error->all(FLERR, "Unknown deepmd/kk pimd_reuse mode");
    }
  }
  if (graph_option_seen && reuse_fix_id.empty()) {
    error->all(FLERR, "deepmd/kk pimd_graph requires pimd_reuse FIX MODE");
  }
  if (graph_profile && (!device_graph || reuse_fix_id.empty())) {
    error->all(
        FLERR,
        "pimd_graph_profile yes requires pimd_graph device and pimd_reuse");
  }
  selected_evaluator.enable_graph_profile(graph_profile);
  selected_evaluator.use_device_graph(device_graph);
  if (!reuse_fix_id.empty()) {
    int low = reuse_mode + 4 * device_graph + 8 * graph_profile, high = low;
    MPI_Allreduce(MPI_IN_PLACE, &low, 1, MPI_INT, MPI_MIN, universe->uworld);
    MPI_Allreduce(MPI_IN_PLACE, &high, 1, MPI_INT, MPI_MAX, universe->uworld);
    if (low != high || ordinary.empty()) {
      error->all(FLERR,
                 "pimd_reuse requires identical modes/graph backends and a "
                 "model in every partition");
    }
    // .pt2 loading is file based. Compare complete bytes once before loading,
    // not just names or metadata, and do not cache any coordinate result here.
    std::ifstream model(ordinary[0], std::ios::binary);
    int invalid = !model;
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX,
                  universe->uworld);
    if (invalid) {
      error->all(FLERR, "pimd_reuse cannot read a model archive");
    }
    std::vector<char> local(65536), shared(65536);
    while (true) {
      model.read(local.data(), local.size());
      const int count = static_cast<int>(model.gcount());
      int root_count = count;
      if (universe->me == 0) {
        shared = local;
      }
      MPI_Bcast(&root_count, 1, MPI_INT, 0, universe->uworld);
      MPI_Bcast(shared.data(), root_count, MPI_CHAR, 0, universe->uworld);
      invalid = model.bad() || count != root_count ||
                !std::equal(local.begin(), local.begin() + root_count,
                            shared.begin());
      MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX,
                    universe->uworld);
      if (invalid) {
        error->all(FLERR,
                   "pimd_reuse requires byte-identical model archives in every "
                   "partition");
      }
      if (root_count == 0) {
        break;
      }
    }
  }
  PairDeepMD::settings(static_cast<int>(ordinary.size()), ordinary.data());
}

template <class DeviceType>
void* PairDeepMDKokkos<DeviceType>::extract(const char* name, int& dim) {
  if (std::string(name) == "pimd_reuse_device_graph_profile") {
    dim = 1;
    return const_cast<double*>(selected_evaluator.graph_profile().data());
  }
  if (std::string(name) == "pimd_reuse_calls") {
    dim = 1;
    return reuse_calls.data();
  }
  if (std::string(name) == "pimd_reuse_stats") {
    dim = 1;
    return reuse_stats.data();
  }
  if (std::string(name) == "pimd_reuse_profile") {
    dim = 1;
    return reuse_profile.data();
  }
  if (std::string(name) == "pimd_reuse_model_profile") {
    dim = 1;
    return const_cast<double*>(selected_evaluator.profile().data());
  }
  return PairDeepMD::extract(name, dim);
}

template <class DeviceType>
bool PairDeepMDKokkos<DeviceType>::compute_pimd_reuse() {
#ifndef DEEPMD_HAS_PIMD_PES_VIEW
  error->all(FLERR,
             "PIMD selected energy requires the PIMD PES view interface");
  return false;
#else
  using namespace deepmd::selected_energy;
  const double started = MPI_Wtime();
  double phase_started = started;
  auto mark_phase = [&](int phase) {
    const double now = MPI_Wtime();
    reuse_profile[phase] += now - phase_started;
    phase_started = now;
  };
  ++reuse_stats[0];
  auto collective_error = [&](const std::string& message) {
    int invalid = !message.empty();
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX,
                  universe->uworld);
    if (invalid) {
      if (!message.empty()) {
        fprintf(stderr, "deepmd/kk pimd_reuse rank %d: %s\n", universe->me,
                message.c_str());
      }
      error->all(FLERR, "Collective deepmd/kk pimd_reuse validation failed");
    }
  };
  auto fallback = [&](int reason, const char* message) {
    MPI_Allreduce(MPI_IN_PLACE, &reason, 1, MPI_INT, MPI_MAX, universe->uworld);
    if (!reason) {
      return false;
    }
    if (reuse_mode == 1) {
      collective_error(message);
    }
    ++reuse_stats[2];
    reuse_stats[11] = reason;
    if (reuse_fallback_reason != message && comm->me == 0) {
      if (screen) {
        fprintf(screen, "deepmd/kk pimd_reuse fallback: %s\n", message);
      }
      if (logfile) {
        fprintf(logfile, "deepmd/kk pimd_reuse fallback: %s\n", message);
      }
    }
    reuse_fallback_reason = message;
    reuse_stats[10] += MPI_Wtime() - started;
    return true;
  };
  int unsupported = !deep_pot.supports_selected_energy() || domain->triclinic ||
                    eflag_atom || vflag_atom || cvflag_atom || dim_fparam ||
                    dim_aparam || do_compute_aparam || do_compute_fparam ||
                    do_fix_fparam || do_ttm || !fparam.empty() ||
                    !aparam.empty();
  for (const int type : type_idx_map) {
    unsupported |= type < 0;
  }
  for (int d = 0; d < 3; ++d) {
    unsupported |= domain->periodicity[d] && domain->prd[d] <= 2 * cutoff;
  }
  if (fallback(unsupported ? 1 : 0,
               "unsupported model, cell, NULL type, parameters or per-atom "
               "output")) {
    return false;
  }
  auto* fix = modify->get_fix_by_id(reuse_fix_id);
  collective_error(fix ? "" : "PIMD fix was removed");
  int dim = 0;
  auto* view = static_cast<PIMDPESView*>(fix->extract("pimd_pes_view_v1", dim));
  int invalid = !view;
  if (view) {
    invalid |= view->version != PIMDPESView::VERSION ||
               view->beads != universe->nworlds ||
               view->bead != universe->iworld || view->nlocal != atom->nlocal ||
               view->natoms != atom->natoms || atom->natoms <= 0 ||
               atom->natoms > std::numeric_limits<int>::max() / 3 ||
               view->tags.size() != static_cast<std::size_t>(atom->nlocal) ||
               view->centroid.size() != 3 * view->tags.size() ||
               view->amplitude.size() != view->tags.size();
  }
  collective_error(invalid ? "Invalid or incompatible PIMD PES view" : "");
  const int n = static_cast<int>(atom->natoms);
  // The current native PIMD view requires consecutive IDs. Reduce only owned
  // rows; empty ranks participate and ghost derivatives are never double added.
  atomKK->sync(Host, X_MASK | TYPE_MASK | TAG_MASK);
  std::vector<double> x(3 * n, 0.0), c(3 * n, 0.0), amplitude(n, 0.0);
  std::vector<int> counts(n, 0), types(n, 0);
  invalid = 0;
  for (int i = 0; i < atom->nlocal; ++i) {
    const tagint tag = atom->tag[i];
    if (tag < 1 || tag > n || view->tags[i] != tag) {
      invalid = 1;
      continue;
    }
    const int node = static_cast<int>(tag - 1);
    ++counts[node];
    const int type = atom->type[i] - 1;
    if (type < 0 || type >= static_cast<int>(type_idx_map.size())) {
      invalid = 1;
      continue;
    }
    types[node] = type_idx_map[type];
    amplitude[node] = view->amplitude[i];
    invalid |= !std::isfinite(amplitude[node]) || amplitude[node] < 0 ||
               amplitude[node] > 1;
    for (int d = 0; d < 3; ++d) {
      x[3 * node + d] =
          (atom->x[i][d] - domain->boxlo[d]) / dist_unit_cvt_factor;
      c[3 * node + d] =
          (view->centroid[3 * i + d] - domain->boxlo[d]) / dist_unit_cvt_factor;
      invalid |=
          !std::isfinite(x[3 * node + d]) || !std::isfinite(c[3 * node + d]);
    }
  }
  collective_error(invalid ? "Invalid owned atom/tag/type/coordinate" : "");
  MPI_Allreduce(MPI_IN_PLACE, counts.data(), n, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, types.data(), n, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, x.data(), 3 * n, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, c.data(), 3 * n, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, amplitude.data(), n, MPI_DOUBLE, MPI_SUM, world);
  for (int count : counts) {
    invalid |= count != 1;
  }
  collective_error(invalid ? "Each PIMD tag must have exactly one owner" : "");
  auto root_c = c, root_amplitude = amplitude;
  auto root_types = types;
  MPI_Bcast(root_c.data(), 3 * n, MPI_DOUBLE, 0, universe->uworld);
  MPI_Bcast(root_amplitude.data(), n, MPI_DOUBLE, 0, universe->uworld);
  MPI_Bcast(root_types.data(), n, MPI_INT, 0, universe->uworld);
  invalid = types != root_types;
  for (int i = 0; i < 3 * n; ++i) {
    invalid |= std::abs(c[i] - root_c[i]) > 1e-12 * (1 + std::abs(root_c[i]));
  }
  for (int i = 0; i < n; ++i) {
    invalid |= std::abs(amplitude[i] - root_amplitude[i]) > 1e-12;
  }
  collective_error(invalid ? "Partition centroid, amplitude or types differ"
                           : "");
  mark_phase(0);
  // Preserve support even at zero instantaneous displacement. A union protects
  // tiny nonzero amplitudes when roundoff differs across partitions.
  std::vector<int> support, selected(n, 0);
  for (int i = 0; i < n; ++i) {
    selected[i] = amplitude[i] > 0;
  }
  MPI_Allreduce(MPI_IN_PLACE, selected.data(), n, MPI_INT, MPI_MAX,
                universe->uworld);
  for (int i = 0; i < n; ++i) {
    if (selected[i]) {
      support.push_back(i);
    }
  }
  std::fill(selected.begin(), selected.end(), 0);
  std::vector<Position> coordinates(n), centers(n);
  Position box;
  std::array<bool, 3> periodic;
  for (int d = 0; d < 3; ++d) {
    box[d] = domain->prd[d] / dist_unit_cvt_factor;
    periodic[d] = domain->periodicity[d];
    for (int i = 0; i < n; ++i) {
      coordinates[i][d] = x[3 * i + d];
      centers[i][d] = root_c[3 * i + d];
    }
  }
  std::string failure;
  std::unique_ptr<Neighborhood> current, reference;
  try {
    // Leaders do inference and neighbor search; other ranks only participate
    // in ownership and result collectives.
    if (comm->me == 0) {
      current.reset(
          new Neighborhood(coordinates, box, periodic, deep_pot.cutoff()));
      reference.reset(
          new Neighborhood(centers, box, periodic, deep_pot.cutoff()));
      for (const int owner :
           selected_evaluator.affected_owners(*current, support)) {
        selected[owner] = 1;
      }
      // C and the quantum support are shared across partitions. Evaluate its
      // reverse dependencies once; the collective below unions all X_b and C.
      if (universe->me == 0) {
        for (const int owner :
             selected_evaluator.affected_owners(*reference, support)) {
          selected[owner] = 1;
        }
      }
    }
  } catch (const std::exception& e) {
    failure = e.what();
  }
  collective_error(failure);
  MPI_Allreduce(MPI_IN_PLACE, selected.data(), n, MPI_INT, MPI_MAX,
                universe->uworld);
  std::vector<int> owners, all(n);
  std::iota(all.begin(), all.end(), 0);
  for (int i = 0; i < n; ++i) {
    if (selected[i]) {
      owners.push_back(i);
    }
  }
  reuse_stats[6] = owners.size();
  if (reuse_mode == 2 && fallback(owners.size() == all.size() ? 2 : 0,
                                  "selected energy covers every atom")) {
    return false;
  }

  mark_phase(1);
  Evaluation residual;
  residual.force.resize(3 * n);
  double reference_energies[2] = {};
  failure.clear();
  try {
    if (universe->me == 0) {
      ++reuse_calls[2];
      ++reuse_calls[7];
      reuse_calls[3] += !owners.empty();
      ++reuse_calls[7];
      auto reference_pair = selected_evaluator.evaluate_reference_pair(
          deep_pot, *reference, owners, types);
      residual = std::move(reference_pair[0]);
      const auto& selected_reference = reference_pair[1];
      reference_energies[0] = residual.energy;
      reference_energies[1] = selected_reference.energy;
      residual.energy -= selected_reference.energy;
      for (int i = 0; i < 3 * n; ++i) {
        residual.force[i] -= selected_reference.force[i];
      }
      for (int d = 0; d < 9; ++d) {
        residual.virial[d] -= selected_reference.virial[d];
      }
    }
  } catch (const std::exception& e) {
    failure = e.what();
  }
  collective_error(failure);
  MPI_Bcast(&residual.energy, 1, MPI_DOUBLE, 0, universe->uworld);
  MPI_Bcast(residual.force.data(), 3 * n, MPI_DOUBLE, 0, universe->uworld);
  MPI_Bcast(residual.virial.data(), 9, MPI_DOUBLE, 0, universe->uworld);
  MPI_Bcast(reference_energies, 2, MPI_DOUBLE, 0, universe->uworld);

  mark_phase(2);
  Evaluation result;
  result.force.resize(3 * n);
  double work[2] = {};
  failure.clear();
  try {
    if (comm->me == 0) {
      reuse_calls[4] += !owners.empty();
      ++reuse_calls[7];
      result = selected_evaluator.evaluate(deep_pot, *current, owners, types);
      work[0] = result.nodes;
      work[1] = result.edges;
      const double delta = result.energy - reference_energies[1];
      result.energy += residual.energy;
      for (int i = 0; i < 3 * n; ++i) {
        result.force[i] += residual.force[i];
      }
      for (int d = 0; d < 9; ++d) {
        result.virial[d] += residual.virial[d];
      }
      if (reuse_mode == 1) {
        ++reuse_calls[5];
        ++reuse_calls[7];
        const auto full =
            selected_evaluator.evaluate(deep_pot, *current, all, types);
        bool mismatch =
            std::abs(result.energy - full.energy) / n >
                1e-5 + 2e-7 * std::abs(full.energy / n) ||
            std::abs(delta - (full.energy - reference_energies[0])) >
                1e-3 + 1e-5 * std::abs(full.energy - reference_energies[0]);
        for (int i = 0; i < 3 * n; ++i) {
          mismatch |= std::abs(result.force[i] - full.force[i]) >
                      5e-4 + 1e-4 * std::abs(full.force[i]);
        }
        for (int d = 0; d < 9; ++d) {
          mismatch |= std::abs(result.virial[d] - full.virial[d]) >
                      1e-3 + 1e-4 * std::abs(full.virial[d]);
        }
        if (mismatch) {
          throw std::runtime_error(
              "Full/reuse energy, correction, force or virial gate failed");
        }
      }
    }
  } catch (const std::exception& e) {
    failure = e.what();
  }
  collective_error(failure);
  MPI_Bcast(&result.energy, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(result.force.data(), 3 * n, MPI_DOUBLE, 0, world);
  MPI_Bcast(result.virial.data(), 9, MPI_DOUBLE, 0, world);
  MPI_Bcast(work, 2, MPI_DOUBLE, 0, world);
  mark_phase(3);
  atomKK->sync(Host, F_MASK);
  const double fscale = scale[1][1] * force_unit_cvt_factor;
  for (int i = 0; i < atom->nlocal; ++i) {
    const int node = static_cast<int>(atom->tag[i] - 1);
    for (int d = 0; d < 3; ++d) {
      atom->f[i][d] += fscale * result.force[3 * node + d];
    }
  }
  atomKK->modified(Host, F_MASK);
  atomKK->sync(execution_space, F_MASK);
  // World thermo reductions count these once, before the PIMD bead average.
  if (comm->me == 0) {
    if (eflag_global) {
      eng_vdwl += scale[1][1] * ener_unit_cvt_factor * result.energy;
    }
    if (vflag_global) {
      const int component[6] = {0, 4, 8, 3, 6, 7};
      for (int d = 0; d < 6; ++d) {
        virial[d] +=
            scale[1][1] * ener_unit_cvt_factor * result.virial[component[d]];
      }
    }
  }
  if (reuse_stats[1] == 0 && comm->me == 0) {
    if (screen) {
      fprintf(screen,
              "deepmd/kk pimd_reuse active: mode=%s atoms=%d support=%zu "
              "owners=%zu nodes=%.0f edges=%.0f\n",
              reuse_mode == 1 ? "verify" : "fallback", n, support.size(),
              owners.size(), work[0], work[1]);
    }
    if (logfile) {
      fprintf(logfile,
              "deepmd/kk pimd_reuse active: mode=%s atoms=%d support=%zu "
              "owners=%zu nodes=%.0f edges=%.0f\n",
              reuse_mode == 1 ? "verify" : "fallback", n, support.size(),
              owners.size(), work[0], work[1]);
    }
  }
  mark_phase(4);
  ++reuse_stats[1];
  ++reuse_stats[3];
  reuse_stats[4] += !owners.empty();
  reuse_stats[5] += !owners.empty();
  reuse_stats[7] = work[0];
  reuse_stats[8] = work[1];
  reuse_stats[9] += reuse_mode == 1;
  reuse_stats[10] += MPI_Wtime() - started;
  reuse_stats[11] = 0;
  return true;
#endif
}
