// SPDX-License-Identifier: LGPL-3.0-or-later
// Model-independent C-library runner for check_pimd_reuse.py.
#include <mpi.h>
#define LAMMPS_LIB_MPI
#include <cstdio>
#include <cstring>

#include "library.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const char* input = nullptr;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "-in") == 0) {
      input = argv[i + 1];
    }
  }
  if (!input) {
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  void* lmp = lammps_open(argc, argv, MPI_COMM_WORLD, nullptr);
  lammps_file(lmp, input);
  int failed = lammps_has_error(lmp);
  int any_failed = failed;
  MPI_Allreduce(MPI_IN_PLACE, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (any_failed) {
    if (failed) {
      char message[2048] = {};
      lammps_get_last_error_message(lmp, message, sizeof(message));
      std::fprintf(stderr, "PIMD_REUSE_INPUT_ERROR rank=%d %s\n", rank,
                   message);
    }
    lammps_close(lmp);
    lammps_kokkos_finalize();
    MPI_Finalize();
    return 1;
  }

  auto* stats =
      static_cast<double*>(lammps_extract_pair(lmp, "pimd_reuse_stats"));
  if (!stats) {
    MPI_Abort(MPI_COMM_WORLD, 3);
  }
  // One line per rank fits normal stdio buffering; parser checks every rank.
  char line[1024];
  int offset = std::snprintf(line, sizeof(line), "PIMD_REUSE_STATS %d", rank);
  for (int i = 0; i < 12; ++i) {
    offset +=
        std::snprintf(line + offset, sizeof(line) - offset, " %.17g", stats[i]);
  }
  std::printf("%s\n", line);
  auto* calls =
      static_cast<double*>(lammps_extract_pair(lmp, "pimd_reuse_calls"));
  if (calls) {
    offset = std::snprintf(line, sizeof(line), "PIMD_REUSE_CALLS %d", rank);
    for (int i = 0; i < 8; ++i) {
      offset += std::snprintf(line + offset, sizeof(line) - offset, " %.17g",
                              calls[i]);
    }
    std::printf("%s\n", line);
  }
  for (int group = 0; group < 2; ++group) {
    const char* name =
        group == 0 ? "pimd_reuse_profile" : "pimd_reuse_model_profile";
    const int size = group == 0 ? 5 : 4;
    auto* values = static_cast<double*>(lammps_extract_pair(lmp, name));
    if (values) {
      offset = std::snprintf(line, sizeof(line), "PIMD_REUSE_PROFILE %d %s",
                             rank, name);
      for (int i = 0; i < size; ++i) {
        offset += std::snprintf(line + offset, sizeof(line) - offset, " %.17g",
                                values[i]);
      }
      std::printf("%s\n", line);
    }
  }
  auto* graph_profile = static_cast<double*>(
      lammps_extract_pair(lmp, "pimd_reuse_device_graph_profile"));
  if (graph_profile) {
    offset = std::snprintf(line, sizeof(line),
                           "PIMD_REUSE_DEVICE_GRAPH_PROFILE %d", rank);
    for (int i = 0; i < 8; ++i) {
      offset += std::snprintf(line + offset, sizeof(line) - offset, " %.17g",
                              graph_profile[i]);
    }
    std::printf("%s\n", line);
  }
  lammps_close(lmp);
  lammps_kokkos_finalize();
  MPI_Finalize();
  return 0;
}
