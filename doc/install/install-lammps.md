# Install LAMMPS

An AI agent can build LAMMPS with the DeePMD-kit module and Kokkos. DPA4/SeZM
and DPA4C use `pair_style deepmd/kk`; compressed native-spin DPA4C models use
`pair_style dpa4spin/kk`. See [Install with an AI agent](install-with-agent.md).

There are two ways to install LAMMPS: the built-in mode and the plugin mode. The built-in mode builds LAMMPS along with the DeePMD-kit and DeePMD-kit will be loaded automatically when running LAMMPS. The plugin mode builds LAMMPS and a plugin separately, so one needs to use `plugin load` command to load the DeePMD-kit's LAMMPS plugin library.

## Install LAMMPS's DeePMD-kit module (built-in mode)

Before following this section, [DeePMD-kit C++ interface](install-from-source.md) should have be installed.

DeePMD-kit provides a module for running MD simulations with LAMMPS. Now make the DeePMD-kit module for LAMMPS.

```bash
cd $deepmd_source_dir/source/build
make lammps
```

DeePMD-kit will generate a module called `USER-DEEPMD` in the `build` directory, which supports either double or single float precision interface. Now download the LAMMPS code, and uncompress it.

```bash
cd /some/workspace
wget https://github.com/lammps/lammps/archive/stable_22Jul2025_update2.tar.gz
tar xf stable_22Jul2025_update2.tar.gz
```

The source code of LAMMPS is stored in the directory `lammps-stable_22Jul2025_update2`.

Then, you can [build LAMMPS](https://docs.lammps.org/Build.html) with either make or CMake.

### With make

Now go into the LAMMPS code and copy the DeePMD-kit module like this

```bash
cd lammps-stable_22Jul2025_update2/src/
cp -r $deepmd_source_dir/source/build/USER-DEEPMD .
make yes-kspace
make yes-extra-fix
make yes-user-deepmd
```

You can enable any other package you want. Now build LAMMPS

```bash
make mpi -j4
```

If everything works fine, you will end up with an executable `lmp_mpi`.

```bash
./lmp_mpi -h
```

The DeePMD-kit module can be removed from the LAMMPS source code by

```bash
make no-user-deepmd
```

### With CMake

Now go into the LAMMPS directory and create a directory called `build`:

```bash
mkdir -p lammps-stable_22Jul2025_update2/build/
cd lammps-stable_22Jul2025_update2/build/
```

Patch the LAMMPS `CMakeLists.txt` file:

```bash
echo "include(${deepmd_source_dir}/source/lmp/builtin.cmake)" >> ../cmake/CMakeLists.txt
```

It's expected to see one extra line in the end of `CMakeLists.txt`.

Now build LAMMPS. You can install any other package you want.

```bash
cmake -D LAMMPS_INSTALL_RPATH=ON -D BUILD_SHARED_LIBS=yes -D CMAKE_INSTALL_PREFIX=${deepmd_root} -DCMAKE_PREFIX_PATH=${deepmd_root} ../cmake
make -j4
make install
```

To build the GPU-resident pair styles `deepmd/kk` and `dpa4spin/kk`, enable the
LAMMPS Kokkos package. The pair styles use the DeePMD C API:

```bash
cmake -D PKG_KOKKOS=ON \
    -D LAMMPS_INSTALL_RPATH=ON -D BUILD_SHARED_LIBS=yes \
    -D CMAKE_INSTALL_PREFIX=${deepmd_root} \
    -D CMAKE_PREFIX_PATH=${deepmd_root} ../cmake
```

If everything works fine, you will end up with an executable `${deepmd_root}/bin/lmp`.

```bash
${deepmd_root}/bin/lmp -h
```

## Install LAMMPS (plugin mode)

Starting from `8Apr2021`, LAMMPS also provides a plugin mode, allowing one to build LAMMPS and a plugin separately.

Now download the LAMMPS code (`8Apr2021` or later), and uncompress it:

```bash
cd /some/workspace
wget https://github.com/lammps/lammps/archive/stable_22Jul2025_update2.tar.gz
tar xf stable_22Jul2025_update2.tar.gz
```

The source code of LAMMPS is stored in the directory `lammps-stable_22Jul2025_update2`. The directory of the source code should be specified as the CMAKE argument `LAMMPS_SOURCE_ROOT` during installation of the DeePMD-kit C++ interface. Now go into the LAMMPS directory and create a directory called `build`

```bash
mkdir -p lammps-stable_22Jul2025_update2/build/
cd lammps-stable_22Jul2025_update2/build/
```

Now build LAMMPS. Note that `PLUGIN` must be enabled, and `BUILD_SHARED_LIBS` must be set to `yes`. You can install any other package you want.

```bash
cmake -D PKG_PLUGIN=ON -D LAMMPS_INSTALL_RPATH=ON -D BUILD_SHARED_LIBS=yes -D CMAKE_INSTALL_PREFIX=${deepmd_root} -D CMAKE_INSTALL_LIBDIR=lib -D CMAKE_INSTALL_FULL_LIBDIR=${deepmd_root}/lib ../cmake
make -j4
make install
```

Configure LAMMPS with `PKG_KOKKOS=ON`, then configure the DeePMD plugin with
`DEEPMD_LAMMPS_KOKKOS=ON` and `Kokkos_DIR` pointing to the Kokkos package
exported by the LAMMPS build. This registers the GPU-resident pair styles
`deepmd/kk` and `dpa4spin/kk`; the plugin and built-in integrations use the same
C API device-graph interface.

If everything works fine, you will end up with an executable `${deepmd_root}/bin/lmp`.

```bash
${deepmd_root}/bin/lmp -h
```

> [!NOTE]
> If `${tensorflow_root}`, `${deepmd_root}`, or the path to TensorFlow Python package if applicable is different from the prefix of LAMMPS, you need to append the library path to [`RUNPATH`](https://man7.org/linux/man-pages/man8/ld.so.8.html) of `liblammps.so`. For example, use patchelf >= 0.13
>
> ```sh
> patchelf --add-rpath "${tensorflow_root}/lib" liblammps.so
> ```

## MPI for runs on several GPUs

With the KOKKOS package, each MPI rank of LAMMPS uses one GPU and keeps its atom data in GPU memory, and the ranks exchange atom data at every step. Build LAMMPS and the DeePMD-kit C++ interface against the same MPI library, one that can access GPU memory (CUDA-aware MPI), such as Open MPI with UCX. When LAMMPS detects an MPI library without CUDA support, such as the MPICH package of conda-forge, it turns GPU-aware communication off and copies the positions and forces through host memory at every step. An MPI library that LAMMPS cannot identify is assumed to support CUDA; if it does not, run LAMMPS with `-pk kokkos gpu/aware off`.

The Open MPI package of conda-forge is built with CUDA support but disables it by default; `ompi_info --all | grep opal_built_with_cuda_support` shows whether an Open MPI installation was built with it. Install the package in an environment of its own:

```bash
conda create -n openmpi -c conda-forge openmpi ucx
```

Configure LAMMPS and the DeePMD-kit C++ interface in new build directories with both compiler wrappers of that environment, `-D MPI_C_COMPILER=${mpi_root}/bin/mpicc -D MPI_CXX_COMPILER=${mpi_root}/bin/mpicxx`, where `${mpi_root}` is its prefix; CMake finds the MPI of each language separately. Launch a run on several GPUs with the `mpirun` of that installation, enabling its CUDA support for the launch:

```bash
mpirun --mca opal_cuda_support 1 -x UCX_MEMTYPE_CACHE=n -np 2 lmp -k on g 2 -sf kk -in in.lammps
```

With `UCX_MEMTYPE_CACHE=n`, UCX queries the memory type of every buffer instead of taking it from a cache that can report a GPU buffer as host memory and make UCX abort.

Launch a run on a single MPI rank without these options: it exchanges no atom data, and the CUDA support adds a memory-type query to every MPI call, which can lengthen the step of a small system by more than 10%.

### Experimental short-range PIMD energy reuse

The interface uses C API 31 and the automatic byte-word canonical index schema.
The current local graph builder uses 32-bit node indices and 64-bit CSR offsets
and source-order positions within its admitted size range. Export compressed
models again for this schema before using the local evaluator.

The selected-energy capability query extends the project's DeepPot backend
interface. Its factories use the explicit
v2_selected_energy_v1 suffix. Rebuild the C/C++ interface and backend together.
The loader rejects a stock v2 factory and may then locate the matching project
backend among its remaining candidates. Record the loaded provider and backend
paths when validating an installation.


A compatible LAMMPS build exposing version 1 of the collective
`pimd_pes_view_v1` interface may use the optional GPU pair keyword:

```lammps
pair_style deepmd/kk model.pt2 pimd_reuse pi verify
pair_coeff * * O H
fix pi all pimd/adaptive temp 300 center 10 10 10 core 2 outer 5 initialize auxiliary
```

Enable the same reuse option/mode in every partition. Complete model archive
bytes are compared collectively once before loading; different weights are
rejected even when metadata match. The pair does not define the adaptive integrator. The named fix must exist when
initialization starts. Omitting `pimd_reuse`, or selecting `full`, retains the
ordinary complete-bead inference path. `verify` evaluates both the split and
complete potential on every force refresh and rejects mismatched energy,
correction energy, force or virial. It is a correctness mode, not a benchmark.
`fallback` uses the split when admitted and evaluates the ordinary complete
potential when the selected set covers all atoms or the model/cell/request is
unsupported. The reason is logged. A validation failure never silently discards
a force or continues with an incomplete graph.

For a short-range atomic decomposition, the implemented identity is
`E(C) - E_A(C) + mean_b E_A(X_b)`. The same loaded model is used in all terms.
The selected owners cover every energy influenced by any nonzero path amplitude,
using the union of the centroid and every bead geometry. Their full neighbor
environments are retained, and forces on all halo atoms are scattered back.
The selection is rebuilt at every force call, including repeated calls at the
same timestep. It is not a cached force, a separate force field, or a physical
boundary on the reaction.

This initial adapter admits audited, one-hop DPA4C canonical FP32 exported
models with extensive differentiable atomic energies, no message passing, spin,
conditional inputs, exclusions, bridging or unrecognized extra energy terms.
It requires an orthorhombic fixed cell whose periodic dimensions exceed twice
the cutoff, consecutive stable atom IDs, and no NULL atom types. Per-atom
energy/virial requests use the full path in fallback mode and are rejected in
verify mode. Full node ownership is checked collectively, including empty ranks.
MPI partition leaders perform local inference; one universe leader performs
the shared reference. No PLUMED history or bias operation occurs in this layer.
Use the integrator's existing force pullback and bias conventions unchanged.
The reported PES virial does not establish adaptive NPT support.

The pair extract key `pimd_reuse_stats` returns twelve doubles: attempted calls,
split calls, fallback calls, shared full-reference calls, nonempty selected
reference calls, nonempty bead-local calls, last owner count, last local node
count, last local edge count, verification full-bead calls, accumulated adapter
seconds, and last fallback reason (0 none, 1 unsupported request, 2 full coverage).
Counts are replicated diagnostics; do not sum shared-reference counts over
ranks or beads. Adapter time excludes ordinary fallback inference. Reduced
graph work is not proof of a whole-step speedup.

Validate model/backend parity before dynamics, then use a small system and
matched full/verify/fallback runs. Regional path mapping has its own sampling
ensemble; exact PES reuse neither changes that ensemble nor proves full-quantum
core statistics, model accuracy, or converged reaction free energies.

The small-system plugin regression accepts caller-supplied model and atomic data
(1..174 atoms). Compile its model-independent C-library driver with the same MPI
and LAMMPS installation, then run the Python test in an allocation:

```bash
mpicxx -std=c++17 -I /path/to/lammps/include/lammps \
  source/lmp/plugin/tests/pimd_reuse_driver.cpp \
  -L /path/to/lammps/lib -llammps -o pimd_reuse_driver
python source/lmp/plugin/tests/check_pimd_reuse.py \
  --driver ./pimd_reuse_driver --plugin /path/to/libdeepmd_lmp.so \
  --model /path/to/model.pt2 --data /path/to/small.data --type-map O H \
  --center 10 10 10 --core 0.8 --outer 3 --empty-center 45 45 45 \
  --all-center 22 10 10 --all-core 15 --all-outer 16 \
  --output /path/to/new-test-results
```

Choose regions for the supplied geometry: local must select a strict nonempty
subset, empty must select none, and all must include every atom. The last
layout splits the box along z to exercise empty-rank ownership with a compact
fixture. Each run compares physical/auxiliary endpoints and energy and checks
actual inference counters. Optional `--only-case p4-r1-local` selects one case;
`--plumed-input /path/to/plumed.dat --bias-mode cv_mean --contraction 0.5`
adds an externally supplied, compatible bias. The other bias modes are
`coordinate_mean` and `bias_mean`; contraction applies only to `cv_mean`.
Set the timestep before constructing any PLUMED fix. With TIME=ps and
PRINT STRIDE=1 FILE=COLVAR, add --check-plumed-time to check the final clock.
These short tests are interface regressions, not physical equilibration.


The optional reuse diagnostics expose `pimd_reuse_profile` (five cumulative
per-rank seconds: snapshot/validation, affected-owner search, reference phase,
local phase, scatter/tally) and `pimd_reuse_model_profile` (four nested cumulative
seconds: compact graph construction, host-to-device staging, synchronized model
inference, device-to-host retrieval/scatter). The reference/local phases include
collective waiting. Model timings are nested inside these phases, so do not add
the two arrays or sum ranks to claim wall time. Counters reset with pair settings;
no extra synchronization or model evaluation is inserted by profiling. The full
mode bypasses reuse and leaves these arrays zero. Early fallbacks are counted in
`pimd_reuse_stats` and may leave incomplete phase totals.


For auditable benchmark accounting, `pimd_reuse_calls` returns eight cumulative
per-rank doubles: force-call sequence, ordinary full-bead calls, full-reference
model calls, nonempty selected-reference model calls, nonempty selected-bead
model calls, verification full calls, actual fallback calls, and selected graph
builds. These count the executing rank, unlike the replicated legacy stats.
Only the universe leader executes reference calls; only partition leaders
execute selected-bead calls. Force-call sequence advances even for repeated
force evaluations at one MD timestep. Empty selections build a graph but do
not invoke the model. In pure reuse timing, full-bead, verification and actual
fallback counts must all be zero. The `fallback` keyword permits fallback; its
name alone does not establish that a fallback was executed. Counters reset when
pair settings are read and introduce no extra model call or synchronization.


The optional `pimd_graph host|device` selector controls graph preparation for
selected-energy calls; `host` preserves the independent CPU reference and is
the default. `device` uses Kokkos cell bins and the same canonical dual-CSR
model interface, retaining all halo forces. It requires an explicit
`pimd_reuse FIX MODE` and the same selection in every partition:

```lammps
pair_style deepmd/kk model.pt2 pimd_reuse pi verify pimd_graph device
```

First qualify with `verify`, then use `fallback` for matched timing and inspect
actual call counts. The device option does not change the model, physical
region, path amplitudes, bias, or supported cell/output contract. A full-mode
control still uses the native complete-bead path. Device graphs are rebuilt at
every force refresh; capacity reuse does not mean topology or force caching.
The separate reference geometry must have its own covered neighbors; the
current bead neighbor list is not assumed valid for the centroid. Node and
edge ordering is deterministic and the CPU builder remains a test oracle.

The device backend also performs the complete reverse-dependency query. The
centroid membership query runs only on the universe leader; a collective union
includes every current bead geometry. For the two centroid energy evaluations,
the selected graph extracts complete rows from the full graph built in the same
call. This sharing does not persist into the next force refresh. Spatial bins
use contiguous CSR storage, cooperative neighbor traversal and deterministic
team sorting. The graph-build counter counts model-input graphs, including
same-call row extraction; membership-only queries invoke no model.

GPU graph preparation is optional and is not a speedup guarantee. Benchmark
complete steps with the same GPU count, MPI mapping, MPS setting, precision,
model, region, outputs and bias. Many small kernels shared by multiple CUDA
processes can outweigh fewer selected energy evaluations. If enabling MPS,
use the site's job-isolated setup and compare both backends with MPS enabled;
do not count the scheduling improvement as an algorithmic speedup.

The standalone graph regression uses CTest and Kokkos, with either an installed
GTest package or a caller-provided existing `GTEST_SOURCE_DIR` checkout. It does
not download dependencies. Configure `source/lmp/plugin/tests/selected_graph`
with the same compiler/Kokkos configuration as the plugin, then run CTest in
that build directory. Tests compare full/empty/selected graphs, dual CSR,
periodic/mixed/open boundaries, cutoff crossings, large vacuum and buffer
regrowth, reverse-dependency membership and same-call full-reference row
extraction over changing geometries. Native integration accepts `--graph-backend device` to exercise the
same option with a caller-provided small model/data fixture.


For diagnosis only, `pimd_graph_profile yes` with `pimd_graph device` enables
synchronized internal phase measurements exposed as
`pimd_reuse_device_graph_profile`. The eight cumulative per-rank seconds are:
preparation/upload, spatial bins, neighbor count/fill, destination-row ordering,
node compaction/output allocation, edge arrays, source CSR/ordering, and guards.
This option adds synchronization and can perturb multi-process GPU scheduling.
It defaults to `no`; never mix its times with ordinary production benchmark
results or add them to the outer nested arrays. Match counts and force-call
scope before deriving per-call values.

Restart inputs must preserve restored physical and auxiliary state. In
particular, do not repeat a fresh-run `velocity all set ...` command after
`read_restart`. The small restart checker removes its fixture's initial
velocity command, as well as its initial bead displacement, only in the resumed
phase; it retains the integrator's validation of external state changes.
