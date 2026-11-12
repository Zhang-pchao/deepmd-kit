#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Small-system full/verify/fallback integration on a supplied one-hop model.

Compile pimd_reuse_driver.cpp against the matching LAMMPS C library first.
The caller supplies an admitted <=174-atom atomic data file and region geometry.
No model, private structure, or cluster layout is embedded in this test.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np


def quoted(path):
    text = str(path.resolve(strict=True))
    if any(c in text for c in ('"', "$", chr(10), chr(13))):
        raise ValueError("Unsafe LAMMPS path")
    return f'"{text}"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "plugin", "model", "data", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--type-map", nargs="+", required=True)
    parser.add_argument("--center", nargs=3, type=float, required=True)
    parser.add_argument("--empty-center", nargs=3, type=float, required=True)
    parser.add_argument("--all-center", nargs=3, type=float, required=True)
    parser.add_argument("--core", type=float, required=True)
    parser.add_argument("--outer", type=float, required=True)
    parser.add_argument("--all-core", type=float, required=True)
    parser.add_argument("--all-outer", type=float, required=True)
    parser.add_argument("--mpi-command", default="mpirun")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--beads", type=int, choices=(2, 4, 8, 16), default=4)
    parser.add_argument("--timestep", type=float, default=1e-6)
    parser.add_argument("--only-case", action="append", default=[])
    parser.add_argument("--plumed-input", type=Path)
    parser.add_argument(
        "--bias-mode",
        choices=("cv_mean", "coordinate_mean", "bias_mean"),
        default="cv_mean",
    )
    parser.add_argument("--contraction", type=float, default=0.5)
    parser.add_argument("--check-plumed-time", action="store_true")
    parser.add_argument("--check-migration", action="store_true")
    parser.add_argument("--check-support-change", action="store_true")
    parser.add_argument("--initial-velocity", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--bead-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--graph-backend", choices=("host", "device"), default="host")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("At least one GPU is required")
    if not 0 <= args.contraction <= 1:
        parser.error("Contraction must lie in [0,1]")
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", s) for s in args.type_map):
        parser.error("Invalid type labels")
    atoms = re.search(r"^\s*(\d+)\s+atoms\s*$", args.data.read_text(), re.M)
    if not atoms or not 1 <= int(atoms[1]) <= 174:
        parser.error("This functional test requires 1..174 atoms")
    if not 1 <= args.steps <= 20 or not 0 < args.timestep <= 1e-5:
        parser.error("Use a short, small-timestep engineering check")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "runs": [], "comparisons": []}
    env = os.environ.copy()
    env.pop("LAMMPS_PLUGIN_PATH", None)
    regions = {
        "local": (args.center, args.core, args.outer),
        "empty": (args.empty_center, args.core, args.outer),
        "all": (args.all_center, args.all_core, args.all_outer),
    }
    # Splitting z allows an empty-rank case for a compact supplied geometry.
    cases = [
        (args.beads, 1, "local"),
        (args.beads, 1, "empty"),
        (args.beads, 1, "all"),
        (1, 1, "local"),
        (args.beads, 2, "local"),
    ]
    names = {f"p{b}-r{r}-{region}" for b, r, region in cases}
    if set(args.only_case) - names:
        parser.error("Unknown test case")
    for beads, ranks, region in cases:
        if args.only_case and f"p{beads}-r{ranks}-{region}" not in args.only_case:
            continue
        reference = None
        center, core, outer = regions[region]
        for mode in ("full", "verify", "fallback"):
            label = f"p{beads}-r{ranks}-{region}-{mode}"
            work = output / label
            work.mkdir()
            ownership_start = ownership_end = ""
            if args.check_migration or args.check_support_change:
                ownership_start = "write_dump all custom owners-start-${bead}.dump id proc f_pi[10] modify sort id format float %.17g"
                ownership_end = "write_dump all custom owners-end-${bead}.dump id proc f_pi[10] modify sort id format float %.17g"
            sort = "1 1.0" if args.check_migration else "0 0"
            neighbor_check = "no" if args.check_migration else "yes"
            shifts = ["0", "0", "0"]
            shifts["xyz".index(args.bead_axis)] = "${shift}"
            bias_command = ""
            if args.plumed_input:
                (work / "plumed.dat").write_text(args.plumed_input.read_text())
                bias_command = f"fix pl all plumed plumedfile plumed.dat outfile plumed.log path_integral {args.bias_mode} pimd_fix pi"
                if args.bias_mode == "cv_mean":
                    bias_command += f" path_contraction {args.contraction}"
            text = f"""variable bead world {" ".join(map(str, range(beads)))}
variable shift equal 0.01*(v_bead-0.5*({beads}-1))
plugin load {quoted(args.plugin)}
units metal
atom_style atomic
boundary p p p
atom_modify map array sort {sort}
processors 1 1 {ranks}
read_data {quoted(args.data)}
displace_atoms all move {" ".join(shifts)} units box
pair_style deepmd/kk {quoted(args.model)} pimd_reuse pi {mode} pimd_graph {args.graph_backend}
pair_coeff * * {" ".join(args.type_map)}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check {neighbor_check}
timestep {args.timestep}
velocity all set {" ".join(map(str, args.initial_velocity))} units box
fix pi all pimd/adaptive temp 300 center {" ".join(map(str, center))} core {core} outer {outer} initialize auxiliary
{bias_command}
thermo 1
thermo_style custom step pe f_pi[1] f_pi[2] f_pi[4] f_pi[5]
thermo_modify norm no
run 0
run 0
{ownership_start}
run {args.steps}
{ownership_end}
write_dump all custom endpoint-${{bead}}.dump id x y z vx vy vz fx fy fz f_pi[1] f_pi[2] f_pi[3] f_pi[4] f_pi[5] f_pi[6] f_pi[7] f_pi[8] f_pi[9] f_pi[10] modify sort id format float %.17g
print "$(pe:%.17g) $(f_pi[1]:%.17g) $(f_pi[2]:%.17g) $(f_pi[4]:%.17g) $(f_pi[5]:%.17g)" file energy-${{bead}}.txt screen no
"""
            (work / "in.lammps").write_text(text)
            command = [
                *shlex.split(args.mpi_command),
                "-np",
                str(beads * ranks),
                str(args.driver.resolve()),
                "-k",
                "on",
                "g",
                str(args.gpus),
                "-sf",
                "kk",
                "-partition",
                f"{beads}x{ranks}",
                "-in",
                "in.lammps",
                "-log",
                "log",
                "-screen",
                "screen",
                "-nocite",
            ]
            with (work / "stdout.log").open("w") as log:
                proc = subprocess.run(
                    command,
                    cwd=work,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            report["runs"].append(
                {"label": label, "command": command, "exit_code": proc.returncode}
            )
            (output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
            if proc.returncode:
                raise RuntimeError(f"{label} exited {proc.returncode}")
            stats = re.findall(
                r"^PIMD_REUSE_STATS (.+)$", (work / "stdout.log").read_text(), re.M
            )
            if len(stats) != beads * ranks:
                raise AssertionError("Missing per-rank provider counters")
            for line in stats:
                values = np.array(list(map(float, line.split()[1:])))
                if values.size != 12 or not np.isfinite(values).all():
                    raise AssertionError("Invalid provider counters")
                if mode != "full":
                    if (
                        values[0] < args.steps + 3
                        or values[0] != values[1] + values[2]
                        or values[3] != values[1]
                    ):
                        raise AssertionError("Missing fresh-force evaluations")
                    if region == "local" and not 0 < values[6] < int(atoms[1]):
                        raise AssertionError(
                            "Local test must exercise a strict owner subset"
                        )
                    if region == "empty" and values[6] != 0:
                        raise AssertionError("Expected empty support")
                if mode == "verify" and (values[1] == 0 or values[9] != values[1]):
                    raise AssertionError("Verify did not execute split and full")
                if mode == "fallback":
                    expected = 2 if region == "all" else 1
                    if values[expected] == 0:
                        raise AssertionError(
                            "Expected execution path was not exercised"
                        )
            calls = re.findall(
                r"^PIMD_REUSE_CALLS (.+)$", (work / "stdout.log").read_text(), re.M
            )
            if len(calls) != beads * ranks:
                raise AssertionError("Missing actual per-rank evaluation counters")
            actual = {}
            for line in calls:
                rank, *raw = line.split()
                values = np.array(list(map(float, raw)))
                rank = int(rank)
                if (
                    rank in actual
                    or not 0 <= rank < beads * ranks
                    or values.size != 8
                    or not np.isfinite(values).all()
                    or np.any(values < 0)
                    or np.any(values != np.floor(values))
                ):
                    raise AssertionError("Invalid actual evaluation counters")
                actual[rank] = values
            for line in stats:
                rank, *raw = line.split()
                rank = int(rank)
                legacy = np.array(list(map(float, raw)))
                split, fallback = legacy[1], legacy[2]
                values = actual[rank]
                sequence = values[0]
                if sequence < args.steps + 3:
                    raise AssertionError("Missing unique force-call sequence")
                expected = np.zeros(8)
                expected[0] = sequence
                if mode == "full":
                    expected[1] = sequence
                else:
                    if sequence != split + fallback:
                        raise AssertionError("Force-call sequence does not balance")
                    expected[1] = expected[6] = fallback
                    if rank == 0:
                        expected[2] = split
                        expected[3] = legacy[4]
                        expected[7] += 2 * split
                    if rank % ranks == 0:
                        expected[4] = legacy[5]
                        expected[5] = legacy[9]
                        expected[7] += split + legacy[9]
                if not np.array_equal(values, expected):
                    raise AssertionError(
                        f"Actual calls mismatch rank {rank}: {values} != {expected}"
                    )
                if (
                    mode == "fallback"
                    and region == "local"
                    and (values[1] or values[5] or values[6])
                ):
                    raise AssertionError(
                        "Pure reuse includes full/verify/fallback work"
                    )
            profiles = re.findall(
                r"^PIMD_REUSE_PROFILE (.+)$", (work / "stdout.log").read_text(), re.M
            )
            if len(profiles) != 2 * beads * ranks:
                raise AssertionError("Missing per-rank nested timing arrays")
            seen = set()
            for line in profiles:
                rank, name, *raw = line.split()
                expected = {"pimd_reuse_profile": 5, "pimd_reuse_model_profile": 4}
                values = np.array(list(map(float, raw)))
                key = (int(rank), name)
                if key in seen or not 0 <= int(rank) < beads * ranks:
                    raise AssertionError("Duplicate or invalid timing rank")
                seen.add(key)
                if (
                    name not in expected
                    or values.size != expected[name]
                    or not np.isfinite(values).all()
                    or (values < 0).any()
                ):
                    raise AssertionError("Invalid nested timing array")
                if mode == "full" and values.any():
                    raise AssertionError(
                        "Full mode unexpectedly entered reuse profiling"
                    )
            tables = [
                np.loadtxt(work / f"endpoint-{b}.dump", skiprows=9, ndmin=2)
                for b in range(beads)
            ]
            energies = np.array(
                [np.loadtxt(work / f"energy-{b}.txt") for b in range(beads)]
            )
            if reference is None:
                reference = (tables, energies)
            for b, table in enumerate(tables):
                if not np.array_equal(table[:, 0], reference[0][b][:, 0]):
                    raise AssertionError("Atom ordering differs")
                # x, v, F, auxiliary x, auxiliary v, auxiliary force, amplitude.
                bounds = np.array(
                    [2e-6] * 3 + [5e-4] * 6 + [2e-6] * 3 + [5e-4] * 6 + [1e-6]
                )
                error = np.abs(table[:, 1:] - reference[0][b][:, 1:])
                passed = bool(np.isfinite(table).all() and (error <= bounds).all())
                report["comparisons"].append(
                    {
                        "label": label,
                        "bead": b,
                        "passed": passed,
                        "max_bound_fraction": float((error / bounds).max()),
                    }
                )
                if not passed:
                    raise AssertionError(f"{label}: endpoint parity")
            if args.check_plumed_time:
                files = list(work.glob("COLVAR*"))
                if not files:
                    raise AssertionError("Missing PLUMED time diagnostics")
                for path in files:
                    columns = np.loadtxt(path, comments="#", ndmin=2)
                    if abs(columns[-1, 0] - args.steps * args.timestep) > 1e-12:
                        raise AssertionError(
                            "PLUMED and LAMMPS times differ; use TIME=ps and PRINT STRIDE=1"
                        )
            if args.check_migration or args.check_support_change:
                for bead in range(beads):
                    start = np.loadtxt(
                        work / f"owners-start-{bead}.dump", skiprows=9, ndmin=2
                    )
                    end = np.loadtxt(
                        work / f"owners-end-{bead}.dump", skiprows=9, ndmin=2
                    )
                    if not np.array_equal(start[:, 0], end[:, 0]):
                        raise AssertionError("Ownership diagnostic atom IDs changed")
                    if args.check_migration and not (start[:, 1] != end[:, 1]).any():
                        raise AssertionError(
                            "No actual MPI ownership migration occurred"
                        )
                    if (
                        args.check_support_change
                        and not ((start[:, 2] > 0) != (end[:, 2] > 0)).any()
                    ):
                        raise AssertionError("No actual path support crossing occurred")
            n = tables[0].shape[0]
            if not (
                np.isfinite(energies).all()
                and (
                    np.abs(energies - reference[1]) / n
                    <= 1e-5 + 2e-7 * np.abs(reference[1] / n)
                ).all()
            ):
                raise AssertionError(f"{label}: energy parity")
            (output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    report["passed"] = True
    (output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.stdout.write("PIMD_REUSE_SMALL_INTEGRATION_PASS" + chr(10))


if __name__ == "__main__":
    main()
