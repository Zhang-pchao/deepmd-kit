#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare one CUDA instance with MPI ranks and partitions on two GPUs.

Use an admitted graph-input model and atomic LAMMPS data in metal units.
Run inside an allocation with at least four MPI ranks and two visible GPUs.
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


def quoted(path: Path) -> str:
    text = str(path.resolve(strict=True))
    if any(c in text for c in ('"', "$", "\n", "\r")):
        raise ValueError("LAMMPS paths cannot contain quotes, dollars or newlines")
    return f'"{text}"'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lammps", "plugin", "model", "data", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--type-map", nargs="+", required=True)
    parser.add_argument("--mpi-command", default="mpirun")
    parser.add_argument("--force-atol", type=float, default=2e-4)
    parser.add_argument("--force-rtol", type=float, default=2e-5)
    parser.add_argument("--energy-atol", type=float, default=1e-5)
    parser.add_argument("--energy-rtol", type=float, default=2e-7)
    parser.add_argument("--virial-atol", type=float, default=1e-4)
    parser.add_argument("--virial-rtol", type=float, default=2e-5)
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", item) for item in args.type_map):
        parser.error("Type names must be alphanumeric element labels")
    for kind in ("force", "energy", "virial"):
        if getattr(args, f"{kind}_atol") <= 0 or getattr(args, f"{kind}_rtol") < 0:
            parser.error(
                "Absolute tolerances must be positive and relative tolerances nonnegative"
            )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("LAMMPS_PLUGIN_PATH", None)
    report = {"passed": False, "runs": [], "checks": []}
    reference = None
    # The 4x1 layout exposes a partition-local rank of zero on both CUDA devices.
    for label, partitions, ranks, gpus in (
        ("reference", 1, 1, 1),
        ("ranks", 1, 4, 2),
        ("partitions", 4, 1, 2),
        ("split", 2, 2, 2),
    ):
        work = output / label
        work.mkdir()
        script = f"""variable bead world {" ".join(map(str, range(partitions)))}
plugin load {quoted(args.plugin)}
units metal
atom_style atomic
boundary p p p
atom_modify map array sort 0 0
read_data {quoted(args.data)}
pair_style deepmd {quoted(args.model)}
pair_coeff * * {" ".join(args.type_map)}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check no
compute pv all pressure NULL pair
thermo_style custom step pe c_pv[1] c_pv[2] c_pv[3]
thermo_modify norm no
run 0
write_dump all custom forces-${{bead}}.dump id fx fy fz modify sort id format float %.17g
print "$(pe:%.17g) $(c_pv[1]*vol/1.6021765e6:%.17g) $(c_pv[2]*vol/1.6021765e6:%.17g) $(c_pv[3]*vol/1.6021765e6:%.17g) $(c_pv[4]*vol/1.6021765e6:%.17g) $(c_pv[5]*vol/1.6021765e6:%.17g) $(c_pv[6]*vol/1.6021765e6:%.17g)" file values-${{bead}}.txt screen no
"""
        (work / "in.lammps").write_text(script)
        command = [
            *shlex.split(args.mpi_command),
            "-np",
            str(partitions * ranks),
            str(args.lammps.resolve(strict=True)),
            "-k",
            "on",
            "g",
            str(gpus),
            "-sf",
            "kk",
            "-partition",
            f"{partitions}x{ranks}",
            "-in",
            "in.lammps",
            "-log",
            "log",
            "-screen",
            "screen",
        ]
        with (work / "stdout.log").open("w") as log:
            proc = subprocess.run(
                command,
                cwd=work,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        report["runs"].append(
            {"name": label, "command": command, "exit_code": proc.returncode}
        )
        (output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
        if proc.returncode:
            raise SystemExit(proc.returncode)
        for bead in range(partitions):
            force_table = np.loadtxt(work / f"forces-{bead}.dump", skiprows=9, ndmin=2)
            n = len(force_table)
            if not np.array_equal(force_table[:, 0], np.arange(1, n + 1)):
                raise AssertionError("Atom IDs must be contiguous and sorted")
            values = np.loadtxt(work / f"values-{bead}.txt")
            current = {
                "energy": values[0] / n,
                "force": force_table[:, 1:],
                "virial": values[1:] / n,
            }
            if reference is None:
                reference = current
            for kind in current:
                error = np.abs(current[kind] - reference[kind])
                bound = getattr(args, f"{kind}_atol") + getattr(
                    args, f"{kind}_rtol"
                ) * np.abs(reference[kind])
                passed = bool(
                    np.isfinite(current[kind]).all() and (error <= bound).all()
                )
                report["checks"].append(
                    {
                        "name": f"{label}-{bead}-{kind}",
                        "passed": passed,
                        "max_error": float(error.max()),
                        "max_bound_fraction": float((error / bound).max()),
                    }
                )
                (output / "RESULTS.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                if not passed:
                    raise AssertionError(f"{label}-{bead}-{kind}")
    report["passed"] = True
    (output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {
                "passed": True,
                "runs": len(report["runs"]),
                "checks": len(report["checks"]),
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
