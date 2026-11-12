#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Replay a qualified small-system input across a native restart boundary.

Compare matched segmented execution (including repeated same-step force calls)
with a restart at the same boundary. Restore complete OPES STATE when present.
This does not claim bitwise stochastic continuity or monolithic-run identity.
"""

import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np

from check_pimd_reuse import quoted


def last_state(path):
    blocks = path.read_text().split("#! FIELDS ")
    text = "#! FIELDS " + blocks[-1]
    fields = text.splitlines()[0]
    metadata, rows = {}, []
    for line in text.splitlines()[1:]:
        if line.startswith("#! SET "):
            key, value = line[7:].split(maxsplit=1)
            try:
                value = float(value)
            except ValueError:
                pass
            metadata[key] = value
        elif line and not line.startswith("#"):
            rows.append(list(map(float, line.split())))
    return fields, metadata, np.array(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "driver", "plugin", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--beads", type=int, default=4)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--split", type=int, default=2)
    parser.add_argument("--mpi-command", default="mpirun")
    args = parser.parse_args()
    source = (args.template / "in.lammps").read_text()
    steps = re.findall(r"^run ([1-9][0-9]*)$", source, re.M)
    if len(steps) != 1 or not 0 < args.split < int(steps[0]) <= 20:
        parser.error("Template must contain one small positive run and a valid split")
    total = int(steps[0])
    source = re.sub(
        r"^plugin load .+$", "plugin load " + quoted(args.plugin), source, flags=re.M
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "runs": [],
        "endpoint_comparisons": [],
        "state_comparisons": [],
    }
    for phase in ["continuous", "checkpoint", "restarted"]:
        work = output / phase
        work.mkdir()
        text = source
        if phase == "continuous":
            replacement = f"run {args.split}\nrun 0\nrun 0\nrun {total - args.split}"
        elif phase == "checkpoint":
            replacement = (
                f"run {args.split}\nwrite_restart checkpoint-${{bead}}.restart"
            )
        else:
            text = re.sub(
                r"^read_data .+$",
                "read_restart checkpoint-${bead}.restart",
                text,
                flags=re.M,
            )
            text = re.sub(r"^displace_atoms .+\n", "", text, flags=re.M)
            # The fixture's initial velocity belongs only to a fresh run.
            # read_restart already restores physical and auxiliary velocities;
            # resetting one side would violate the integrator state contract.
            text = re.sub(r"^velocity all set .+\n", "", text, flags=re.M)

            replacement = f"run {total - args.split}"
            for path in (output / "checkpoint").glob("checkpoint-*.restart"):
                shutil.copy2(path, work / path.name)
            for path in (output / "checkpoint").glob("STATE*"):
                shutil.copy2(path, work / path.name)
        text = text.replace("run " + str(total) + "\n", replacement + "\n")
        (work / "in.lammps").write_text(text)
        if (args.template / "plumed.dat").exists():
            plumed = (args.template / "plumed.dat").read_text()
            if phase == "restarted" and "OPES_METAD" in plumed:
                plumed = plumed.replace("RESTART=NO", "RESTART=YES")
                plumed = plumed.replace(
                    "STATE_WFILE=STATE", "STATE_RFILE=STATE STATE_WFILE=RESUMED_STATE"
                )
                plumed = "RESTART\n" + plumed
            (work / "plumed.dat").write_text(plumed)
        command = [
            *shlex.split(args.mpi_command),
            "-np",
            str(args.beads * args.ranks),
            str(args.driver.resolve()),
            "-k",
            "on",
            "g",
            "1",
            "-sf",
            "kk",
            "-partition",
            f"{args.beads}x{args.ranks}",
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
                command, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        report["runs"].append(
            {"phase": phase, "command": command, "exit_code": proc.returncode}
        )
        (output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
        if proc.returncode:
            raise RuntimeError(f"{phase} exited {proc.returncode}")
    for bead in range(args.beads):
        expected = np.loadtxt(
            output / f"continuous/endpoint-{bead}.dump", skiprows=9, ndmin=2
        )
        actual = np.loadtxt(
            output / f"restarted/endpoint-{bead}.dump", skiprows=9, ndmin=2
        )
        if not np.array_equal(expected[:, 0], actual[:, 0]):
            raise AssertionError("Restart atom IDs differ")
        bounds = np.array([2e-6] * 3 + [5e-4] * 6 + [2e-6] * 3 + [5e-4] * 6 + [1e-6])
        delta = np.abs(actual[:, 1:] - expected[:, 1:])
        passed = bool(np.isfinite(actual).all() and (delta <= bounds).all())
        report["endpoint_comparisons"].append(
            {
                "bead": bead,
                "passed": passed,
                "max_bound_fraction": float((delta / bounds).max()),
            }
        )
        if not passed:
            raise AssertionError("Restart endpoint parity failed")
        ref = np.loadtxt(output / f"continuous/energy-{bead}.txt")
        val = np.loadtxt(output / f"restarted/energy-{bead}.txt")
        n = len(expected)
        if not (
            np.isfinite(val).all()
            and (np.abs(val - ref) / n <= 1e-5 + 2e-7 * np.abs(ref / n)).all()
        ):
            raise AssertionError("Restart energy parity failed")
    for state in (output / "continuous").glob("STATE*"):
        resumed = output / "restarted" / ("RESUMED_" + state.name)
        if not resumed.exists():
            raise AssertionError("Complete resumed OPES STATE is missing")
        f1, m1, r1 = last_state(state)
        f2, m2, r2 = last_state(resumed)
        if f1 != f2 or m1.keys() != m2.keys():
            raise AssertionError("OPES state schema differs")
        for key, val in m1.items():
            if isinstance(val, str):
                if val != m2[key]:
                    raise AssertionError("OPES metadata differs")
            else:
                np.testing.assert_allclose(m2[key], val, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r2, r1, rtol=1e-12, atol=1e-12)
        report["state_comparisons"].append({"file": state.name, "passed": True})
    report["passed"] = True
    (output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.stdout.write("PIMD_REUSE_RESTART_PASS\n")


if __name__ == "__main__":
    main()
