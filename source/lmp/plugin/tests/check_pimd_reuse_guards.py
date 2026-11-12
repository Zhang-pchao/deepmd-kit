#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model-free MPI rejection checks for the optional PIMD reuse interface."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from check_pimd_reuse import quoted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "plugin", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--mpi-command", default="mpirun")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for label in (
        "different-bytes",
        "different-length",
        "late-byte",
        "modes",
        "graph-modes",
        "missing-file",
    ):
        work = output / label
        work.mkdir()
        for bead in range(4):
            if label == "missing-file" and bead == 2:
                continue
            data = b"a" * 65536
            if bead == 2:
                if label == "different-bytes":
                    data = b"b" + data[1:]
                if label == "different-length":
                    data += b"a"
                if label == "late-byte":
                    data += b"b"
            elif label == "late-byte":
                data += b"a"
            (work / f"model-{bead}.pt2").write_bytes(data)
        modes = (
            "verify fallback verify fallback"
            if label == "modes"
            else "verify verify verify verify"
        )
        graphs = (
            "host device host device"
            if label == "graph-modes"
            else "host host host host"
        )
        text = f"""variable bead world 0 1 2 3
variable mode world {modes}
variable graph world {graphs}
plugin load {quoted(args.plugin)}
units metal
atom_style atomic
pair_style deepmd/kk model-${{bead}}.pt2 pimd_reuse pi ${{mode}} pimd_graph ${{graph}}
"""
        (work / "in.lammps").write_text(text)
        cmd = [
            *shlex.split(args.mpi_command),
            "-np",
            "4",
            str(args.driver.resolve()),
            "-k",
            "on",
            "g",
            "1",
            "-sf",
            "kk",
            "-partition",
            "4x1",
            "-in",
            "in.lammps",
            "-log",
            "none",
            "-screen",
            "screen",
            "-nocite",
        ]
        # This is an expected-rejection test: a finite hang guard catches a
        # collective mismatch, not a scientific workload walltime.
        with (work / "stdout.log").open("w") as log:
            proc = subprocess.run(
                cmd,
                cwd=work,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=180,
            )
        message = "byte-identical model archives"
        if label in ("modes", "graph-modes"):
            message = "identical modes"
        if label == "missing-file":
            message = "cannot read a model archive"
        logs = (work / "stdout.log").read_text() + "".join(
            p.read_text() for p in work.glob("screen*")
        )
        passed = proc.returncode != 0 and message in logs
        results.append(
            {
                "case": label,
                "exit_code": proc.returncode,
                "passed": passed,
                "expected_message": message,
                "command": cmd,
            }
        )
        (output / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n")
        if not passed:
            raise AssertionError(
                f"{label}: expected collective rejection was not observed"
            )
    sys.stdout.write("PIMD_REUSE_GUARDS_PASS\n")


if __name__ == "__main__":
    main()
