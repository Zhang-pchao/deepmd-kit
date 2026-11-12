#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check actionable errors for invalid optional PIMD graph keywords.

Uses an existing compatible plugin and driver in a caller-managed GPU allocation.
No model is evaluated and no particle dynamics is run.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from check_pimd_reuse import quoted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "plugin", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mpi-command", default="mpirun")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        (
            "unknown",
            "pimd_reuse pi fallback pimd_graph invalid",
            "pimd_graph must be host or device",
        ),
        (
            "missing",
            "pimd_reuse pi fallback pimd_graph",
            "expects one pimd_graph host|device",
        ),
        (
            "duplicate",
            "pimd_reuse pi fallback pimd_graph host pimd_graph device",
            "expects one pimd_graph host|device",
        ),
        ("no_reuse", "pimd_graph device", "pimd_graph requires pimd_reuse FIX MODE"),
        (
            "mode",
            "pimd_reuse pi invalid pimd_graph device",
            "Unknown deepmd/kk pimd_reuse mode",
        ),
    ]
    cases.extend(
        [
            (
                "profile_value",
                "pimd_reuse pi fallback pimd_graph device pimd_graph_profile invalid",
                "pimd_graph_profile must be yes or no",
            ),
            (
                "profile_requires_device",
                "pimd_reuse pi fallback pimd_graph host pimd_graph_profile yes",
                "pimd_graph_profile yes requires pimd_graph device and pimd_reuse",
            ),
        ]
    )
    results = []
    for name, options, expected in cases:
        work = args.output / name
        work.mkdir()
        (work / "in.lammps").write_text(
            f"plugin load {quoted(args.plugin)}\nunits metal\natom_style atomic\n"
            f"pair_style deepmd/kk {quoted(args.model)} {options}\n"
        )
        command = [
            *shlex.split(args.mpi_command),
            "-np",
            "1",
            str(args.driver),
            "-k",
            "on",
            "g",
            "1",
            "-sf",
            "kk",
            "-in",
            "in.lammps",
            "-nocite",
        ]
        with (work / "stdout.log").open("w") as stream:
            run = subprocess.run(
                command, cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        output = (work / "stdout.log").read_text()
        passed = run.returncode != 0 and expected in output
        results.append(
            {
                "case": name,
                "command": command,
                "exit_code": run.returncode,
                "expected": expected,
                "passed": passed,
            }
        )
        (args.output / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n")
        if not passed:
            raise AssertionError(f"{name}: expected diagnostic not observed")


if __name__ == "__main__":
    main()
