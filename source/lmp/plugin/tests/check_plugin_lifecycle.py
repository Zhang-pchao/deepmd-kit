#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check native plugin registration, reload and exit without a model."""

import argparse
import json
import os
import sys
from pathlib import Path
import re
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--kokkos", action="store_true", help="Check CUDA Kokkos aliases"
    )
    args = parser.parse_args()
    executable = args.lammps.resolve(strict=True)
    plugin = args.plugin.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if any(c in str(plugin) for c in ('"', "\n", "\r")):
        parser.error("The plugin path cannot contain quotes or newlines")
    env = os.environ.copy()
    env.pop("LAMMPS_PLUGIN_PATH", None)
    names = [
        "deepmd",
        "deepspin",
        "deeptensor/atom",
        "deepmd/fparam/dedn",
        "dplr",
        "pppm/dplr",
    ]
    aliases = ["deepmd/kk", "deepmd/kk/device", "deepmd/kk/host"]
    expected = names + aliases if args.kokkos else names
    results = []
    for gpu in [False, True] if args.kokkos else [False]:
        for reload in (False, True):
            name = ("cuda" if gpu else "plain") + ("-reload" if reload else "-load")
            commands = [f'plugin load "{plugin}"']
            if reload:
                commands += [
                    "plugin clear",
                    "plugin list",
                    'print "CLEARED"',
                    f'plugin load "{plugin}"',
                ]
            commands += ['print "REGISTERED"', "plugin list", 'print "COMPLETE"']
            text = "\n".join(commands) + "\n"
            (output / f"{name}.in").write_text(text)
            command = [str(executable)]
            if gpu:
                command += ["-k", "on", "g", "1", "-sf", "kk"]
            command += [
                "-echo",
                "none",
                "-log",
                "none",
                "-in",
                str(output / f"{name}.in"),
            ]
            with tempfile.TemporaryDirectory(prefix="deepmd-plugin-") as cwd:
                try:
                    proc = subprocess.run(
                        command,
                        cwd=cwd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=180,
                        check=False,
                    )
                    log = proc.stdout.decode(errors="replace")
                    code, timed_out = proc.returncode, False
                except subprocess.TimeoutExpired as exc:
                    log = (exc.stdout or b"").decode(errors="replace")
                    code, timed_out = None, True
            (output / f"{name}.log").write_text(log)
            listed = log.split("REGISTERED\n")[-1].split("COMPLETE\n")[0]
            found = set(
                re.findall(r"(?:pair|compute|fix|kspace) style plugin (\S+)", listed)
            )
            missing = sorted(set(expected) - found)
            unexpected = sorted(set(aliases) & found) if not args.kokkos else []
            cleared = (
                not reload or "Currently loaded plugins: 0" in log.split("CLEARED\n")[0]
            )
            passed = (
                code == 0
                and not timed_out
                and not missing
                and not unexpected
                and cleared
                and "COMPLETE\n" in log
            )
            results.append(
                {
                    "name": name,
                    "command": command,
                    "returncode": code,
                    "timeout": timed_out,
                    "missing": missing,
                    "unexpected": unexpected,
                    "cleared": cleared,
                    "passed": passed,
                }
            )
    passed = all(row["passed"] for row in results)
    (output / "RESULTS.json").write_text(
        json.dumps({"passed": passed, "cases": results}, indent=2) + "\n"
    )
    sys.stdout.write(json.dumps({"passed": passed, "cases": len(results)}) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
