# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build-option regressions that do not require either neural-network backend."""

from pathlib import (
    Path,
)

import pytest


@pytest.mark.parametrize("interface", ["python", "ipi", "lammps"])
@pytest.mark.parametrize("tensorflow", ["0", "1", None])
def test_cpp_interface_respects_tensorflow_selection(
    monkeypatch, interface, tensorflow
):
    # The build backend belongs to the source checkout, not the installed wheel.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from backend import (
        read_env,
    )

    for key in (
        "DP_VARIANT",
        "DP_ENABLE_IPI",
        "DP_LAMMPS_VERSION",
        "DP_ENABLE_TENSORFLOW",
        "DP_ENABLE_PYTORCH",
        "DP_BUILD_TESTING",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DP_ENABLE_PYTORCH", "1")
    if tensorflow is not None:
        monkeypatch.setenv("DP_ENABLE_TENSORFLOW", tensorflow)
    if interface == "ipi":
        monkeypatch.setenv("DP_ENABLE_IPI", "1")
    elif interface == "lammps":
        monkeypatch.setenv("DP_LAMMPS_VERSION", "test-version")
    tf_enabled = tensorflow != "0"

    def find_tensorflow():
        assert tf_enabled, "Disabled TensorFlow must not be discovered"
        return "/mock/tensorflow", []

    monkeypatch.setattr(read_env, "find_tensorflow", find_tensorflow)
    monkeypatch.setattr(read_env, "get_tf_version", lambda path: "2.20.0")
    monkeypatch.setattr(read_env, "find_pytorch", lambda: ("/mock/torch", []))
    monkeypatch.setattr(read_env, "get_pt_version", lambda path: "2.11.0")
    read_env.get_argument_from_env.cache_clear()
    try:
        _, args, _, scripts, tf_version, pt_version = read_env.get_argument_from_env()
    finally:
        read_env.get_argument_from_env.cache_clear()
    assert (
        "-DENABLE_TENSORFLOW=ON" if tf_enabled else "-DENABLE_TENSORFLOW=OFF"
    ) in args
    if tf_enabled:
        assert "-DTENSORFLOW_VERSION=2.20.0" in args
        assert "-DTENSORFLOW_ROOT:PATH=/mock/tensorflow" in args
        assert tf_version == "2.20.0"
    else:
        assert not any(arg.startswith("-DTENSORFLOW_VERSION=") for arg in args)
        assert not any(arg.startswith("-DTENSORFLOW_ROOT:PATH=") for arg in args)
        assert tf_version is None
    assert "-DENABLE_PYTORCH=ON" in args
    assert "-DCMAKE_PREFIX_PATH=/mock/torch" in args
    assert pt_version == "2.11.0"
    if interface != "python":
        selected = "TRUE" if tf_enabled else "FALSE"
        assert f"-DUSE_TF_PYTHON_LIBS:BOOL={selected}" in args
        assert "-DBUILD_CPP_IF:BOOL=TRUE" in args
    else:
        assert "-DBUILD_CPP_IF:BOOL=FALSE" in args
        assert "-DUSE_TF_PYTHON_LIBS:BOOL=TRUE" not in args
    assert ("dp_ipi" in scripts) == (interface == "ipi")


def test_project_milestone_tags_preserve_package_version(tmp_path):
    import subprocess
    from setuptools_scm import get_version

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())["tool"][
        "setuptools_scm"
    ]
    repository = tmp_path / "repository"
    repository.mkdir()
    for arguments in (
        ["init"],
        ["config", "user.name", "Version test"],
        ["config", "user.email", "version@example.invalid"],
        ["commit", "--allow-empty", "-m", "Initial version"],
        ["tag", "v3.2.0"],
        ["commit", "--allow-empty", "-m", "Project integration"],
        ["tag", "plm-pimd-api31-v1"],
    ):
        subprocess.run(
            ["git", "-C", str(repository), *arguments], check=True, capture_output=True
        )
    version = get_version(
        root=repository, git_describe_command=config["git_describe_command"]
    )
    assert version.startswith("3.2.1.dev1+")
