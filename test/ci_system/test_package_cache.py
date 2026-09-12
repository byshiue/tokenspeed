import os
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("package_cache.sh")


def run_bash(command: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; {command}'],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_other_clusters_do_not_enable_package_cache(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "CI_RUNNER_LABEL": "gb200-4gpu",
            "FLASHINFER_CACHE_DIR": str(tmp_path / "flashinfer"),
        }
    )
    env.pop("PIP_CACHE_DIR", None)
    env.pop("CI_WHEEL_CACHE_DIR", None)
    result = run_bash(
        'configure_package_cache; printf "%s|%s" "${PIP_CACHE_DIR:-}" "${CI_WHEEL_CACHE_DIR:-}"',
        env,
    )
    assert result.stdout == "|"


def test_b200v2_uses_persistent_cache_next_to_flashinfer(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "CI_RUNNER_LABEL": "b200v2-8gpu",
            "FLASHINFER_CACHE_DIR": str(tmp_path / "flashinfer"),
        }
    )
    env.pop("PIP_CACHE_DIR", None)
    env.pop("CI_WHEEL_CACHE_DIR", None)
    result = run_bash(
        'configure_package_cache >/dev/null; printf "%s|%s" "${PIP_CACHE_DIR}" "${CI_WHEEL_CACHE_DIR}"',
        env,
    )
    assert result.stdout == f"{tmp_path / 'pip'}|{tmp_path / 'wheelhouse'}"
    assert (tmp_path / "pip").is_dir()
    assert (tmp_path / "wheelhouse").is_dir()


def test_slurm_uses_mounted_persistent_cache(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "CI_RUNNER_LABEL": "slurm-gb300-4gpu",
            "XDG_CACHE_HOME": str(tmp_path),
        }
    )
    env.pop("PIP_CACHE_DIR", None)
    env.pop("CI_WHEEL_CACHE_DIR", None)
    result = run_bash(
        'configure_package_cache >/dev/null; printf "%s|%s" "${PIP_CACHE_DIR}" "${CI_WHEEL_CACHE_DIR}"',
        env,
    )
    assert result.stdout == f"{tmp_path / 'pip'}|{tmp_path / 'wheelhouse'}"


def _b200_cache_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("CI_CACHE_ROOT", "PIP_CACHE_DIR", "CI_WHEEL_CACHE_DIR"):
        env.pop(key, None)
    env.update(
        CI_RUNNER_LABEL="b200v2-4gpu",
        FLASHINFER_CACHE_DIR=str(tmp_path / "shared" / "flashinfer"),
        XDG_CACHE_HOME=str(tmp_path / "user-cache"),
    )
    return env


def _configure_and_write_caches(env: dict[str, str]) -> list[Path]:
    result = run_bash(
        "set -e; configure_package_cache >/dev/null; "
        'printf test > "${PIP_CACHE_DIR}/test-entry"; '
        'printf test > "${CI_WHEEL_CACHE_DIR}/test-entry"; '
        'printf "%s|%s" "${PIP_CACHE_DIR}" "${CI_WHEEL_CACHE_DIR}"',
        env,
    )
    paths = [Path(path) for path in result.stdout.split("|")]
    for path in paths:
        assert (path / "test-entry").read_text() == "test"
        assert not list(path.glob(".tokenspeed-cache.*"))
    return paths


@pytest.mark.parametrize("blocked", ["root", "pip", "wheelhouse"])
def test_b200v2_falls_back_for_unusable_automatic_cache(tmp_path: Path, blocked: str):
    shared = tmp_path / "shared"
    if blocked == "root":
        shared.write_text("not a directory")
    else:
        shared.mkdir()
        (shared / blocked).write_text("not a directory")

    paths = _configure_and_write_caches(_b200_cache_env(tmp_path))

    assert paths == [
        (tmp_path / "user-cache" if blocked in ("root", kind) else shared) / kind
        for kind in ("pip", "wheelhouse")
    ]


@pytest.mark.parametrize(
    "setting", ["CI_CACHE_ROOT", "PIP_CACHE_DIR", "CI_WHEEL_CACHE_DIR"]
)
def test_b200v2_preserves_explicit_cache_paths(tmp_path: Path, setting: str):
    (tmp_path / "shared").write_text("unusable automatic root")
    env = _b200_cache_env(tmp_path)
    explicit = tmp_path / "explicit"
    env[setting] = str(explicit)

    paths = _configure_and_write_caches(env)

    expected = [tmp_path / "user-cache" / kind for kind in ("pip", "wheelhouse")]
    if setting == "CI_CACHE_ROOT":
        expected = [explicit / kind for kind in ("pip", "wheelhouse")]
    else:
        expected[0 if setting == "PIP_CACHE_DIR" else 1] = explicit
    assert paths == expected


@pytest.mark.parametrize(
    "setting",
    ["CI_CACHE_ROOT", "PIP_CACHE_DIR", "CI_WHEEL_CACHE_DIR", "XDG_CACHE_HOME"],
)
def test_b200v2_fails_for_unusable_explicit_or_fallback_cache(
    tmp_path: Path, setting: str
):
    env = _b200_cache_env(tmp_path)
    unusable = tmp_path / "unusable"
    unusable.write_text("not a directory")
    env[setting] = str(unusable)
    if setting == "XDG_CACHE_HOME":
        (tmp_path / "shared").write_text("unusable automatic root")

    with pytest.raises(subprocess.CalledProcessError) as exc:
        _configure_and_write_caches(env)

    assert str(unusable) in exc.value.stderr


def test_cached_remote_wheel_downloads_only_once(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    cache_dir = tmp_path / "wheelhouse"
    bin_dir.mkdir()
    cache_dir.mkdir()
    fake_curl = bin_dir / "curl"
    fake_curl.write_text("""#!/bin/bash
set -e
printf 'called\\n' >> "${CURL_CALLS}"
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--output" ]; then
        printf 'complete wheel' > "$2"
        exit 0
    fi
    shift
done
exit 1
""")
    fake_curl.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "CI_WHEEL_CACHE_DIR": str(cache_dir),
            "CURL_CALLS": str(tmp_path / "curl-calls"),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    (cache_dir / "pkg.whl").write_text("bad wheel")
    expected_sha256 = sha256(b"complete wheel").hexdigest()
    command = f'for i in 1 2 3 4; do cache_remote_wheel "https://example.test/pkg.whl" "{expected_sha256}" & done; wait'
    result = run_bash(command, env)
    assert result.stdout.splitlines() == [str(cache_dir / "pkg.whl")] * 4
    assert (tmp_path / "curl-calls").read_text().splitlines() == ["called"]
    assert (cache_dir / "pkg.whl").read_text() == "complete wheel"
    assert not list(cache_dir.glob("*.tmp.*"))
