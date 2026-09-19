"""Find, download and health-check the native ``inkvec`` binary.

Resolution order, first hit wins:

1. ``INKVEC_BIN``: path to an ``inkvec`` / ``inkvec.exe`` executable.
2. ``bin/`` next to this file: drop a binary in to pin a build.
3. A cached download: ``<ComfyUI>/models/inkvec/<version>/`` inside ComfyUI, otherwise a
   per-user cache directory. The highest cached version is used.
4. A download of the latest GitHub release of logolabs/inkvec for this platform, verified
   against the release's ``SHA256SUMS`` before it is installed into the cache.

``INKVEC_VERSION`` (for example ``0.1.3``) pins steps 3 and 4 to one release instead of the
latest. Every resolved binary is health-checked with ``inkvec --version`` once per process.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("inkvec-comfyui")

REPO = "logolabs/inkvec"
RELEASES_URL = f"https://github.com/{REPO}/releases"
_API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
_WEB_LATEST = f"{RELEASES_URL}/latest"

#: Oldest release whose flags this package passes. Every flag the node uses exists in 0.1.0.
MIN_VERSION = (0, 1, 0)
NET_TIMEOUT = 60
HEALTH_TIMEOUT = 30

EXE = "inkvec.exe" if sys.platform == "win32" else "inkvec"
_USER_AGENT = "inkvec-comfyui"

_lock = threading.Lock()
_resolved: tuple[Path, str] | None = None


class BinaryError(RuntimeError):
    """The inkvec binary could not be found, downloaded or run."""


def no_window_kwargs() -> dict:
    """subprocess kwargs that keep a console window from flashing up on Windows."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def platform_target() -> tuple[str, str]:
    """Rust target triple and archive extension of the release asset for this machine."""
    if sys.platform == "win32":
        # Only an x86_64 Windows build is published; Windows on Arm runs it under emulation.
        return "x86_64-pc-windows-msvc", ".zip"
    machine = platform.machine().lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine)
    if arch is None:
        raise BinaryError(
            f"no prebuilt inkvec for the {machine!r} architecture. Build it from source "
            f"(https://github.com/{REPO}) and set INKVEC_BIN to the executable."
        )
    if sys.platform == "darwin":
        return f"{arch}-apple-darwin", ".tar.gz"
    if sys.platform.startswith("linux"):
        return f"{arch}-unknown-linux-gnu", ".tar.gz"
    raise BinaryError(
        f"no prebuilt inkvec for {sys.platform}. Build it from source "
        f"(https://github.com/{REPO}) and set INKVEC_BIN to the executable."
    )


def asset_name(version: str, target: str, ext: str) -> str:
    """Release asset name, e.g. ``inkvec-0.1.3-x86_64-pc-windows-msvc.zip``."""
    return f"inkvec-{version}-{target}{ext}"


def cache_root() -> Path:
    """Where downloaded binaries live: ComfyUI's models dir, else a per-user cache."""
    try:
        import folder_paths  # type: ignore  # only importable inside ComfyUI

        return Path(folder_paths.models_dir) / "inkvec"
    except Exception:
        pass
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "inkvec-comfyui"


def parse_version(text: str) -> tuple[int, int, int] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(g) for g in m.groups()) if m else None  # type: ignore[return-value]


def _cached(version: str | None) -> Path | None:
    """The cached binary for `version`, or the highest cached version when None."""
    root = cache_root()
    if version:
        p = root / version / EXE
        return p if p.is_file() else None
    best = None
    if root.is_dir():
        for d in root.iterdir():
            v = parse_version(d.name)
            if v and d.name == ".".join(map(str, v)) and (d / EXE).is_file():
                if best is None or v > best[0]:
                    best = (v, d / EXE)
    return best[1] if best else None


def _open(url: str, method: str = "GET"):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": _USER_AGENT})
    return urllib.request.urlopen(req, timeout=NET_TIMEOUT)


def latest_version() -> str:
    """Version of the latest release, e.g. ``0.1.3``."""
    import json

    try:
        with _open(_API_LATEST) as resp:
            tag = json.loads(resp.read().decode("utf-8"))["tag_name"]
    except Exception as api_exc:  # rate limit (60/h unauthenticated), proxy, API outage
        try:
            # The web page redirects to .../releases/tag/<tag>; no API quota involved.
            with _open(_WEB_LATEST, method="HEAD") as resp:
                final = resp.geturl()
            m = re.search(r"/releases/tag/([^/?#]+)$", final)
            if not m:
                raise BinaryError(f"unexpected redirect target {final}")
            tag = m.group(1)
        except Exception as web_exc:
            raise BinaryError(
                f"could not look up the latest inkvec release ({api_exc}; {web_exc}). "
                f"Check the network, or download a binary from {RELEASES_URL} and set INKVEC_BIN."
            ) from web_exc
    return tag[1:] if tag.startswith("v") else tag


def _download(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    try:
        with _open(url) as resp, open(part, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_sha256(sums_text: str, name: str) -> str | None:
    for line in sums_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == name:
            return parts[0].lower()
    return None


def _extract(archive: Path, dest_dir: Path) -> Path:
    """Copy the executable (and LICENSE, NOTICE and options.schema.json, when the archive
    has them) out of a release archive.

    Members are matched by basename and written to fixed names, so no path inside the
    archive can place a file outside `dest_dir`.
    """
    wanted = {EXE, "LICENSE", "NOTICE", "options.schema.json"}
    found: dict[str, bytes] = {}
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                base = os.path.basename(info.filename)
                if base in wanted and not info.is_dir() and base not in found:
                    found[base] = zf.read(info)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                base = os.path.basename(member.name)
                if base in wanted and member.isfile() and base not in found:
                    fh = tf.extractfile(member)
                    if fh is not None:
                        found[base] = fh.read()
    if EXE not in found:
        raise BinaryError(f"{archive.name} contains no {EXE}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for base, data in found.items():
        tmp = dest_dir / f"{base}.part-{os.getpid()}"
        tmp.write_bytes(data)
        if base == EXE and sys.platform != "win32":
            tmp.chmod(0o755)
        os.replace(tmp, dest_dir / base)
    return dest_dir / EXE


def download(version: str | None = None) -> Path:
    """Download, verify and install one release into the cache; return the executable."""
    version = version or latest_version()
    target, ext = platform_target()
    name = asset_name(version, target, ext)
    base_url = f"{RELEASES_URL}/download/v{version}"
    log.info("[inkvec] downloading %s from %s", name, base_url)
    with tempfile.TemporaryDirectory(prefix="inkvec-dl-") as tmp:
        archive = Path(tmp) / name
        sums = Path(tmp) / "SHA256SUMS"
        try:
            _download(f"{base_url}/{name}", archive)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise BinaryError(
                    f"release v{version} has no asset {name} for this platform; "
                    f"see {RELEASES_URL} or set INKVEC_BIN to a binary built from source"
                ) from exc
            raise BinaryError(f"downloading {name} failed: {exc}") from exc
        except Exception as exc:
            raise BinaryError(f"downloading {name} failed: {exc}") from exc
        try:
            _download(f"{base_url}/SHA256SUMS", sums)
            expected = _expected_sha256(sums.read_text(encoding="utf-8"), name)
        except Exception as exc:
            raise BinaryError(
                f"release v{version} publishes no readable SHA256SUMS ({exc}); refusing to "
                f"install an unverified binary. Download it yourself and set INKVEC_BIN."
            ) from exc
        if expected is None:
            raise BinaryError(f"SHA256SUMS of v{version} does not list {name}; refusing to install it")
        actual = sha256_file(archive)
        if actual != expected:
            raise BinaryError(
                f"SHA256 mismatch for {name}: got {actual}, release lists {expected}; refusing to install it"
            )
        exe = _extract(archive, cache_root() / version)
    log.info("[inkvec] installed %s (sha256 %s verified)", exe, expected)
    return exe


def health_check(binary: Path) -> str:
    """Run ``inkvec --version``; return what it printed, or raise with the reason."""
    try:
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=HEALTH_TIMEOUT,
            stdin=subprocess.DEVNULL,
            **no_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise BinaryError(f"`{binary} --version` did not answer within {HEALTH_TIMEOUT} s") from exc
    except OSError as exc:
        raise BinaryError(
            f"cannot execute {binary}: {exc}. Set INKVEC_BIN to a working inkvec binary, or "
            f"delete {binary.parent} to force a fresh download."
        ) from exc
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        detail = ((proc.stderr or "") + (proc.stdout or "")).strip()
        raise BinaryError(f"`{binary} --version` exited with status {proc.returncode}: {detail}")
    version = parse_version(out)
    if not out.startswith("inkvec") or version is None:
        raise BinaryError(f"{binary} does not look like inkvec: --version printed {out!r}")
    if version < MIN_VERSION:
        raise BinaryError(
            f"{binary} is {out}; this node needs inkvec {'.'.join(map(str, MIN_VERSION))} or newer"
        )
    return out


def resolve(download_missing: bool = True) -> Path:
    """Locate a binary by the documented order, without health-checking it."""
    env = os.environ.get("INKVEC_BIN")
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise BinaryError(f"INKVEC_BIN is set to {env!r}, which is not a file")
        return p
    bundled = Path(__file__).resolve().parent / "bin" / EXE
    if bundled.is_file():
        return bundled
    pinned = os.environ.get("INKVEC_VERSION", "").strip().lstrip("v") or None
    cached = _cached(pinned)
    if cached is not None:
        return cached
    if not download_missing:
        raise BinaryError(f"no inkvec binary found and downloading is disabled; see {RELEASES_URL}")
    return download(pinned)


def ensure_binary() -> Path:
    """Resolved and health-checked binary path, cached for the life of the process."""
    return binary_info()[0]


def binary_info() -> tuple[Path, str]:
    """(path, ``--version`` output) of the binary the node will run."""
    global _resolved
    with _lock:
        if _resolved is None or not _resolved[0].is_file():
            path = resolve()
            _resolved = (path, health_check(path))
            log.info("[inkvec] using %s (%s)", path, _resolved[1])
        return _resolved


def reset() -> None:
    """Forget the resolved binary (tests, or after changing INKVEC_BIN at runtime)."""
    global _resolved
    with _lock:
        _resolved = None
