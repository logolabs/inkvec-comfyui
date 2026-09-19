"""Running the inkvec CLI and rendering its SVG. No ComfyUI imports here.

The CLI is invoked as ``inkvec [widget flags] [extra_args] -o <out.svg> <in.png>``. The
input path comes last on purpose: the CLI takes the last positional argument as the input,
so a stray word in ``extra_args`` cannot replace the image being traced.
"""

from __future__ import annotations

import io
import logging
import shlex
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import binary_manager, options

log = logging.getLogger("inkvec-comfyui")

#: Flags extra_args may not contain: the node owns the input/output paths and the process.
_RESERVED = {"-o", "--output", "-h", "--help", "-V", "--version"}


class InkvecError(RuntimeError):
    """The CLI failed; the message carries its own error output."""


class TraceInterrupted(RuntimeError):
    """The trace was cancelled from outside (ComfyUI's interrupt button)."""


def parse_extra_args(text: str) -> list[str]:
    """Split the free-text flags with shell quoting rules; refuse the ones the node owns."""
    if not text or not text.strip():
        return []
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as exc:
        raise ValueError(f"extra_args could not be parsed ({exc}): {text!r}") from exc
    bad = [t for t in tokens if t in _RESERVED or t.startswith("--output=")]
    if bad:
        raise ValueError(
            f"extra_args may not contain {', '.join(bad)}: the node sets the input and output itself"
        )
    return tokens


def build_args(values: dict | None = None, *, has_alpha: bool = False, extra_args: str = "") -> list[str]:
    """CLI flags for one trace: the option table's values (see options.py for the mapping),
    then extra_args, which come last and therefore override a widget's value."""
    return options.build_args(values or {}, has_alpha) + parse_extra_args(extra_args)


def _summarise_failure(stderr: str, stdout: str) -> str:
    """The CLI's own error lines. On a usage error it also prints its whole --help text,
    which would bury the one line that matters, so keep the `error:` lines when there are any."""
    text = (stderr or "").strip() or (stdout or "").strip()
    lines = text.splitlines()
    errors = [ln for ln in lines if ln.lower().startswith("error")]
    if errors:
        extra = " (run `inkvec --help` for the option list)" if len(lines) > len(errors) + 5 else ""
        return "\n".join(errors) + extra
    return "\n".join(lines[-30:]) or "(no output)"


def run_inkvec(
    png_path: Path,
    svg_path: Path,
    args: list[str],
    *,
    timeout_sec: float = 300,
    binary: Path | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[str, str]:
    """Trace one PNG. Returns (svg_text, the CLI's log). Raises InkvecError on failure,
    TraceInterrupted when `should_stop` turns true; the process is killed in both cases."""
    binary = Path(binary) if binary else binary_manager.ensure_binary()
    cmd = [str(binary), *args, "-o", str(svg_path), str(png_path)]
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **binary_manager.no_window_kwargs(),
    )
    while True:
        try:
            out, err = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            stop = should_stop is not None and should_stop()
            late = time.monotonic() - t0 > timeout_sec
            if stop or late:
                proc.kill()
                proc.communicate()
                if stop:
                    raise TraceInterrupted("inkvec was interrupted")
                raise InkvecError(
                    f"inkvec did not finish within {timeout_sec:g} s and was stopped. Raise "
                    f"timeout_sec, lower max_dim, or set time_budget so the tracer trades "
                    f"polish for time."
                )
    elapsed = time.monotonic() - t0
    if proc.returncode != 0 or not Path(svg_path).is_file():
        raise InkvecError(
            f"inkvec exited with status {proc.returncode}:\n{_summarise_failure(err, out)}\n"
            f"command: {shlex.join(cmd)}"
        )
    cli_log = "\n".join(s for s in ((err or "").strip(), (out or "").strip()) if s)
    for line in cli_log.splitlines():
        if line.lower().lstrip().startswith("warning"):
            log.warning("[inkvec] %s", line.strip())
    svg_text = Path(svg_path).read_text(encoding="utf-8")
    log.info("[inkvec] traced %s in %.2f s (%d bytes of SVG)", Path(png_path).name, elapsed, len(svg_text))
    log.debug("[inkvec] CLI log:\n%s", cli_log)
    return svg_text, cli_log


# --------------------------------------------------------------------------- rendering
_renderer_name: str | None = None


def renderer() -> str | None:
    """Name of the optional SVG rasteriser that is importable: resvg_py, cairosvg, or None."""
    global _renderer_name
    if _renderer_name is None:
        _renderer_name = ""
        try:
            import resvg_py  # noqa: F401

            _renderer_name = "resvg_py"
        except Exception:
            try:
                import cairosvg  # noqa: F401  (raises OSError when the cairo library is missing)

                _renderer_name = "cairosvg"
            except Exception:
                pass
    return _renderer_name or None


def _decode_png(data) -> np.ndarray:
    from PIL import Image

    if isinstance(data, str):  # early resvg_py releases returned base64 text
        import base64

        data = base64.b64decode(data)
    elif isinstance(data, list):  # ... or a list of byte values
        data = bytes(data)
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0


def render_svg(svg_text: str, width: int | None = None, height: int | None = None) -> np.ndarray | None:
    """Rasterise to straight-alpha RGBA float32 [H, W, 4] in [0, 1], at the SVG's own size
    unless width/height are given. None when no renderer is installed."""
    name = renderer()
    if name == "resvg_py":
        import resvg_py

        kw = {"svg_string": svg_text}
        if width and height:
            kw.update(width=int(width), height=int(height))
        try:
            data = resvg_py.svg_to_bytes(skip_system_fonts=True, **kw)
        except TypeError:  # older resvg_py without skip_system_fonts
            data = resvg_py.svg_to_bytes(**kw)
        return _decode_png(data)
    if name == "cairosvg":
        import cairosvg

        kw = {"bytestring": svg_text.encode("utf-8")}
        if width and height:
            kw.update(output_width=int(width), output_height=int(height))
        return _decode_png(cairosvg.svg2png(**kw))
    return None
