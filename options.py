"""The trace node's options: one table, from which both the node's widgets and the CLI
arguments are built.

Where the table comes from, first hit wins:

1. ``INKVEC_OPTIONS_SCHEMA``: path to an ``options.schema.json``.
2. ``options.schema.json`` next to the inkvec binary. No current release ships one; the file
   is picked up automatically once a release does (the downloader already extracts it).
3. ``BUILTIN`` below, which mirrors ``inkvec --help`` of the 0.1.x releases.

The schema is JSON Schema: an object whose ``properties`` each carry ``type`` (``number``,
``integer``, ``boolean`` or ``string``), ``default``, ``description`` and, where they apply,
``minimum`` / ``maximum`` / ``multipleOf`` / ``enum``. How a property becomes a CLI argument:

* The flag is ``--`` plus the property name with ``_`` turned into ``-``
  (``min_area`` -> ``--min-area``), unless the property sets ``"x-cli-flag"``.
* ``number`` / ``integer`` / string ``enum``: always passed as ``--flag value``, so the value
  the widget shows is the value that runs.
* ``boolean``: a presence flag, passed only when the value differs from the default. A
  default-false option set true passes ``--flag``; a default-true option set false passes
  the negative form ``--no-<flag>`` (``harmonize`` -> ``--no-harmonize``).
* ``string`` without ``enum``: ``--flag value`` when the value is not empty.

Properties the node cannot use are skipped (``SKIP``: the input/output paths it owns, and
the neural pre-passes, which need model weights the node does not provide), as are
properties marked ``"deprecated": true`` or ``"x-experimental": true``. Options that exist in
both the schema and ``BUILTIN`` keep the built-in order, so saved workflows keep their
widget positions; options only the schema has are appended after them. Anything not in the
table can still be passed through the node's ``extra_args``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("inkvec-comfyui")

SCHEMA_FILENAME = "options.schema.json"
SCHEMA_ENV = "INKVEC_OPTIONS_SCHEMA"

#: Schema properties the node never shows: owned by the node (including the names of the
#: node's own inputs), or not usable from it.
SKIP = frozenset({
    "image", "mask", "timeout_sec", "extra_args",
    "input", "output", "quiet", "help", "version",
    "restore", "restore_threshold", "restore_weights", "restore_command",
    "sr", "sr_threshold", "sr_scale", "sr_no_recolour", "sr_command",
})

_INT_MAX = 2**31 - 1
_FLOAT_MAX = 1.0e6


@dataclass(frozen=True)
class Option:
    """One CLI option. ``kind`` is float, int, bool, enum or string."""

    name: str
    kind: str
    default: object
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple = ()
    flag: str = ""

    @property
    def cli_flag(self) -> str:
        return self.flag or "--" + self.name.replace("_", "-")

    def cli_args(self, value) -> list[str]:
        """The CLI arguments for `value`, by the mapping in the module docstring."""
        if self.kind == "bool":
            value, default = bool(value), bool(self.default)
            if value == default:
                return []
            return [self.cli_flag] if value else ["--no-" + self.cli_flag[2:]]
        if self.kind == "int":
            return [self.cli_flag, str(int(value))]
        if self.kind == "float":
            return [self.cli_flag, f"{float(value):.6g}"]
        if self.kind == "enum":
            return [self.cli_flag, str(value)]
        return [self.cli_flag, str(value)] if str(value) else []


#: inkvec 0.1.x, from `inkvec --help` of v0.1.3 (identical flags in 0.1.0 to 0.1.3).
BUILTIN: tuple[Option, ...] = (
    Option("precision", "float", 0.1,
           "Sets the MDL cost of a coordinate: lambda = ln(extent / precision). Smaller keeps more "
           "detail with more points. It does not set the digits the emitter writes; output "
           "coordinates are fixed at 2 decimals.", 0.01, 10.0, 0.01),
    Option("min_area", "float", 2.0, "Discard features below this area, in px^2.", 0.0, 10000.0, 0.5),
    Option("colors", "int", 64, "Maximum palette size.", 2, 1024, 1),
    Option("merge", "float", 0.035, "OKLab distance below which two colours are one ink.", 0.0, 0.5, 0.001),
    Option("max_dim", "int", 2048,
           "Inputs larger than this on their longer side are traced at this size and the SVG is "
           "written at the original size. Trace time grows with the pixel count. 0 means no cap.",
           0, 16384, 1),
    Option("time_budget", "float", 0.0,
           "Advisory wall-clock budget in seconds. When it runs out the output is still a correct "
           "trace, with more fills or a less polished outline. 0 means no budget.", 0.0, 3600.0, 0.5),
    Option("margin", "float", 0.0,
           "Transparent margin around the output, as a fraction of the larger side; the viewBox "
           "grows, the geometry does not move.", 0.0, 0.5, 0.01),
    Option("cutout", "bool", False,
           "Carry the input's transparency into the output: transparent areas stay holes, a shape "
           "drawn at one opacity comes back with fill-opacity, and white artwork on a transparent "
           "ground survives."),
    Option("no_background", "bool", False,
           "Knock the background out: the face that covers the whole canvas is not painted, so the "
           "artwork sits on transparency."),
    Option("minify", "bool", False, "No ids or groups, no trailing zeros. Same geometry, typically about a tenth smaller."),
    Option("lossy", "enum", "auto",
           "Treat the input as lossily compressed. auto reads the container, on forces noise-aware "
           "intake, off trusts the pixels.", choices=("auto", "on", "off")),
    Option("harmonize", "bool", True, "Repeating shape harmonization (on by default)."),
    Option("harmonize_threshold", "float", 0.92, "Shape equivalence IoU threshold.", 0.5, 1.0, 0.01),
)


@dataclass(frozen=True)
class Override:
    """How the node presents an option, where that differs from the plain CLI option."""

    tooltip: str = ""
    #: Show a boolean as auto/on/off, where auto means on when the input has transparency.
    auto_on_alpha: bool = False


HARMONIZE_TOOLTIP = (
    "Shape harmonization (the CLI default: on). Marks that repeat across the drawing are matched "
    "and redrawn from one consensus shape, which saves paths on repetitive art. Known cost: on "
    "fine line art near the raster's resolution (hairlines, thin rings, small rounded details) it "
    "can move thin lines by about a pixel and push rounded details toward the shared shape. Turn "
    "it off for fine line art."
)

OVERRIDES = {
    "cutout": Override(
        "Carry the input's transparency into the SVG: transparent areas stay holes, a shape drawn "
        "at one opacity keeps fill-opacity, and white artwork on a transparent ground survives. "
        "auto = on when the image or the mask input has any transparency, off for opaque images.",
        auto_on_alpha=True,
    ),
    "harmonize": Override(HARMONIZE_TOOLTIP),
    "lossy": Override(
        "Noise-aware intake for compressed input. The CLI's auto reads the file type, but ComfyUI "
        "hands the node decoded pixels (written as PNG), so auto behaves like a PNG here. Set on "
        "for JPEG/WebP sources and for images from Inkvec Denoise (inkvec --restore does the same)."
    ),
}


# ---------------------------------------------------------------------------- schema
def _kind(prop: dict) -> str | None:
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    enum = prop.get("enum")
    if enum and all(isinstance(e, str) for e in enum):
        return "enum"
    return {"boolean": "bool", "integer": "int", "number": "float", "string": "string"}.get(t)


def option_from_schema(name: str, prop: dict) -> Option | None:
    """One schema property as an Option, or None when the node cannot show it."""
    if name in SKIP or prop.get("deprecated") or prop.get("x-experimental"):
        return None
    kind = _kind(prop)
    if kind is None:
        log.debug("[inkvec] options schema: %s has an unsupported type; use extra_args for it", name)
        return None
    minimum = prop.get("minimum", prop.get("exclusiveMinimum"))
    maximum = prop.get("maximum", prop.get("exclusiveMaximum"))
    default = prop.get("default")
    if default is None:
        default = {"bool": False, "enum": (prop.get("enum") or [""])[0], "string": ""}.get(
            kind, minimum if minimum is not None else 0)
    return Option(
        name=name,
        kind=kind,
        default=default,
        description=str(prop.get("description") or prop.get("title") or ""),
        minimum=minimum,
        maximum=maximum,
        step=prop.get("multipleOf"),
        choices=tuple(prop.get("enum") or ()),
        flag=str(prop.get("x-cli-flag") or ""),
    )


def parse_schema(schema: dict) -> list[Option]:
    props = schema.get("properties")
    if not isinstance(props, dict):
        raise ValueError("the schema has no 'properties' object")
    out = []
    for name, prop in props.items():
        if isinstance(prop, dict):
            opt = option_from_schema(name, prop)
            if opt is not None:
                out.append(opt)
    return out


def merge_with_builtin(schema_opts: list[Option]) -> list[Option]:
    """Built-in order first (stable widget positions), then options only the schema has.
    Values come from the schema, which describes the binary that will run."""
    by_name = {o.name: o for o in schema_opts}
    builtin_names = {o.name for o in BUILTIN}
    dropped = [o.name for o in BUILTIN if o.name not in by_name]
    if dropped:
        log.info("[inkvec] options schema does not list %s; those widgets are not shown", ", ".join(dropped))
    return [by_name[o.name] for o in BUILTIN if o.name in by_name] + [
        o for o in schema_opts if o.name not in builtin_names
    ]


def schema_path() -> Path | None:
    """The schema file to read, without downloading anything."""
    env = os.environ.get(SCHEMA_ENV, "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        log.warning("[inkvec] %s=%s is not a file; using the built-in option table", SCHEMA_ENV, env)
        return None
    try:
        from .binary_manager import resolve

        p = resolve(download_missing=False).parent / SCHEMA_FILENAME
        return p if p.is_file() else None
    except Exception:
        return None


def load_table() -> tuple[list[Option], str]:
    """(options, where they came from)."""
    path = schema_path()
    if path is not None:
        try:
            opts = parse_schema(json.loads(path.read_text(encoding="utf-8")))
            if opts:
                return merge_with_builtin(opts), str(path)
            log.warning("[inkvec] %s lists no usable options; using the built-in option table", path)
        except Exception as exc:
            log.warning("[inkvec] could not read %s (%s); using the built-in option table", path, exc)
    return list(BUILTIN), "built-in table (inkvec 0.1.x)"


_table: tuple[list[Option], str] | None = None
_table_lock = threading.Lock()


def table() -> list[Option]:
    """The option table for this process (read once: widgets and arguments must agree)."""
    global _table
    with _table_lock:
        if _table is None:
            _table = load_table()
            log.info("[inkvec] trace options: %s", _table[1])
        return _table[0]


def reset() -> None:
    global _table
    with _table_lock:
        _table = None


def _auto_step(default) -> float:
    s = f"{float(default):.6g}"
    decimals = len(s.split(".")[1]) if "." in s and "e" not in s else 0
    return 10.0 ** -max(2, decimals)


def widget(opt: Option) -> tuple:
    """The ComfyUI input declaration for one option."""
    ov = OVERRIDES.get(opt.name, Override())
    tip = (ov.tooltip or opt.description).strip()
    tip = f"{tip} CLI {opt.cli_flag}." if tip else f"CLI {opt.cli_flag}."
    if opt.kind == "bool" and ov.auto_on_alpha:
        return (["auto", "on", "off"], {"default": "auto", "tooltip": tip})
    if opt.kind == "bool":
        return ("BOOLEAN", {"default": bool(opt.default), "tooltip": tip})
    if opt.kind == "enum":
        return (list(opt.choices), {"default": str(opt.default), "tooltip": tip})
    if opt.kind == "string":
        return ("STRING", {"default": str(opt.default or ""), "tooltip": tip})
    if opt.kind == "int":
        lo = int(opt.minimum) if opt.minimum is not None else min(0, int(opt.default))
        hi = int(opt.maximum) if opt.maximum is not None else _INT_MAX
        return ("INT", {"default": int(opt.default), "min": lo, "max": hi,
                        "step": int(opt.step or 1), "tooltip": tip})
    lo = float(opt.minimum) if opt.minimum is not None else min(0.0, float(opt.default))
    hi = float(opt.maximum) if opt.maximum is not None else _FLOAT_MAX
    step = float(opt.step) if opt.step else _auto_step(opt.default)
    return ("FLOAT", {"default": float(opt.default), "min": lo, "max": hi, "step": step, "tooltip": tip})


def resolve_value(opt: Option, value, has_alpha: bool):
    """A widget value as the option's own type (auto/on/off -> bool)."""
    if opt.kind == "bool" and OVERRIDES.get(opt.name, Override()).auto_on_alpha and isinstance(value, str):
        return value == "on" or (value == "auto" and has_alpha)
    return value


def build_args(values: dict, has_alpha: bool = False, opts: list[Option] | None = None) -> list[str]:
    """CLI arguments for the given widget values; missing values take the option's default."""
    args: list[str] = []
    for opt in opts if opts is not None else table():
        value = values.get(opt.name, opt.default)
        args += opt.cli_args(resolve_value(opt, value, has_alpha))
    return args
