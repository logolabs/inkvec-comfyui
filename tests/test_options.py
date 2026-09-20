"""The option table: one declaration drives both the widgets and the CLI arguments."""

import json

import pytest


def test_builtin_table_matches_the_cli_defaults(ink):
    o = ink.options
    args = o.build_args({}, opts=list(o.BUILTIN))
    assert args == [
        "--precision", "0.1", "--min-area", "2", "--colors", "64", "--merge", "0.035",
        "--max-dim", "2048", "--time-budget", "0", "--margin", "0", "--lossy", "auto",
        "--harmonize-threshold", "0.92",
    ]


def test_boolean_mapping(ink):
    o = ink.options
    table = list(o.BUILTIN)
    args = o.build_args({"harmonize": False, "no_background": True, "minify": True}, opts=table)
    assert "--no-harmonize" in args and "--no-background" in args and "--minify" in args
    assert "--harmonize" not in args  # the default is never passed for a boolean
    # cutout is shown as auto/on/off; auto follows the input's transparency.
    assert "--cutout" in o.build_args({"cutout": "auto"}, has_alpha=True, opts=table)
    assert "--cutout" not in o.build_args({"cutout": "auto"}, has_alpha=False, opts=table)
    assert "--cutout" in o.build_args({"cutout": "on"}, has_alpha=False, opts=table)
    assert "--cutout" not in o.build_args({"cutout": "off"}, has_alpha=True, opts=table)
    # The 0.1.4 booleans: defaults pass nothing, moving off the default passes the flag.
    assert "--no-native-alpha" in o.build_args({"native_alpha": False}, opts=table)
    assert "--native-alpha" not in o.build_args({"native_alpha": True}, opts=table)
    assert "--content-units" in o.build_args({"content_units": True}, opts=table)
    assert "--content-units" not in o.build_args({"content_units": False}, opts=table)


def test_widgets_come_from_the_table(ink):
    o = ink.options
    types = ink.nodes.InkvecTrace.INPUT_TYPES()["required"]
    assert list(types)[0] == "image" and list(types)[-1] == "timeout_sec"
    assert [k for k in types if k not in ("image", "timeout_sec")] == [x.name for x in o.table()]
    assert types["cutout"][0] == ["auto", "on", "off"]
    assert types["lossy"][0] == ["auto", "on", "off"]
    assert types["harmonize"][1]["default"] is True
    assert "within 0.1 px" in types["harmonize"][1]["tooltip"]
    assert types["native_alpha"][1]["default"] is True
    assert types["content_units"][1]["default"] is False
    assert types["merge"][1]["step"] == 0.001


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "precision": {"type": "number", "default": 0.2, "minimum": 0.01, "description": "Coordinate cost."},
        "colors": {"type": "integer", "default": 32, "minimum": 1, "maximum": 256},
        "harmonize": {"type": "boolean", "default": True, "description": "Harmonize."},
        "lossy": {"type": "string", "enum": ["auto", "on", "off"], "default": "auto"},
        "stroke_balance": {"type": "number", "default": 0.9, "minimum": 0, "maximum": 1},
        "strokes": {"type": "boolean", "default": False, "description": "Emit strokes."},
        "max_colors_legacy": {"type": "integer", "default": 64, "x-cli-flag": "--colors-legacy"},
        "native_alpha": {"type": "boolean", "default": False, "x-experimental": True},
        "old_thing": {"type": "boolean", "default": False, "deprecated": True},
        "sr": {"type": "string", "enum": ["auto", "on", "off"], "default": "off"},
        "output": {"type": "string"},
        "blob": {"type": "object"},
    },
}


def test_schema_drives_the_table(ink, tmp_path, monkeypatch):
    o = ink.options
    path = tmp_path / "options.schema.json"
    path.write_text(json.dumps(SCHEMA), encoding="utf-8")
    monkeypatch.setenv(o.SCHEMA_ENV, str(path))
    table, source = o.load_table()
    assert source == str(path)
    names = [x.name for x in table]
    # Built-in order first, then schema-only options in schema order; skipped ones absent.
    assert names == ["precision", "colors", "lossy", "harmonize", "stroke_balance", "strokes", "max_colors_legacy"]
    by = {x.name: x for x in table}
    assert by["precision"].default == 0.2  # the schema, not the built-in table, is authoritative
    assert by["stroke_balance"].cli_flag == "--stroke-balance"  # snake_case -> --kebab-case
    assert by["max_colors_legacy"].cli_flag == "--colors-legacy"  # x-cli-flag wins
    args = o.build_args({"strokes": True, "harmonize": False, "colors": 16}, opts=table)
    assert args == [
        "--precision", "0.2", "--colors", "16", "--lossy", "auto", "--no-harmonize",
        "--stroke-balance", "0.9", "--strokes", "--colors-legacy", "64",
    ]
    w = o.widget(by["colors"])
    assert w[0] == "INT" and (w[1]["min"], w[1]["max"], w[1]["default"]) == (1, 256, 32)
    assert o.widget(by["strokes"])[0] == "BOOLEAN"


def test_bad_schema_falls_back_to_builtin(ink, tmp_path, monkeypatch):
    o = ink.options
    path = tmp_path / "options.schema.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv(o.SCHEMA_ENV, str(path))
    table, source = o.load_table()
    assert source.startswith("built-in") and table == list(o.BUILTIN)
    monkeypatch.setenv(o.SCHEMA_ENV, str(tmp_path / "missing.json"))
    assert o.load_table()[1].startswith("built-in")


@pytest.mark.parametrize("text,ok", [("", True), ("--strokes --tau 3", True), ("--output x.svg", False),
                                     ("-h", False), ('--sr-command "a b"', True), ('"unclosed', False)])
def test_extra_args_parsing(ink, text, ok):
    if ok:
        ink.tracer.parse_extra_args(text)
    else:
        with pytest.raises(ValueError):
            ink.tracer.parse_extra_args(text)
