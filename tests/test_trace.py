"""End to end: the real inkvec binary, downloaded through binary_manager, driven through
the ComfyUI node classes with ComfyUI-shaped tensors. Needs network access on first run."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

SVG_NS = "{http://www.w3.org/2000/svg}"
COPPER = (201, 117, 74)


# ---------------------------------------------------------------------------- helpers
def rgba_image(kind: str, size: int = 128) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 128
    if kind == "white_disc":  # white artwork on a transparent ground
        d.ellipse([20 * s, 20 * s, 108 * s, 108 * s], fill=(255, 255, 255, 255))
    elif kind == "copper_frame":  # coloured square with a transparent hole
        d.rectangle([24 * s, 24 * s, 104 * s, 104 * s], fill=COPPER + (255,))
        d.ellipse([48 * s, 48 * s, 80 * s, 80 * s], fill=(0, 0, 0, 0))
    elif kind == "copper_field":  # colour edge to edge, with a small transparent hole
        d.rectangle([0, 0, 127 * s, 127 * s], fill=COPPER + (255,))
        d.ellipse([56 * s, 56 * s, 72 * s, 72 * s], fill=(0, 0, 0, 0))
    else:
        raise ValueError(kind)
    return img


def opaque_image(size: int = 128) -> Image.Image:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([20, 20, 108, 108], fill=(38, 36, 33))
    d.polygon([(64, 36), (92, 88), (36, 88)], fill=COPPER)
    return img


def as_loadimage(img: Image.Image):
    """(IMAGE, MASK) exactly as ComfyUI's LoadImage returns them: RGB with alpha dropped,
    MASK = 1 - alpha, or a 64x64 zero mask when the file has no alpha."""
    rgb = torch.from_numpy(np.asarray(img.convert("RGB"), np.float32) / 255.0)[None]
    if "A" in img.getbands():
        a = torch.from_numpy(np.asarray(img.getchannel("A"), np.float32) / 255.0)
        mask = (1.0 - a)[None]
    else:
        mask = torch.zeros((1, 64, 64))
    return rgb, mask


def defaults(ink) -> dict:
    """Every option widget at its default value, as ComfyUI would pass them."""
    return {o.name: ink.options.widget(o)[1]["default"] for o in ink.options.table()}


def run_trace(ink, image, mask=None, **overrides):
    values = defaults(ink)
    values.update(overrides)
    svgs, preview, mask_out = ink.nodes.InkvecTrace().trace(image, timeout_sec=120, mask=mask, **values)
    return svgs, preview, mask_out


def parse_svg(text: str) -> ET.Element:
    root = ET.fromstring(text)
    assert root.tag == SVG_NS + "svg", root.tag
    return root


def full_canvas_rects(root: ET.Element) -> list:
    """Filled <rect> elements that cover the whole viewBox: how the emitter paints a background."""
    x0, y0, w, h = (float(v) for v in root.get("viewBox").split())
    hits = []
    for el in root.iter(SVG_NS + "rect"):
        x, y = float(el.get("x", 0)), float(el.get("y", 0))
        rw, rh = float(el.get("width", 0)), float(el.get("height", 0))
        if el.get("fill", "black") != "none" and x <= x0 + 0.5 and y <= y0 + 0.5 \
                and x + rw >= x0 + w - 0.5 and y + rh >= y0 + h - 0.5:
            hits.append(el)
    return hits


# ---------------------------------------------------------------------------- binary
def test_binary_downloads_verifies_and_runs(ink):
    bm = ink.binary_manager
    bm.reset()
    root = bm.cache_root()
    path, version = bm.binary_info()
    latest = bm.latest_version()
    # Came from the release download into the (stubbed) ComfyUI models dir, not from
    # INKVEC_BIN or bin/.
    assert path == root / latest / bm.EXE, path
    assert version == f"inkvec {latest}", version
    target, ext = bm.platform_target()
    schema = path.parent / ink.options.SCHEMA_FILENAME
    print(f"\n  {bm.asset_name(latest, target, ext)} -> {path} ({version}); "
          f"options.schema.json in the release: {'yes' if schema.is_file() else 'no'}")


# ---------------------------------------------------------------------------- trace
def test_opaque_image(ink):
    image, mask = as_loadimage(opaque_image())  # 64x64 zero placeholder mask, as LoadImage
    svgs, preview, mask_out = run_trace(ink, image, mask)
    assert len(svgs) == 1
    root = parse_svg(svgs[0])
    assert (root.get("width"), root.get("height")) == ("128", "128")
    assert len(list(root.iter(SVG_NS + "path"))) >= 2
    assert tuple(preview.shape) == (1, 128, 128, 3) and preview.dtype == torch.float32
    assert tuple(mask_out.shape) == (1, 128, 128)
    assert float(mask_out.max()) < 0.01  # opaque trace
    # The render matches the input closely.
    err = (preview - image).abs().mean().item()
    assert err < 0.02, err


@pytest.mark.parametrize("kind", ["white_disc", "copper_frame"])
def test_transparent_input_through_mask(ink, kind):
    src = rgba_image(kind)
    image, mask = as_loadimage(src)
    svgs, preview, mask_out = run_trace(ink, image, mask)  # cutout=auto -> --cutout
    root = parse_svg(svgs[0])
    assert full_canvas_rects(root) == [], "the SVG paints a full-canvas background"
    alpha_in = 1.0 - mask[0]
    alpha_out = 1.0 - mask_out[0]
    # Transparent where the input is transparent: corners, and the hole of the frame.
    for y, x in [(2, 2), (2, 125), (125, 2), (125, 125)] + ([(64, 64)] if kind == "copper_frame" else []):
        assert float(alpha_out[y, x]) < 0.02, (kind, y, x, float(alpha_out[y, x]))
    # Opaque shape in the same place.
    inter = ((alpha_in > 0.5) & (alpha_out > 0.5)).sum().item()
    union = ((alpha_in > 0.5) | (alpha_out > 0.5)).sum().item()
    assert inter / union > 0.97, inter / union
    # Right colour inside the shape.
    want = torch.tensor([1.0, 1.0, 1.0]) if kind == "white_disc" else torch.tensor(COPPER) / 255.0
    probe = (64, 64) if kind == "white_disc" else (32, 32)
    assert torch.allclose(preview[0, probe[0], probe[1]], want, atol=0.02), preview[0, probe[0], probe[1]]


def test_rgba_tensor_without_mask(ink):
    src = rgba_image("white_disc")
    image = torch.from_numpy(np.asarray(src, np.float32) / 255.0)[None]  # [1, H, W, 4]
    svgs, _, mask_out = run_trace(ink, image)
    assert full_canvas_rects(parse_svg(svgs[0])) == []
    assert float((1.0 - mask_out[0])[2, 2]) < 0.02


def resolved_version(ink) -> tuple:
    """The (major, minor, patch) of the binary the node resolved, once it is in place."""
    return ink.binary_manager.parse_version(ink.binary_manager.binary_info()[1])


def test_native_alpha_keeps_holes_without_cutout(ink):
    """inkvec 0.1.4 traces transparency natively, so a hole survives with cutout off (the
    no-flag default): the copper field is painted, the hole in it stays transparent."""
    if resolved_version(ink) < (0, 1, 4):
        pytest.skip("native transparency is the default from inkvec 0.1.4")
    image, mask = as_loadimage(rgba_image("copper_field"))
    svgs, _, mask_out = run_trace(ink, image, mask, cutout="off")  # native_alpha at its default
    assert full_canvas_rects(parse_svg(svgs[0])) == [], "the SVG paints a full-canvas matte"
    assert float((1.0 - mask_out[0])[2, 2]) > 0.98  # the field itself is painted
    assert float(mask_out[0, 64, 64]) > 0.98  # the hole is a hole, not a patch of white


def test_matte_path(ink):
    """native_alpha off and cutout off -- the old matte path, as inkvec up to 0.1.3
    defaulted to: the input is composited onto white first, so a full-bleed picture with a
    small transparent hole comes back with the hole painted over."""
    if resolved_version(ink) < (0, 1, 4):
        pytest.skip("needs inkvec 0.1.4's --no-native-alpha")
    image, mask = as_loadimage(rgba_image("copper_field"))
    svgs, _, mask_out = run_trace(ink, image, mask, cutout="off", native_alpha=False)
    assert full_canvas_rects(parse_svg(svgs[0])), "expected the matte path to paint the canvas"
    assert float(mask_out.max()) < 0.01


def test_batch_traces_every_image(ink):
    a, _ = as_loadimage(opaque_image())
    b, _ = as_loadimage(rgba_image("copper_frame").convert("RGB"))
    svgs, preview, mask_out = run_trace(ink, torch.cat([a, b]))
    assert len(svgs) == 2 and svgs[0] != svgs[1]
    assert tuple(preview.shape) == (2, 128, 128, 3) and tuple(mask_out.shape) == (2, 128, 128)


def test_widget_values_reach_the_cli(ink):
    args = ink.tracer.build_args({**defaults(ink), "harmonize": False, "colors": 8, "minify": True})
    assert "--no-harmonize" in args and "--minify" in args
    assert args[args.index("--colors") + 1] == "8"
    svgs, _, _ = run_trace(ink, as_loadimage(opaque_image())[0], harmonize=False, minify=True, colors=8)
    assert ' id="' not in svgs[0]  # --minify drops ids


def test_cli_errors_are_surfaced(ink):
    image, _ = as_loadimage(opaque_image(64))
    with pytest.raises(RuntimeError) as exc:
        run_trace(ink, image, extra_args="--no-such-flag")
    msg = str(exc.value)
    assert "unknown option --no-such-flag" in msg
    assert "SUPER-RESOLUTION" not in msg  # the usage text is trimmed away
    with pytest.raises(ValueError):
        run_trace(ink, image, extra_args="-o elsewhere.svg")


def test_extra_args_pass_newer_flags_through(ink):
    image, _ = as_loadimage(opaque_image(64))
    svgs, _, _ = run_trace(ink, image, extra_args="--tau 3 --no-gradients")
    parse_svg(svgs[0])


_NO_EXTRAS = r"""
import importlib.util, sys, types
for name in ("resvg_py", "cairosvg", "onnxruntime", "einops", "huggingface_hub"):
    sys.modules[name] = None  # any import of these now raises ImportError
fp = types.ModuleType("folder_paths")
fp.models_dir = sys.argv[2]
sys.modules["folder_paths"] = fp
spec = importlib.util.spec_from_file_location("inkvec_comfyui", sys.argv[1] + "/__init__.py",
                                              submodule_search_locations=[sys.argv[1]])
pkg = importlib.util.module_from_spec(spec); sys.modules["inkvec_comfyui"] = pkg; spec.loader.exec_module(pkg)
import torch
N = pkg.nodes
for cls in N.NODE_CLASS_MAPPINGS.values():
    cls.INPUT_TYPES()
img = torch.zeros(1, 32, 32, 3); img[:, 8:24, 8:24] = 1.0
values = {o.name: pkg.options.widget(o)[1]["default"] for o in pkg.options.table()}
svgs, preview, mask = N.InkvecTrace().trace(img, timeout_sec=60, **values)
assert svgs[0].lstrip().startswith("<svg") and torch.equal(preview, img), "fallback preview"
for call, hint in ((lambda: N.InkvecDenoise().denoise(img, "on", "cpu", 0), "pip install onnxruntime"),
                   (lambda: N.InkvecUpscale().upscale(img, "4x", "cpu", 256), "pip install einops")):
    try:
        call()
        raise AssertionError("expected an install hint: " + hint)
    except RuntimeError as e:
        assert hint in str(e), e
print("OK")
"""


def test_trace_works_without_any_optional_dependency(ink):
    """No renderer, no onnxruntime/einops/huggingface_hub: the trace node still traces (the
    preview falls back to the input) and the cleaners fail with an install hint."""
    import subprocess
    import sys
    from pathlib import Path

    ink.binary_manager.ensure_binary()  # make sure the binary is in the models dir
    root = str(Path(ink.__file__).parent)
    proc = subprocess.run([sys.executable, "-c", _NO_EXTRAS, root, ink.binary_manager.cache_root().parent.as_posix()],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and proc.stdout.strip().endswith("OK"), proc.stdout + proc.stderr
    assert "no SVG renderer installed" in proc.stderr


def test_save_svg_numbers_files(ink, work_dir):
    image, _ = as_loadimage(opaque_image(64))
    svg = run_trace(ink, image)[0][0]
    node = ink.nodes.InkvecSaveSVG()
    r1 = node.save(svg, "tests/logo")
    r2 = node.save(svg, "tests/logo")
    names = [r["ui"]["images"][0]["filename"] for r in (r1, r2)]
    assert names == ["logo_00001_.svg", "logo_00002_.svg"]
    saved = work_dir / "output" / "tests" / names[1]
    assert saved.read_text(encoding="utf-8") == svg
    with pytest.raises(ValueError):
        node.save(svg, "../outside")
