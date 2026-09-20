# inkvec-comfyui

ComfyUI nodes for [Inkvec](https://github.com/logolabs/inkvec), the LogoLabs raster-to-SVG
tracer. Inkvec is built for logos, icons, diagrams and other flat artwork; photographs are
outside its design scope.

| Node | What it does | Needs |
|---|---|---|
| **Inkvec Trace (raster to SVG)** | Traces an `IMAGE` (with optional `MASK` for transparency) to SVG | the `inkvec` binary, downloaded on first use |
| **Inkvec Save SVG** | Writes SVG text to the ComfyUI output folder | nothing |
| **Inkvec Denoise (ConvNeXt)** | Removes JPEG/WebP/AI-decoder damage before tracing | model weights (80 MB, first use) |
| **Inkvec Upscale x4 (MambaIRv2)** | x4 super-resolution for small or blurred logos before tracing | model weights (20 MB, first use) |

The trace node runs the native `inkvec` command-line binary; it is not a Python
re-implementation. Try the tracer without installing anything on the
[Hugging Face Space](https://huggingface.co/spaces/Logolabs/inkvec).

## Install

**ComfyUI Manager:** search for "Inkvec" and install, then restart ComfyUI. The Manager also
installs the package's Python dependencies from `requirements.txt`.

**Manually:**

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/logolabs/inkvec-comfyui.git
```

and install the dependencies into the Python that runs ComfyUI (for the Windows portable
build that is `python_embeded\python.exe -m pip install ...`):

```sh
pip install -r inkvec-comfyui/requirements.txt
```

then restart ComfyUI. The dependencies are everything the four nodes use, so every feature
works after the install: `resvg-py` renders the traced SVG back to pixels (the preview and
mask outputs, and Denoise's auto mode), `onnxruntime` runs the denoiser, `einops` the
upscaler, and `huggingface_hub` fetches the model weights (with a plain-HTTPS fallback if it
is missing). PyTorch, NumPy and Pillow come with ComfyUI. `onnxruntime-gpu` can replace
`onnxruntime` for CUDA, and `cairosvg` works as the renderer too if its native cairo library
is installed.

## First run: the inkvec binary

The first trace downloads the `inkvec` binary for your platform from the latest
[GitHub release](https://github.com/logolabs/inkvec/releases), checks it against the
release's `SHA256SUMS` (it refuses to install a file that does not match), stores it in
`ComfyUI/models/inkvec/<version>/` and runs `inkvec --version` as a health check. The
archive is about 2 MB. Prebuilt binaries exist for Windows x64, Linux x64 and arm64 (glibc),
and macOS x64 and arm64; release v0.1.4, for example, publishes
`inkvec-0.1.4-x86_64-pc-windows-msvc.zip`, `inkvec-0.1.4-x86_64-unknown-linux-gnu.tar.gz`,
`inkvec-0.1.4-aarch64-unknown-linux-gnu.tar.gz`, `inkvec-0.1.4-x86_64-apple-darwin.tar.gz`
and `inkvec-0.1.4-aarch64-apple-darwin.tar.gz`. Native transparency needs inkvec 0.1.4; on
an older release the trace node still carries transparency through `--cutout`.

The binary is looked up in this order:

1. `INKVEC_BIN`: path to an `inkvec` executable.
2. `custom_nodes/inkvec-comfyui/bin/`: drop `inkvec` (or `inkvec.exe`) there to pin a build.
3. The newest version already downloaded into `ComfyUI/models/inkvec/`.
4. A download of the latest release. Set `INKVEC_VERSION=0.1.4` (for example) to pin steps 3
   and 4 to one release.

A downloaded binary is not updated automatically: delete `ComfyUI/models/inkvec/<version>/`
to fetch the latest release again. Offline machines can download the archive by hand and
point `INKVEC_BIN` at the executable inside it. Other platforms can build the binary from
source (`cargo install --path crates/inkvec-cli` in the Inkvec repository).

## Inkvec Trace

**Inputs:** `image` (`IMAGE`), optional `mask` (`MASK`), the options below, `timeout_sec`,
and optional `extra_args`.

**Outputs:**

- `svg` (`STRING`): the SVG source, one string per image in the batch (a list output, so
  downstream nodes run once per image). Connect it to **Inkvec Save SVG**.
- `preview` (`IMAGE`): the SVG rendered back to pixels at the SVG's size, straight colour
  (transparent areas are black). Needs `resvg-py` or `cairosvg`; without one the node passes
  the input image through and logs a note.
- `mask` (`MASK`): the rendered SVG's transparency, 1 = transparent. Together with `preview`
  it gives an RGBA image via ComfyUI's **Join Image with Alpha**.

Every image in a batch is traced separately.

| Option | Default | CLI flag | Notes |
|---|---|---|---|
| `precision` | 0.1 | `--precision` | Sets the MDL cost of a coordinate, lambda = ln(extent / precision); smaller keeps more detail with more points. Output coordinates are always written with 2 decimals. |
| `min_area` | 2.0 | `--min-area` | Features below this area (px²) are discarded. |
| `colors` | 64 | `--colors` | Maximum palette size. |
| `merge` | 0.035 | `--merge` | OKLab distance below which two colours are one ink. |
| `max_dim` | 2048 | `--max-dim` | Inputs larger than this on their longer side are traced at this size; the SVG keeps the original size. 0 = no cap. |
| `time_budget` | 0 | `--time-budget` | Advisory wall-clock budget in seconds; the output is still a correct trace when it runs out. 0 = no budget. |
| `margin` | 0 | `--margin` | Transparent margin, as a fraction of the larger side; the canvas grows, the geometry does not move. |
| `cutout` | auto | `--cutout` | Carry the input's transparency into the SVG; only matters with `native_alpha` off. auto = on when the input has any transparency. See [Transparency](#transparency). |
| `no_background` | off | `--no-background` | Do not paint the face that covers the whole canvas. |
| `minify` | off | `--minify` | No ids or groups, no trailing zeros; same geometry, about a tenth smaller. |
| `lossy` | auto | `--lossy` | Noise-aware intake for compressed input. See below. |
| `harmonize` | on | `--no-harmonize` when off | Shape harmonization. See [Shape harmonization](#shape-harmonization-on-by-default). |
| `harmonize_threshold` | 0.92 | `--harmonize-threshold` | Shape-equivalence IoU threshold for harmonization. |
| `native_alpha` | on | `--no-native-alpha` when off | Trace transparency natively (inkvec 0.1.4): inks carry opacity, holes stay holes. See [Transparency](#transparency). |
| `content_units` | off | `--content-units` | Scale the fit tolerances with the raster: a large, simple drawing gets the parameter count of a small one, at a fidelity cost. |
| `timeout_sec` | 300 | (node only) | The node stops the tracer after this many seconds and reports an error. |
| `extra_args` | empty | (appended) | Further flags, e.g. `--strokes`, `--layers`, `--tau 3`. Appended last, so they override the widgets. `-o`, `--output`, `--help`, `--version` are refused. |

Defaults are the CLI's own (inkvec 0.1.4). The node passes every numeric option
explicitly, so the value a widget shows is the value that runs; booleans are passed only when
they differ from the default. Errors from the CLI (an unknown flag in `extra_args`, a bad
value) are shown in the node's error message.

**`lossy`**: the CLI's `auto` decides by file type (JPEG and lossy WebP on, PNG off). ComfyUI
hands the node decoded pixels, which it writes as PNG, so `auto` here behaves as for a PNG.
Set `on` for images that came from JPEG or WebP files and for the output of Inkvec Denoise
(`inkvec --restore` turns it on for the same reason).

**Options schema.** The option list above is one table in `options.py`, from which both the
widgets and the command line are built. If an `options.schema.json` (JSON Schema:
`properties` with `type`, `default`, `description`, `minimum`, `maximum`) is found next to
the binary, or at the path in `INKVEC_OPTIONS_SCHEMA`, the node builds its widgets from that
instead, so options added in a later Inkvec release appear without a change to this
package. Current releases do not ship the file; the built-in table is used. Property names
map to flags as `snake_case` to `--kebab-case`, booleans become presence flags
(`--no-<name>` for an option that defaults to on), and the neural pre-pass options are left
out. The schema is read once, when ComfyUI starts. The schema is the tracer's `Options`
contract, so the CLI-only flags (such as `--lossy`) would leave the widgets and have to go
through `extra_args` if a release starts shipping it.

### Transparency

ComfyUI's **Load Image** drops the alpha channel from `IMAGE` and returns it as `MASK`, with
1 = transparent (mask = 1 - alpha). Connect that `MASK` to the trace node's `mask` input and
the node rebuilds the RGBA image before tracing. A 4-channel `IMAGE` tensor is also accepted
as RGBA directly; a connected `mask` takes precedence over a fourth channel. A mask where 1
marks the subject (a segmentation mask) is the other way round and needs **Invert Mask**
first.

Inkvec 0.1.4 traces transparency natively (`native_alpha`, on by default): each ink is a
colour and an opacity, and the transparent ground is an ink of its own. Transparent areas
stay holes instead of being painted, a shape drawn at a single opacity comes back with
`fill-opacity`, a glow or fade becomes one gradient of `stop-color` and `stop-opacity`, and
white artwork on a transparent ground traces at all. An opaque input traces exactly as
without it. Set `native_alpha` off to composite onto a matte first, as inkvec releases up to
0.1.3 did (the CLI flag is `--no-native-alpha`).

`cutout` is the older transparency carrier: under the matte path it punches the input's
transparent areas out of the faces above them and picks the matte so white artwork survives.
The node's `cutout` widget defaults to auto, which passes `--cutout` for any input that has
transparency: that is what carries the holes on inkvec 0.1.3 and older, and under native
tracing it changes nothing, so the default is safe on both.

### Shape harmonization (on by default)

After fitting, marks that repeat across the drawing (a run of identical tabs, segmented
rings, tiled glyphs) are matched by outline similarity (IoU threshold `harmonize_threshold`,
default 0.92) and redrawn from one consensus shape per cluster. This saves parameters on
repetitive art.

Since inkvec 0.1.4 the pass is held to the traced boundary: a mark takes the consensus only
where that stays within 0.1 px of where its own pixels put it and costs fewer parameters, a
face another face is drawn against is never moved (so harmonizing cannot open a gap onto a
transparent ground), and neither is a fitted circle or rounded rectangle. On Inkvec's
246-icon screen set with the default flags these changes brought the mean colour error
(dE00) to 0.148, with no icon above 1.0, and the alpha-channel error of harmonized icons
back to the unharmonized level. Turn `harmonize` off to skip the pass; it stays on by
default to match the CLI.

## Inkvec Save SVG

Writes the `svg` string to `ComfyUI/output/<prefix>_00001_.svg`, numbering files the way
**Save Image** does; `filename_prefix` may contain a subfolder (`logos/inkvec`). Paths outside
the output folder are refused. The saved file is listed in the node's output; the ComfyUI
frontend shows it as an image where it can display SVG.

## Cleaners (optional)

Two networks from the Inkvec project that prepare a raster before tracing. Both are image to
image, keep transparency, and download their weights from Hugging Face on first use into
`ComfyUI/models/inkvec/<model>/`, pinned to a fixed revision and checked against a SHA-256
before use. A local file can be given in `weights_path` instead (used as-is, not
hash-checked). Each node fails with an install hint if its dependency is missing; the trace
nodes work without either.

### Inkvec Denoise (ConvNeXt)

For inputs damaged by JPEG or WebP compression or by a diffusion model's VAE decode. It is
the restorer behind `inkvec --restore`: a 19.7M-parameter ConvNeXt U-Net,
[`Logolabs/inkvec-denoiser-001`](https://huggingface.co/Logolabs/inkvec-denoiser-001)
(Apache-2.0), run through ONNX Runtime. The model card reports, end to end with tracing on
144 damaged image-format pairs, colour error down 28%, DISTS down 53% and parameter count
down 31%.

The node reproduces the engine's processing (`crates/inkvec-restore`): the image is
composited onto white (the network is RGB-only and was trained on opaque renders), padded
to a multiple of 16 by replicating the last row and column, restored, clamped, quantised to
8-bit levels, and pixels within 6 levels of pure white or black on every channel are snapped
to it. Alpha passes through unchanged, and so does the `mask`.

- `mode`: `on` restores every image. `auto` does what `inkvec --restore auto` does: it traces
  the image once (with the CLI defaults), renders the trace, and restores only if the input
  disagrees with its own trace where the trace is flat (interior residual above 0.5);
  otherwise the image passes through untouched. `auto` needs the inkvec binary and
  `resvg-py`. The restorer costs a few percent of colour accuracy on clean input, which is
  why `auto` exists.
- `tile`: 0 runs the whole image at once, as the engine does. The model card's accuracy
  figures were measured with 128-px tiles (32-px overlap), which also avoid a faint grey wash
  the network paints over large mostly-white images at whole-image inference; set 128 for
  that. Tiling also bounds memory on large images.
- `device`: `auto` uses ONNX Runtime's CUDA provider when it is available
  (`onnxruntime-gpu`, CUDA 12, cuDNN 9), else the CPU. Repeated runs give identical output on
  both. On a Ryzen 7 5800X a whole-image pass took 1.7 s at 256 px, 7 s at 512 px and 25 s at
  1024 px.

After Denoise, set the trace node's `lossy` to `on`.

### Inkvec Upscale x4 (MambaIRv2)

For small, blurred or low-resolution logos. It is the upscaler behind `inkvec --sr`: a
fine-tune of MambaIRv2-Small (9.77M parameters) on logo and icon art,
[`Logolabs/inkvec-sr-001`](https://huggingface.co/Logolabs/inkvec-sr-001) (Apache-2.0).

- `scale`: `4x` returns the network's output. `2x` is the recipe `inkvec --sr on` traces:
  upscale x4, box-average back to x2, then refit the flat colours to the source's with one
  affine map per channel (on clean input the network alone costs about 0.6 dE00 in flat
  interiors; the refit takes the colour there from the source).
- Transparency: the network sees RGB only. Colour is zeroed where alpha is 0 before the
  network sees it, so transparent logos do not pick up a dark halo, and alpha is resized
  separately with Lanczos. The `mask` output is at the new size.
- `tile`: tiles of this many input pixels with a 32-px overlap, blended with a feathered
  weight so that a seam does not become an edge for the tracer.
- Determinism: MambaIRv2 routes pixels with a Gumbel-softmax that samples noise on every
  forward pass, even in eval mode; unpinned, repeated runs of one image differ by up to 12.45
  levels. The node pins the random generator to the reference seed `0x56414331` around every
  forward pass and restores ComfyUI's generator state afterwards, so the same input gives the
  same output. (Replacing the sampling with its argmax limit is deterministic too but was
  measured worse, dE00 0.5492 against 0.5364.)
- Hardware: CUDA is recommended. The selective scan uses the `mamba-ssm` CUDA kernel when that
  package is installed and imports cleanly, and otherwise a pure-PyTorch parallel scan
  (vendored from the Inkvec repository) that runs on CPU and any GPU. On CUDA the forward
  pass runs in bfloat16. CPU works but is slow. Measured through the node without
  `mamba-ssm`, on a Ryzen 7 5800X: 36 s for a 64-px input, 62 s for 128 px; on an RTX 4060
  that was also running other work: 1.3 s for 64 px, 17 s for 256 px, 153 s for 512 px
  (nine 256-px tiles). Indicative only.

MambaIRv2 is by Hang Guo, Yong Guo, Yaohua Zha, Yulun Zhang, Wenbo Li, Tao Dai, Shu-Tao Xia
and Yawei Li, *MambaIRv2: Attentive State Space Restoration*, CVPR 2025
([arXiv:2411.15269](https://arxiv.org/abs/2411.15269)); it builds on MambaIR (Guo et al.,
ECCV 2024). The official implementation is [github.com/csguoh/MambaIR](https://github.com/csguoh/MambaIR)
(Apache-2.0). `vendor/mambairv2_arch.py` is the standalone architecture file from the
`inkvec-sr-001` model repository, derived from it; its header lists the changes.

## Example workflows

Drag a file from `examples/` onto the ComfyUI canvas.

- [`inkvec_trace.json`](examples/inkvec_trace.json): Load Image (IMAGE and MASK) -> Inkvec
  Trace -> Inkvec Save SVG, with the preview in Preview Image.
- [`inkvec_clean_and_trace.json`](examples/inkvec_clean_and_trace.json): Load Image -> Inkvec
  Denoise -> Inkvec Upscale (2x recipe) -> Inkvec Trace (`lossy` on) -> Inkvec Save SVG, the
  same order as `inkvec --restore on --sr on`. The masks are chained through each node.

## Limits

- The trace node cannot run the CLI's neural pre-passes (`--restore`, `--sr`): the released
  default binaries have no restorer compiled in and need its weights, and `--sr` needs the
  Python upscaler package. Use the Denoise and Upscale nodes instead, which run the same
  networks inside ComfyUI. The CLI's `--sr auto` decision is not reproduced; the Upscale node
  always upscales.
- Large inputs are traced at `max_dim` (2048 px by default) on their longer side; trace time
  grows with the pixel count. Raise `timeout_sec` or set `time_budget` for very large or
  gradient-heavy images.
- Text is traced as outlines, not `<text>`; photographs trace poorly (banding, many paths).
  See the Inkvec repository's `docs/LIMITATIONS.md`.
- `native_alpha` and `content_units` need inkvec 0.1.4. On an older installed release,
  moving either widget off its default is reported by the CLI as an unknown option; with the
  defaults the node traces fine on any 0.1.x release.

## Development

Tests run without ComfyUI: `tests/conftest.py` stubs ComfyUI's `folder_paths` module and
loads the package the way ComfyUI does. They download the real binary (and, for the cleaner
tests, the model weights) into a temporary directory; set `INKVEC_TEST_MODELS_DIR` to a
persistent directory to reuse the downloads.

```sh
python -m pytest tests
```

Cleaner tests skip when `einops` or `onnxruntime` is not installed, and run on CUDA as well
when it is available.

## Licence

Apache-2.0, the same as [Inkvec](https://github.com/logolabs/inkvec). The downloaded binaries
and both models are released under Apache-2.0 by LogoLabs. `vendor/` contains code derived
from MambaIR/MambaIRv2 (Apache-2.0, see above) and from the Inkvec repository.
