"""Test harness: loads the package the way ComfyUI does, with no ComfyUI install.

ComfyUI imports a custom node directory as a package from its ``__init__.py``; the same is
done here under the name ``inkvec_comfyui``. ``folder_paths`` (ComfyUI's module) is replaced
by a stub whose models/temp/output directories live in a fresh temporary directory, so the
binary and model downloads are exercised for real on every run. Set
``INKVEC_TEST_MODELS_DIR`` to a persistent directory to reuse downloads between runs.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_WORK = Path(tempfile.mkdtemp(prefix="inkvec-comfyui-test-"))

for var in ("INKVEC_BIN", "INKVEC_VERSION", "INKVEC_OPTIONS_SCHEMA"):
    os.environ.pop(var, None)

folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = os.environ.get("INKVEC_TEST_MODELS_DIR") or str(_WORK / "models")
folder_paths.get_temp_directory = lambda: str(_WORK / "temp")
folder_paths.get_output_directory = lambda: str(_WORK / "output")
sys.modules["folder_paths"] = folder_paths

_spec = importlib.util.spec_from_file_location(
    "inkvec_comfyui", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
pkg = importlib.util.module_from_spec(_spec)
sys.modules["inkvec_comfyui"] = pkg
_spec.loader.exec_module(pkg)


@pytest.fixture(scope="session")
def work_dir() -> Path:
    return _WORK


@pytest.fixture(scope="session")
def ink():
    """The loaded package; submodules are attributes (ink.nodes, ink.options, ...)."""
    import inkvec_comfyui.binary_manager  # noqa: F401
    import inkvec_comfyui.cleaners  # noqa: F401
    import inkvec_comfyui.options  # noqa: F401
    import inkvec_comfyui.tracer  # noqa: F401

    return pkg


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_WORK, ignore_errors=True)
