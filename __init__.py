"""Inkvec for ComfyUI: raster-to-SVG tracing plus two optional raster cleaners."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Browser-side branding: web/js/inkvec_brand.js draws the LogoLabs flask and the
# wordmark on every Inkvec node. Removing the directory (and this line) changes
# nothing about how the nodes run.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
