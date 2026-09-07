"""ComfyUI node: H3 multi-shot render locked to the photoreal (default) look."""

from .style_render_base import make_style_node
from .styles.style_photoreal import STYLE

AIStudioH3RenderPhotoreal = make_style_node(STYLE)

NODE_CLASS_MAPPINGS = {"AIStudioH3RenderPhotoreal": AIStudioH3RenderPhotoreal}
NODE_DISPLAY_NAME_MAPPINGS = {"AIStudioH3RenderPhotoreal": "H3 Render — Photoreal"}
