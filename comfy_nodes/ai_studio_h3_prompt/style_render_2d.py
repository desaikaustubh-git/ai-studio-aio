"""ComfyUI node: H3 multi-shot render locked to the 2D hand-drawn look."""

from .style_render_base import make_style_node
from .styles.style_2d import STYLE

AIStudioH3Render2D = make_style_node(STYLE)

NODE_CLASS_MAPPINGS = {"AIStudioH3Render2D": AIStudioH3Render2D}
NODE_DISPLAY_NAME_MAPPINGS = {"AIStudioH3Render2D": "H3 Render — 2D Hand-Drawn"}
