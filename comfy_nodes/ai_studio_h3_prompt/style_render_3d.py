"""ComfyUI node: H3 multi-shot render locked to the 3D Pixar-style look."""

from .style_render_base import make_style_node
from .styles.style_3d import STYLE

AIStudioH3Render3D = make_style_node(STYLE)

NODE_CLASS_MAPPINGS = {"AIStudioH3Render3D": AIStudioH3Render3D}
NODE_DISPLAY_NAME_MAPPINGS = {"AIStudioH3Render3D": "H3 Render — 3D Pixar-style"}
