"""ComfyUI node: H3 multi-shot render locked to the Studio Ghibli look."""

from .style_render_base import make_style_node
from .styles.style_ghibli import STYLE

AIStudioH3RenderGhibli = make_style_node(STYLE)

NODE_CLASS_MAPPINGS = {"AIStudioH3RenderGhibli": AIStudioH3RenderGhibli}
NODE_DISPLAY_NAME_MAPPINGS = {"AIStudioH3RenderGhibli": "H3 Render — Studio Ghibli"}
