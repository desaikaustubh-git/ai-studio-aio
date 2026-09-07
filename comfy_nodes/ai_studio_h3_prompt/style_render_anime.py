"""ComfyUI node: H3 multi-shot render locked to the modern anime look."""

from .style_render_base import make_style_node
from .styles.style_anime import STYLE

AIStudioH3RenderAnime = make_style_node(STYLE)

NODE_CLASS_MAPPINGS = {"AIStudioH3RenderAnime": AIStudioH3RenderAnime}
NODE_DISPLAY_NAME_MAPPINGS = {"AIStudioH3RenderAnime": "H3 Render — Modern Anime"}
