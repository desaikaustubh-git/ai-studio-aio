"""The AIStudioH3PromptWriter node.

Input  : a plain-language description of the video you want.
Output : a ready-to-run MiniMax H3 prompt (fl2va three-field form, or ref2va
         six-section form), plus an ``info`` string.

The rule set that shapes the prompt lives in ``system_prompts/fl2va.txt`` and
``system_prompts/ref2va.txt`` next to this file - verbatim copies of the
"H3 Prompt Formulas" Claude instruction files. Edit those files to change how
prompts are written; no code change needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_DIR = os.path.join(HERE, "system_prompts")
DRIVER = os.path.join(HERE, "_chatgpt_driver.py")
GEMINI_DRIVER = os.path.join(HERE, "_gemini_driver.py")
QWEN_DRIVER = os.path.join(HERE, "_qwen_driver.py")
IMAGE_DRIVER = os.path.join(HERE, "_image_driver.py")

# Default interpreter that has playwright + can import browser_llm (the ComfyUI
# python has neither). Any interpreter with `pip install playwright` + a
# `playwright install chromium` works - override it on the node if you move machines.
DEFAULT_CHATGPT_PYTHON = (
    r"C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\python.exe"
)

# The chatgpt backend needs browser_llm.py + a logged-in browser_profiles/chatgpt.
# A copy of both is vendored under vendor/ so this whole folder can be relocated
# without the backend losing its dependency; if that copy is missing we fall back
# to the original AI Studio v2 checkout.
BUNDLED_DIR = os.path.join(HERE, "vendor")
FALLBACK_V2_DIR = r"E:\Work\AI Studio ComfyUI"


def _resolve_v2_dir(v2_dir: str = "") -> str:
    """Folder that holds browser_llm.py + browser_profiles/ for the chatgpt backend.

    An explicit widget value wins; then the vendored copy next to this node;
    then the original AI Studio v2 checkout.
    """
    v2_dir = (v2_dir or "").strip()
    if v2_dir:
        return v2_dir
    if os.path.isfile(os.path.join(BUNDLED_DIR, "browser_llm.py")):
        return BUNDLED_DIR
    return FALLBACK_V2_DIR


DEFAULT_V2_DIR = _resolve_v2_dir()

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)
_UNCLOSED_THINK_RE = re.compile(r"^\s*<think(?:ing)?>.*$", re.S | re.I)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.S)

# A field marker that only ever appears in a real finished prompt, per mode.
_PROMPT_MARKERS = {
    "fl2va": ("integrated_multimodal_description:",),
    "ref2va": ("subject_definitions:", "detailed_description:"),
}


def _strip_reasoning(text: str) -> str:
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text)
    if _UNCLOSED_THINK_RE.match(cleaned):
        return ""
    return cleaned.strip()


def _extract_prompt(text: str) -> str:
    """Pull the prompt out of an LLM reply.

    The rule files say to return one plain code block and nothing else, but the
    ChatGPT web app sometimes wraps it in a fence or adds a stray sentence.
    Take the first fenced block if there is one, otherwise the whole reply.
    """
    text = _strip_reasoning(text)
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _looks_like_question(text: str, mode: str) -> bool:
    """True when the model asked a clarifying one-liner instead of a prompt."""
    if not text:
        return False
    if any(marker in text for marker in _PROMPT_MARKERS.get(mode, ())):
        return False
    # A real prompt is long and multi-line; a clarifying question is one short line.
    return len(text) < 400 and text.count("\n") <= 2


def _build_request_block(brief: str, duration_seconds: float, extra: str) -> str:
    lines = [brief.strip()]
    if duration_seconds and duration_seconds > 0:
        lines.append("")
        lines.append(f"Total video duration: {float(duration_seconds):.2f} seconds.")
    if extra.strip():
        lines.append("")
        lines.append(extra.strip())
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #

def _run_ollama(system: str, user: str, model: str, base_url: str,
                timeout_s: int) -> str:
    base_url = base_url.rstrip("/")
    payload = json.dumps({
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {"temperature": 0.4, "num_ctx": 24576, "num_predict": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        if exc.code == 404 and "model" in detail.lower():
            raise RuntimeError(
                f"Ollama has no model {model!r}. Installed: {_ollama_tags(base_url)}"
            ) from exc
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url} ({exc.reason}). "
            f"Start it with:  ollama serve"
        ) from exc
    try:
        return json.loads(body).get("response", "")
    except ValueError as exc:
        raise RuntimeError(f"Ollama returned non-JSON: {body[:200]}") from exc


def _ollama_tags(base_url: str) -> str:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return ", ".join(sorted(m.get("name", "") for m in data.get("models", []))) or "(none)"
    except Exception:
        return "(could not list)"


def _run_chatgpt(system: str, user: str, chatgpt_python: str, v2_dir: str,
                 timeout_s: int, attach_paths=None) -> str:
    """Send system+user as one message to the ChatGPT web app via a subprocess.

    browser_llm.py needs Playwright and a logged-in profile, neither of which
    lives in the ComfyUI python, so it runs under ``chatgpt_python`` instead.
    ``attach_paths`` are local files (reference stills, sampled frames, or an
    .mp4) uploaded into the message so ChatGPT can actually see them.
    """
    attach_paths = [str(p) for p in (attach_paths or [])]
    combined = (
        f"{system}\n\n"
        "==================================================\n"
        "YOUR REQUEST\n"
        "==================================================\n\n"
        f"{user}\n"
    )
    work = tempfile.mkdtemp(prefix="h3promptwriter_")
    prompt_file = os.path.join(work, "prompt.txt")
    out_file = os.path.join(work, "reply.txt")
    with open(prompt_file, "w", encoding="utf-8") as fh:
        fh.write(combined)

    if not os.path.isfile(chatgpt_python):
        raise RuntimeError(
            f"chatgpt_python not found: {chatgpt_python}\n"
            f"Point it at any interpreter that has playwright installed "
            f"(pip install playwright && playwright install chromium)."
        )
    if not os.path.isfile(os.path.join(v2_dir, "browser_llm.py")):
        raise RuntimeError(
            f"browser_llm.py not found under v2_dir: {v2_dir}\n"
            f"Expected the vendored copy at "
            f"{os.path.join(BUNDLED_DIR, 'browser_llm.py')}, or set the node's "
            f"v2_dir widget to a folder that has browser_llm.py + browser_profiles/."
        )

    for p in attach_paths:
        if not os.path.isfile(p):
            raise RuntimeError(f"attachment not found: {p}")

    env = dict(os.environ)
    env["BROWSER_PROFILES_DIR"] = os.path.join(v2_dir, "browser_profiles")
    env["PYTHONIOENCODING"] = "utf-8"
    args = [chatgpt_python, DRIVER, v2_dir, prompt_file, out_file, str(int(timeout_s))]
    args.extend(attach_paths)
    try:
        proc = subprocess.run(
            args,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_s + 120, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ChatGPT browser run exceeded {timeout_s + 120}s. Is the ChatGPT "
            f"profile logged in?  {chatgpt_python} \"{v2_dir}\\browser_llm.py\" chatgpt-login"
        ) from exc
    if proc.returncode != 0 or not os.path.isfile(out_file):
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"ChatGPT browser run failed:\n{tail}")
    with open(out_file, "r", encoding="utf-8") as fh:
        return fh.read()


def _run_gemini(system: str, user: str, chatgpt_python: str, v2_dir: str,
                timeout_s: int, attach_paths=None) -> str:
    """Send system+user (and optional local file attachments) to the Gemini web app.

    Same subprocess model as ``_run_chatgpt``: browser_llm.py + a logged-in
    ``browser_profiles/gemini`` profile live under ``chatgpt_python`` (any
    interpreter with Playwright), not the ComfyUI python. ``attach_paths`` are
    local files - reference stills, frames sampled from a render, or an .mp4 -
    uploaded into the message so Gemini can actually see them.
    """
    attach_paths = [str(p) for p in (attach_paths or [])]
    if system:
        combined = (
            f"{system}\n\n"
            "==================================================\n"
            "MATERIAL TO REVIEW\n"
            "==================================================\n\n"
            f"{user}\n"
        )
    else:
        combined = user

    work = tempfile.mkdtemp(prefix="h3_gemini_")
    prompt_file = os.path.join(work, "prompt.txt")
    out_file = os.path.join(work, "reply.txt")
    with open(prompt_file, "w", encoding="utf-8") as fh:
        fh.write(combined)

    if not os.path.isfile(chatgpt_python):
        raise RuntimeError(
            f"chatgpt_python not found: {chatgpt_python}\n"
            f"Point it at any interpreter that has playwright installed "
            f"(pip install playwright && playwright install chromium)."
        )
    if not os.path.isfile(os.path.join(v2_dir, "browser_llm.py")):
        raise RuntimeError(
            f"browser_llm.py not found under v2_dir: {v2_dir}\n"
            f"Expected the vendored copy at "
            f"{os.path.join(BUNDLED_DIR, 'browser_llm.py')}, or set the node's "
            f"v2_dir widget to a folder that has browser_llm.py + browser_profiles/."
        )
    for p in attach_paths:
        if not os.path.isfile(p):
            raise RuntimeError(f"attachment not found: {p}")

    env = dict(os.environ)
    env["BROWSER_PROFILES_DIR"] = os.path.join(v2_dir, "browser_profiles")
    env["PYTHONIOENCODING"] = "utf-8"
    args = [chatgpt_python, GEMINI_DRIVER, v2_dir, prompt_file, out_file, str(int(timeout_s))]
    args.extend(attach_paths)
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_s + 120, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Gemini browser run exceeded {timeout_s + 120}s. Is the Gemini profile "
            f"logged in?  {chatgpt_python} \"{v2_dir}\\browser_llm.py\" gemini-login"
        ) from exc
    if proc.returncode != 0 or not os.path.isfile(out_file):
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"Gemini browser run failed:\n{tail}")
    with open(out_file, "r", encoding="utf-8") as fh:
        return fh.read()


def _run_qwen(system: str, user: str, chatgpt_python: str, v2_dir: str,
              timeout_s: int, attach_paths=None, model: str = "") -> str:
    """Send system+user (and optional local file attachments) to the Qwen Chat
    web app (chat.qwen.ai).

    Same subprocess model as ``_run_gemini``: browser_llm.py + a logged-in
    ``browser_profiles/qwen`` profile live under ``chatgpt_python`` (any
    interpreter with Playwright), not the ComfyUI python. ``attach_paths`` are
    local files - the finished .mp4 or frames sampled from a render - uploaded
    into the message so Qwen can actually watch them. ``model`` is the label to
    pick in Qwen's model dropdown (blank = the site default, Qwen3.7-Plus).
    """
    attach_paths = [str(p) for p in (attach_paths or [])]
    if system:
        combined = (
            f"{system}\n\n"
            "==================================================\n"
            "MATERIAL TO REVIEW\n"
            "==================================================\n\n"
            f"{user}\n"
        )
    else:
        combined = user

    work = tempfile.mkdtemp(prefix="h3_qwen_")
    prompt_file = os.path.join(work, "prompt.txt")
    out_file = os.path.join(work, "reply.txt")
    with open(prompt_file, "w", encoding="utf-8") as fh:
        fh.write(combined)

    if not os.path.isfile(chatgpt_python):
        raise RuntimeError(
            f"chatgpt_python not found: {chatgpt_python}\n"
            f"Point it at any interpreter that has playwright installed "
            f"(pip install playwright && playwright install chromium)."
        )
    if not os.path.isfile(os.path.join(v2_dir, "browser_llm.py")):
        raise RuntimeError(
            f"browser_llm.py not found under v2_dir: {v2_dir}\n"
            f"Expected the vendored copy at "
            f"{os.path.join(BUNDLED_DIR, 'browser_llm.py')}, or set the node's "
            f"v2_dir widget to a folder that has browser_llm.py + browser_profiles/."
        )
    for p in attach_paths:
        if not os.path.isfile(p):
            raise RuntimeError(f"attachment not found: {p}")

    env = dict(os.environ)
    env["BROWSER_PROFILES_DIR"] = os.path.join(v2_dir, "browser_profiles")
    env["PYTHONIOENCODING"] = "utf-8"
    if (model or "").strip():
        env["QWEN_MODEL"] = model.strip()
    args = [chatgpt_python, QWEN_DRIVER, v2_dir, prompt_file, out_file, str(int(timeout_s))]
    args.extend(attach_paths)
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_s + 180, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Qwen browser run exceeded {timeout_s + 180}s. Is the Qwen profile "
            f"logged in?  {chatgpt_python} \"{v2_dir}\\browser_llm.py\" qwen-login"
        ) from exc
    if proc.returncode != 0 or not os.path.isfile(out_file):
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"Qwen browser run failed:\n{tail}")
    with open(out_file, "r", encoding="utf-8") as fh:
        return fh.read()


def _run_image(backend: str, items, out_dir: str, chatgpt_python: str, v2_dir: str,
               timeout_s: int):
    """Generate reference images via the browser image providers.

    ``items`` is a list of {"name", "prompt"}. ``backend`` is "gemini" or "flow".
    Returns a list of (name, abs_path) for the images that were produced (a partial
    set is returned rather than failing outright). Raises only if nothing came back.
    """
    items = [{"name": str(i.get("name") or f"ref{n}"), "prompt": str(i.get("prompt") or "")}
             for n, i in enumerate(items, 1)]
    os.makedirs(out_dir, exist_ok=True)
    items_json = os.path.join(out_dir, "_items.json")
    with open(items_json, "w", encoding="utf-8") as fh:
        json.dump(items, fh)

    if not os.path.isfile(chatgpt_python):
        raise RuntimeError(f"chatgpt_python not found: {chatgpt_python}")
    if not os.path.isfile(os.path.join(v2_dir, "browser_llm.py")):
        raise RuntimeError(
            f"browser_llm.py not found under v2_dir: {v2_dir}\n"
            f"Expected {os.path.join(BUNDLED_DIR, 'browser_llm.py')} or set v2_dir.")

    env = dict(os.environ)
    env["BROWSER_PROFILES_DIR"] = os.path.join(v2_dir, "browser_profiles")
    env["PYTHONIOENCODING"] = "utf-8"
    args = [chatgpt_python, IMAGE_DRIVER, v2_dir, backend, out_dir,
            str(int(timeout_s)), items_json]
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_s + 180, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"reference-image run exceeded {timeout_s + 180}s. Is the {backend!r} "
            f"profile logged in?  {chatgpt_python} \"{v2_dir}\\browser_llm.py\" {backend}-login"
        ) from exc

    got = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("OK "):
            parts = line.split(" ", 2)
            if len(parts) == 3:
                got.append((int(parts[1]), parts[2]))
    got.sort()
    paths = [p for _, p in got]
    if not paths:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"reference-image run produced nothing:\n{tail}")
    # pair back to names in original order
    return [(items[i - 1]["name"], p) for i, p in got]


# --------------------------------------------------------------------------- #
# node
# --------------------------------------------------------------------------- #

class AIStudioH3PromptWriter:
    """Plain description -> MiniMax H3 prompt, via ChatGPT or a local Ollama."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "brief": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the video you want, in plain language.",
                }),
                "mode": (["fl2va", "ref2va"], {"default": "fl2va"}),
                "backend": (["chatgpt", "ollama"], {"default": "chatgpt"}),
            },
            "optional": {
                "duration_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.5,
                }),
                "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
                "ollama_model": ("STRING", {"default": "qwen3:32b"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "chatgpt_python": ("STRING", {"default": DEFAULT_CHATGPT_PYTHON}),
                "v2_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Folder with browser_llm.py + browser_profiles/. "
                               "Blank = use the copy bundled in this node's vendor/ folder.",
                }),
                "timeout_s": ("INT", {"default": 300, "min": 30, "max": 1800}),
                "regen": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF,
                                  "tooltip": "Bump this to force a fresh call "
                                             "(the node caches on identical inputs)."}),
                "system_prompt_override": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "info")
    FUNCTION = "write"
    CATEGORY = "AI Studio/H3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, brief, mode, backend, duration_seconds=0.0,
                   extra_instructions="", ollama_model="", ollama_url="",
                   chatgpt_python="", v2_dir="", timeout_s=300, regen=0,
                   system_prompt_override=""):
        blob = "\x1f".join(str(x) for x in (
            brief, mode, backend, duration_seconds, extra_instructions,
            ollama_model, ollama_url, regen, system_prompt_override,
        ))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _system_prompt(self, mode: str, override: str) -> str:
        if override.strip():
            return override.strip()
        path = os.path.join(SYSTEM_PROMPT_DIR, f"{mode}.txt")
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing rule file: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()

    def write(self, brief, mode, backend, duration_seconds=0.0,
              extra_instructions="", ollama_model="qwen3:32b",
              ollama_url="http://127.0.0.1:11434",
              chatgpt_python=DEFAULT_CHATGPT_PYTHON, v2_dir="",
              timeout_s=300, regen=0, system_prompt_override=""):
        v2_dir = _resolve_v2_dir(v2_dir)
        brief = (brief or "").strip()
        if not brief:
            raise RuntimeError("brief is empty - describe the video you want.")

        system = self._system_prompt(mode, system_prompt_override)
        user = _build_request_block(brief, duration_seconds, extra_instructions)

        t0 = time.time()
        if backend == "ollama":
            raw = _run_ollama(system, user, ollama_model, ollama_url, timeout_s)
            model_label = f"ollama:{ollama_model}"
        else:
            raw = _run_chatgpt(system, user, chatgpt_python, v2_dir, timeout_s)
            model_label = "chatgpt (web)"
        elapsed = time.time() - t0

        prompt = _extract_prompt(raw)
        if not prompt:
            raise RuntimeError(
                f"{model_label} returned no usable text ({len(raw)} raw chars). "
                f"If this is qwen, raise num_predict / check the model is loaded."
            )

        is_question = _looks_like_question(prompt, mode)
        flag = "QUESTION (answer it in 'brief' or set duration_seconds, then re-run): " if is_question else ""
        info = (
            f"{flag}mode={mode} backend={model_label} "
            f"{elapsed:.0f}s {len(prompt)} chars"
        )
        return {"ui": {"h3_prompt": [prompt], "info": [info]},
                "result": (prompt, info)}


NODE_CLASS_MAPPINGS = {"AIStudioH3PromptWriter": AIStudioH3PromptWriter}
NODE_DISPLAY_NAME_MAPPINGS = {"AIStudioH3PromptWriter": "H3 Prompt Writer (AI Studio)"}
