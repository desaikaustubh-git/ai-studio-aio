"""Generate reference images via the browser image providers and save them to disk.

Called as a subprocess by prompt_writer._run_image under a python that has
Playwright + can import browser_llm.py (not the ComfyUI python).

    python _image_driver.py <v2_dir> <backend> <out_dir> <timeout_s> <items_json>

<items_json> is a file holding a JSON list: [{"name": "...", "prompt": "..."}, ...]
An item may also carry "refs": ["abs/path.png", ...] -- local reference images fed
to the model so the generated image keeps their identity / geometry (multi-reference
consistency). Honoured for the "gemini" backend only; ignored otherwise.
<backend> is "gemini" or "flow".

For each item it prints one line to stdout:
    OK <index> <abs_path>
    FAIL <index> <name> <error, single line>
Exit 0 if at least one image was produced, else non-zero.
"""

import json
import os
import re
import sys


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())[:40] or "ref"


def main() -> int:
    if len(sys.argv) != 6:
        print("usage: _image_driver.py <v2_dir> <backend> <out_dir> <timeout_s> <items_json>",
              file=sys.stderr)
        return 2
    v2_dir, backend, out_dir, timeout_s, items_json = sys.argv[1:6]
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, v2_dir)
    os.environ.setdefault("BROWSER_PROFILES_DIR", os.path.join(v2_dir, "browser_profiles"))

    try:
        from browser_llm import GeminiAutomation, FlowAutomation
    except Exception as exc:  # noqa: BLE001
        print(f"import browser_llm failed: {exc}", file=sys.stderr)
        return 3

    with open(items_json, "r", encoding="utf-8") as fh:
        items = json.load(fh)
    if not isinstance(items, list) or not items:
        print("items_json is not a non-empty list", file=sys.stderr)
        return 2

    cls = FlowAutomation if backend == "flow" else GeminiAutomation
    bot = cls(headless=False, timeout_s=int(timeout_s))

    produced = 0
    bot.open_session()
    try:
        for idx, it in enumerate(items, 1):
            name = str(it.get("name") or f"ref{idx}")
            prompt = str(it.get("prompt") or "").strip()
            refs = [str(r) for r in (it.get("refs") or []) if os.path.isfile(str(r))]
            out_path = os.path.join(out_dir, f"ref_{idx:02d}_{_safe(name)}.png")
            if not prompt:
                print(f"FAIL {idx} {name} empty prompt")
                continue
            try:
                if refs and backend != "flow" and hasattr(bot, "ask_image_with_refs"):
                    bot.ask_image_with_refs(prompt, refs, out_path)
                else:
                    bot.ask_image(prompt, out_path)
                if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                    produced += 1
                    print(f"OK {idx} {out_path}", flush=True)
                else:
                    print(f"FAIL {idx} {name} no file written", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep going, a partial set is useful
                msg = " ".join(str(exc).split())[:200]
                print(f"FAIL {idx} {name} {msg}", flush=True)
                try:
                    if hasattr(bot, "reset_chat"):
                        bot.reset_chat()
                except Exception:
                    pass
    finally:
        try:
            bot.close_session()
        except Exception:
            pass

    print(f"done: {produced}/{len(items)} images", file=sys.stderr)
    return 0 if produced else 4


if __name__ == "__main__":
    raise SystemExit(main())
