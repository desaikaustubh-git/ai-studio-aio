"""Run one Qwen Chat (chat.qwen.ai) web turn and write the reply to a file.

Called as a subprocess by prompt_writer._run_qwen under a python that has
Playwright and can import browser_llm.py. Kept separate so the ComfyUI python
never needs Playwright. <v2_dir> is the folder that holds browser_llm.py +
browser_profiles/ - normally this node's own vendor/ folder.

    python _qwen_driver.py <v2_dir> <prompt_file> <out_file> <timeout_s> [<file> ...]

Any trailing <file> arguments are uploaded into the message (the finished .mp4 or
frames sampled from a render) before the prompt is sent. The model to select is
read from the QWEN_MODEL environment variable (default: Qwen3.7-Plus).
"""

import os
import sys


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: _qwen_driver.py <v2_dir> <prompt_file> <out_file> "
              "<timeout_s> [file ...]", file=sys.stderr)
        return 2
    v2_dir, prompt_file, out_file, timeout_s = sys.argv[1:5]
    attach = [p for p in sys.argv[5:] if p]

    sys.path.insert(0, v2_dir)
    os.environ.setdefault(
        "BROWSER_PROFILES_DIR", os.path.join(v2_dir, "browser_profiles")
    )

    try:
        from browser_llm import QwenAutomation
    except Exception as exc:  # noqa: BLE001 - surface the import problem verbatim
        print(f"import browser_llm failed: {exc}", file=sys.stderr)
        return 3

    missing = [p for p in attach if not os.path.isfile(p)]
    if missing:
        print(f"attachment(s) not found: {missing}", file=sys.stderr)
        return 5

    with open(prompt_file, "r", encoding="utf-8") as fh:
        text = fh.read()

    try:
        bot = QwenAutomation(headless=False, timeout_s=int(timeout_s),
                             model=os.environ.get("QWEN_MODEL", ""))
        if attach:
            reply = bot.send_prompt_with_files(text, attach)
        else:
            reply = bot.send_prompt(text)
    except Exception as exc:  # noqa: BLE001
        print(f"Qwen automation failed: {exc}", file=sys.stderr)
        return 4

    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write(reply or "")
    print(f"OK chars={len(reply or '')} attach={len(attach)} "
          f"model={os.environ.get('QWEN_MODEL', '(default)')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
