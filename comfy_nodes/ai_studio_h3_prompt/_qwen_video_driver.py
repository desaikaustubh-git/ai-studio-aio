"""Run ONE Qwen Chat (chat.qwen.ai) VIDEO-understanding turn and write the reply
to a file.

Same subprocess contract as _qwen_driver.py -- called under a python that has
Playwright and can import browser_llm.py:

    python _qwen_video_driver.py <v2_dir> <prompt_file> <out_file> <timeout_s> [<file> ...]

The trailing files are uploaded before the prompt is sent; the first one is
expected to be the .mp4, the rest sampled stills.

Why a separate driver instead of _qwen_driver.py + QwenAutomation
----------------------------------------------------------------
QwenAutomation._attach_files decides "the upload finished" by requiring every
attachment's <img>/<video> `src` to have flipped to an https OSS URL. On this box
chat.qwen.ai keeps the <video> preview on a blob: src and the server-side
transcode outlasts the 240 s wait, so a real .mp4 upload ALWAYS times out empty
even though the tile is visibly attached and playable.

This driver's QwenVideoAutomation keeps the exact same picker-open path but waits
on the composer's own "ready to send" signals instead -- the Send button
re-enabling and the attachment tiles settling (no spinner) -- with a long ceiling
for the transcode. If the ceiling is hit it sends anyway and lets _send() confirm
via the one unambiguous signal (the composer actually emptying). Images and text
Qwen turns are unaffected: they still go through _qwen_driver.py.
"""

import os
import sys
import time


def _make_cls(v2_dir):
    from browser_llm import QwenAutomation

    class QwenVideoAutomation(QwenAutomation):
        provider = "qwen"

        # Send-button candidates (same set QwenAutomation._send tries).
        _SEND_SEL = (
            "button[aria-label*='send' i]",
            "button[data-testid*='send' i]",
            "button:has(svg[class*='send' i])",
            "button[type='submit']",
        )

        def __init__(self, headless=False, timeout_s=900, model=""):
            # video understanding + transcode: keep the ceiling generous
            super().__init__(headless=headless,
                             timeout_s=max(int(timeout_s or 0), 900),
                             model=model)

        # -- upload readiness ------------------------------------------------
        def _tile_count(self, page):
            """How many attachment tiles are showing in the composer, by any of
            several class shapes Qwen has used. Only a sanity floor (>=1) -- the
            real 'uploads done' signal is the Send button re-enabling."""
            try:
                return int(page.evaluate(
                    "() => document.querySelectorAll("
                    "'img.vision-item-image, video.vision-item-video, "
                    "[class*=vision-item], [class*=file-card], "
                    "[class*=attachment-card], [class*=upload-file-item], "
                    "[class*=chat-attachment], [class*=file-item]').length"
                ) or 0)
            except Exception:
                return 0

        def _any_spinner(self, page):
            """Best-effort: is any attachment tile still showing an upload
            spinner / progress ring. Class names are a guess; a false 'True'
            only delays us to the ceiling (then we send anyway), a false
            'False' is caught by the Send-button check."""
            try:
                return bool(page.evaluate(
                    "() => !!document.querySelector("
                    "'[class*=vision-item] [class*=loading], "
                    "[class*=vision-item] [class*=progress], "
                    "[class*=file-item] [class*=loading], "
                    "[class*=attachment] [class*=loading], "
                    "[class*=uploadProgress], [class*=upload-progress]')"
                ))
            except Exception:
                return False

        def _send_enabled(self, page):
            """Qwen disables Send while any attachment is still uploading /
            transcoding, so an enabled Send button == every upload landed."""
            for sel in self._SEND_SEL:
                try:
                    b = page.locator(sel).first
                    if b.count() and b.is_visible(timeout=400):
                        return bool(b.is_enabled())
                except Exception:
                    continue
            return False

        def _open_picker_and_set(self, page, paths):
            """The QwenAutomation picker-open path, verbatim, minus its
            https-src wait loop. Raises if the picker never opens."""
            opened = False
            for _ in range(3):
                menu_open = False
                try:
                    menu_open = page.locator(self._UPLOAD_ITEM_SEL).first.is_visible(timeout=400)
                except Exception:
                    menu_open = False
                if not menu_open:
                    try:
                        page.locator("[aria-label='Select Mode']").first.click(timeout=3000)
                        page.wait_for_timeout(700)
                    except Exception:
                        pass
                try:
                    item = page.locator(self._UPLOAD_ITEM_SEL).first
                    with page.expect_file_chooser(timeout=6000) as fc:
                        item.click(timeout=3000)
                    fc.value.set_files(paths, timeout=120_000)
                    opened = True
                    break
                except Exception:
                    pass
                try:
                    fi = page.locator("input#filesUpload, .mode-select input[type='file'], "
                                      "input[type='file']").first
                    if fi.count():
                        fi.set_input_files(paths, timeout=120_000)
                        opened = True
                        break
                except Exception:
                    pass
                try:
                    page.keyboard.press("Escape")
                except Exception:
                    pass
                page.wait_for_timeout(500)
            if not opened:
                raise RuntimeError("Qwen(video): could not open the upload picker "
                                   "('+' / Select Mode -> 'Upload attachment')")

        def _attach_files(self, page, paths, upload_wait_s=600):
            want = len(paths)
            self._open_picker_and_set(page, paths)
            # let Qwen register the upload and flip Send -> disabled before we
            # start trusting a Send-enabled reading (avoids a false-early pass in
            # the gap between set_files and the upload actually starting)
            page.wait_for_timeout(5000)

            # The real "every upload landed" signal is Qwen re-enabling Send (it
            # stays disabled while any attachment is uploading / transcoding).
            # tiles >= 1 is just a floor so we never send with nothing attached.
            deadline = time.time() + max(int(upload_wait_s or 0), 240)
            steady = 0
            last_log = 0.0
            while time.time() < deadline:
                tiles = self._tile_count(page)
                send_ok = self._send_enabled(page)
                spin = self._any_spinner(page)
                if tiles >= 1 and send_ok and not spin:
                    steady += 1
                    if steady >= 4:                     # ~6 s held steady
                        page.wait_for_timeout(1200)
                        return
                else:
                    steady = 0
                if time.time() - last_log > 30:
                    print(f"[qwen-video] waiting on uploads: tiles={tiles} (want {want}) "
                          f"spinner={spin} send_enabled={send_ok}", file=sys.stderr)
                    last_log = time.time()
                page.wait_for_timeout(1500)

            tiles = self._tile_count(page)
            if tiles == 0:
                raise RuntimeError("Qwen(video): no attachment tiles appeared after "
                                   f"{upload_wait_s}s")
            print(f"[qwen-video] upload-ready ceiling {upload_wait_s}s hit "
                  f"(tiles={tiles}/{want}, send_enabled={self._send_enabled(page)}); "
                  "typing prompt and sending anyway -- _send() will confirm",
                  file=sys.stderr)

        # -- generation start window (video first-token lag) ---------------
        def _wait_generation(self, page):
            """QwenAutomation._wait_generation with the 'has generation started'
            window widened 150 -> 330 s: a video turn's first token can lag far
            behind the Send click while the model ingests the clip."""
            deadline = time.time() + self.timeout_s
            time.sleep(2.0)
            start_deadline = min(deadline, time.time() + 330)
            saw_stop = False
            while time.time() < start_deadline:
                try:
                    if page.locator(self._STOP_SEL).first.is_visible(timeout=500):
                        saw_stop = True
                        break
                except Exception:
                    pass
                try:
                    if len(self._reply_node(page).inner_text(timeout=800) or "") > 60:
                        break
                except Exception:
                    pass
                time.sleep(1.0)
            if saw_stop:
                while time.time() < deadline:
                    try:
                        if not page.locator(self._STOP_SEL).first.is_visible(timeout=500):
                            break
                    except Exception:
                        break
                    time.sleep(1.0)
            self._wait_reply_stable(page, deadline)

        def send_prompt_with_files(self, prompt, file_paths, upload_wait_s=600):
            paths = [str(p) for p in file_paths]
            with self._page() as (ctx, page):
                self._select_model(page)
                self._composer(page)
                self._attach_files(page, paths, upload_wait_s)
                box = self._composer(page)
                self._fill_composer(page, box, prompt)
                page.wait_for_timeout(400)
                self._send(page, box)
                return self._read_reply(page)

    return QwenVideoAutomation


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: _qwen_video_driver.py <v2_dir> <prompt_file> <out_file> "
              "<timeout_s> [file ...]", file=sys.stderr)
        return 2
    v2_dir, prompt_file, out_file, timeout_s = sys.argv[1:5]
    attach = [p for p in sys.argv[5:] if p]

    sys.path.insert(0, v2_dir)
    os.environ.setdefault(
        "BROWSER_PROFILES_DIR", os.path.join(v2_dir, "browser_profiles")
    )

    try:
        cls = _make_cls(v2_dir)
    except Exception as exc:  # noqa: BLE001 - surface the import problem verbatim
        print(f"import browser_llm failed: {exc}", file=sys.stderr)
        return 3

    missing = [p for p in attach if not os.path.isfile(p)]
    if missing:
        print(f"attachment(s) not found: {missing}", file=sys.stderr)
        return 5
    if not attach:
        print("no attachments -- this driver is for video review only", file=sys.stderr)
        return 5

    with open(prompt_file, "r", encoding="utf-8") as fh:
        text = fh.read()

    try:
        bot = cls(headless=False, timeout_s=int(timeout_s),
                  model=os.environ.get("QWEN_MODEL", ""))
        reply = bot.send_prompt_with_files(text, attach)
    except Exception as exc:  # noqa: BLE001
        print(f"Qwen video automation failed: {exc}", file=sys.stderr)
        return 4

    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write(reply or "")
    print(f"OK chars={len(reply or '')} attach={len(attach)} "
          f"model={os.environ.get('QWEN_MODEL', '(default)')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
