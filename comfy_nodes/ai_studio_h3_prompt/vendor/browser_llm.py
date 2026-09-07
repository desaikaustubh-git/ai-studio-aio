# ==============================================================================
# PLAYWRIGHT BROWSER AUTOMATION FOR CHATGPT & GEMINI
# Drives the ChatGPT and Gemini web apps (no API key) to get LLM completions.
# Each provider uses a fixed, already-logged-in persistent Playwright profile
# under the user's home folder — no per-run login flow needed.
# ==============================================================================
import base64
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List
from playwright.sync_api import sync_playwright

# Persistent, already-logged-in Chromium profiles. Default is browser_profiles/<name>
# next to this file; BROWSER_PROFILES_DIR overrides the root so a caller in another
# repo (e.g. AI Studio v2) can keep the profiles with itself.
_PROFILE_ROOT = Path(os.environ.get("BROWSER_PROFILES_DIR")
                     or (Path(__file__).resolve().parent / "browser_profiles"))
PERSISTENT_PROFILES = {
    "chatgpt": _PROFILE_ROOT / "chatgpt",
    "gemini": _PROFILE_ROOT / "gemini",
    # Google Flow (Nano Banana 2). Kept separate from "gemini" on purpose: a Flow
    # image session and a Gemini QA session run concurrently in
    # run_studio_qa_production and can't share one Chromium user-data-dir. If this
    # dir has no login yet, seed it from the Gemini one (same Google account):
    #   Copy-Item browser_profiles/gemini browser_profiles/flow -Recurse
    # or log in fresh:  python browser_llm.py flow-login
    "flow": _PROFILE_ROOT / "flow",
    # Qwen Chat (chat.qwen.ai) - Alibaba's flagship Qwen3.7-Plus, used for the H3
    # video-review node (upload the finished mp4, get a shot-by-shot critique).
    # One-time login:  python browser_llm.py qwen-login
    "qwen": _PROFILE_ROOT / "qwen",
}


# Fetches a blob: URL's bytes from within the page and returns them as base64
# (blob URLs aren't reachable from outside the page's JS context).
_FETCH_BLOB_JS = """async (url) => {
    const resp = await fetch(url);
    const buf = await resp.arrayBuffer();
    let binary = '';
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
}"""


def _poll(pred, seconds: float) -> bool:
    """Call `pred` every 0.5 s until it returns true or `seconds` elapse."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.5)
    return False


class BrowserLLM:
    """Base class: persistent browser context so login sessions survive restarts."""

    provider = "base"
    url = ""
    composer_selector = ""  # CSS selector for the chat input box; only visible once logged in

    def __init__(self, headless: bool = False, timeout_s: int = 120):
        # headless=True triggers ChatGPT/Gemini bot detection (composer never loads / page
        # closes mid-interaction) even with a valid logged-in profile — must run visibly.
        self.headless = headless
        self.timeout_s = timeout_s
        self.profile_path = PERSISTENT_PROFILES[self.provider]
        # Per-provider profile override, e.g. QWEN_PROFILE=qwen2 to run a second
        # logged-in account (rate-limit failover). Value is a sibling dir name under
        # the profile root; one-time login:  <PROVIDER>_PROFILE=<name> python browser_llm.py <provider>-login
        _override = os.environ.get(f"{self.provider.upper()}_PROFILE")
        if _override:
            self.profile_path = _PROFILE_ROOT / _override

    def _launch(self, p, headless: bool):
        return p.chromium.launch_persistent_context(user_data_dir=str(self.profile_path), headless=headless)

    @contextmanager
    def _page(self):
        with sync_playwright() as p:
            ctx = self._launch(p, self.headless)
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(self.url, wait_until="domcontentloaded")
            try:
                yield ctx, page
            finally:
                ctx.close()

    def send_prompt(self, prompt: str) -> str:
        with self._page() as (ctx, page):
            return self._ask(page, prompt)

    def _ask(self, page, prompt: str) -> str:
        raise NotImplementedError

    def login(self, wait_s: int = 300) -> bool:
        """Opens a real, visible browser window so you can log in by hand. Doesn't block on
        terminal input (this may run from a non-interactive launcher) — instead it polls the
        page itself for the chat composer to appear, which only happens once you're actually
        logged in. Once detected, the session is saved to the persistent profile for future
        headless runs. Returns True if login was confirmed within wait_s seconds."""
        with sync_playwright() as p:
            ctx = self._launch(p, headless=False)
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(self.url, wait_until="domcontentloaded")
            print(f"[{self.provider}] Browser window opened — log in there. Waiting up to {wait_s}s...")
            ok = False
            try:
                page.locator(self.composer_selector).first.wait_for(state="visible", timeout=wait_s * 1000)
                print(f"[{self.provider}] Login detected — session saved to {self.profile_path}")
                ok = True
            except Exception:
                print(f"[{self.provider}] Timed out after {wait_s}s without detecting login. "
                      f"Whatever session state exists was still saved to {self.profile_path} — "
                      f"rerun login() if it didn't complete.")
            ctx.close()
            return ok

    @staticmethod
    def _save_image_src(ctx, page, src: str, out_path: str) -> str:
        """Downloads an <img> src (data:, blob:, or http(s):) to out_path."""
        if src.startswith("data:"):
            _, b64data = src.split(",", 1)
            Path(out_path).write_bytes(base64.b64decode(b64data))
        elif src.startswith("blob:"):
            b64data = page.evaluate(_FETCH_BLOB_JS, src)
            Path(out_path).write_bytes(base64.b64decode(b64data))
        else:
            resp = ctx.request.get(src)
            Path(out_path).write_bytes(resp.body())
        return out_path


class ChatGPTAutomation(BrowserLLM):
    provider = "chatgpt"
    url = "https://chatgpt.com/"
    composer_selector = "#prompt-textarea"

    def _fill_composer(self, page, box, text: str) -> None:
        """Put `text` into the composer and confirm it actually landed.

        The composer is a ProseMirror contenteditable. Locator.fill() is fine for
        short messages but times out on the ~30 KB rule-file prompt the H3 writer
        sends as its first turn - the editor re-renders on every large mutation
        and never satisfies Playwright's "editable & stable" actionability check
        (it dies at the default 30 s with "waiting for element to be visible,
        enabled and editable"). For anything big, drop it in with one CDP
        keyboard.insert_text() call; fall back to a clipboard paste, then to a
        fill() with a real timeout.
        """
        box.click()
        if len(text) < 2000:
            box.fill(text, timeout=self.timeout_s * 1000)
            return

        def landed() -> bool:
            try:
                return len(box.inner_text() or "") >= len(text) * 0.8
            except Exception:
                return False

        page.keyboard.insert_text(text)
        if _poll(landed, 15):
            return

        try:
            page.context.grant_permissions(
                ["clipboard-read", "clipboard-write"], origin=self.url.rstrip("/")
            )
            box.click()
            page.evaluate("navigator.clipboard.writeText(arguments[0])", text)
            box.press("Meta+V" if sys.platform == "darwin" else "Control+V")
            if _poll(landed, 20):
                return
        except Exception:
            pass

        box.fill(text, timeout=self.timeout_s * 1000)

    def _ask(self, page, prompt: str) -> str:
        box = page.locator("#prompt-textarea")
        box.wait_for(state="visible", timeout=self.timeout_s * 1000)
        self._fill_composer(page, box, prompt)
        page.wait_for_timeout(300)  # let ProseMirror settle after a big insert
        page.keyboard.press("Enter")

        # Wait for generation to finish: the "Stop generating" control disappears.
        stop_btn = page.locator('button[data-testid="stop-button"]')
        deadline = time.time() + self.timeout_s
        time.sleep(1.5)  # let generation start before polling for its end
        while stop_btn.count() > 0 and time.time() < deadline:
            time.sleep(0.5)

        replies = page.locator('div[data-message-author-role="assistant"]')
        return replies.last.inner_text()

    def send_prompt_with_files(self, prompt: str, file_paths: List[str],
                               upload_wait_s: int = 90) -> str:
        """One-shot visual QA for ChatGPT: open a fresh chat, attach local file(s) —
        reference stills, frames sampled from a render, or an .mp4 — ask about them,
        return the reply text, then close the browser. Mirrors
        GeminiAutomation.send_prompt_with_files. The hidden <input type=file> is in
        the DOM even with the attach menu closed, and set_input_files() drives it
        directly; if it isn't present yet, click the attach control first. A video
        upload can take a while to process before Send re-enables, so this polls the
        send button (up to upload_wait_s) rather than sleeping a fixed time."""
        paths = [str(p) for p in file_paths]
        with self._page() as (ctx, page):
            box = page.locator("#prompt-textarea")
            box.wait_for(state="visible", timeout=self.timeout_s * 1000)
            finp = page.locator('input[type="file"]')
            if finp.count() == 0:
                for sel in ('button[aria-label="Attach files"]',
                            'button[aria-label*="Attach" i]',
                            'button[aria-label*="Upload" i]',
                            'button[data-testid="composer-plus-btn"]'):
                    try:
                        page.locator(sel).first.click(timeout=3000)
                        page.wait_for_timeout(600)
                        break
                    except Exception:
                        continue
                finp = page.locator('input[type="file"]')
            finp.first.set_input_files(paths)
            deadline = time.time() + upload_wait_s
            while time.time() < deadline:
                try:
                    sb = page.locator('button[data-testid="send-button"]')
                    if sb.count() and sb.first.is_enabled():
                        break
                except Exception:
                    pass
                page.wait_for_timeout(1000)
            page.wait_for_timeout(2000)  # let the last attachment settle
            self._fill_composer(page, box, prompt)
            page.wait_for_timeout(300)
            page.keyboard.press("Enter")
            stop_btn = page.locator('button[data-testid="stop-button"]')
            gen_deadline = time.time() + self.timeout_s
            time.sleep(1.5)
            while stop_btn.count() > 0 and time.time() < gen_deadline:
                time.sleep(0.5)
            return page.locator('div[data-message-author-role="assistant"]').last.inner_text()


class GeminiAutomation(BrowserLLM):
    provider = "gemini"
    url = "https://gemini.google.com/app"
    composer_selector = "div.ql-editor[contenteditable='true']"

    def _submit(self, page, prompt: str) -> None:
        box = page.locator("div.ql-editor[contenteditable='true']")
        box.wait_for(state="visible", timeout=self.timeout_s * 1000)
        box.click()
        box.fill(prompt)
        page.keyboard.press("Enter")

        # Wait for generation to finish: the "Stop response" control disappears.
        stop_btn = page.locator('button[aria-label="Stop response"]')
        deadline = time.time() + self.timeout_s
        time.sleep(1.5)  # let generation start before polling for its end
        while stop_btn.count() > 0 and time.time() < deadline:
            time.sleep(0.5)

    def _ask(self, page, prompt: str) -> str:
        self._submit(page, prompt)
        reply = page.locator("message-content").last
        # After a big prompt the reply node can be swapped in a beat late; .last can
        # momentarily match nothing. Wait for it before reading (inner_text's own
        # default is only 30 s and returns the wrong turn if the DOM is mid-swap).
        reply.wait_for(state="visible", timeout=self.timeout_s * 1000)
        page.wait_for_timeout(500)
        return reply.inner_text()

    # A bare scene description doesn't reliably put Gemini into image-generation mode
    # (it often just replies with text) — prefix every image prompt with an explicit
    # imperative + the target framing so it renders instead of chatting.
    _IMG_DIRECTIVE = "Generate an image, photorealistic, vertical 9:16 aspect ratio:\n\n"

    def _img_prompt(self, prompt: str) -> str:
        return prompt if prompt.lstrip().lower().startswith("generate an image") else self._IMG_DIRECTIVE + prompt

    def generate_image(self, prompt: str, out_path: str) -> str:
        """One-shot: opens a fresh chat, asks for an image, downloads it, closes the browser.
        See open_session()/ask_image() for a persistent-chat version that keeps context across
        many calls (character/location/prop consistency, multi-step QA) instead of starting a
        brand-new conversation every time."""
        with self._page() as (ctx, page):
            self._submit(page, self._img_prompt(prompt))
            return self._download_last_image(page, out_path)

    @staticmethod
    def _download_last_image(page, out_path: str) -> str:
        """Uses the chat's own "Download full size image" button via Playwright's download
        handling rather than reading the <img> element: the inline chat preview is a
        cropped/fade-masked thumbnail (~2:1 box) regardless of the image's real aspect ratio,
        and its blob: src can't be fetched cross-context via page.evaluate() either
        ("Failed to fetch" — scoped to the chat's own frame/realm). The download button gives
        the untouched original (e.g. a real 1024x559 photo, not the thumbnail crop)."""
        img = page.locator("message-content img").last
        img.wait_for(state="visible", timeout=180_000)
        try:
            download_btn = page.locator('button[aria-label="Download full size image"]').last
            with page.expect_download(timeout=60_000) as dl_info:
                download_btn.click()
            dl_info.value.save_as(out_path)
            return out_path
        except Exception:
            # The download control sometimes never fires an actual download event (or isn't
            # present). Fall back to the inline <img> — a slightly cropped thumbnail is still a
            # usable reference; a hard failure here poisons the whole persistent session.
            src = img.get_attribute("src")
            if not src:
                raise
            return BrowserLLM._save_image_src(page.context, page, src, out_path)

    def _send(self, page, box) -> None:
        """Submits the composer via the explicit Send button rather than Enter — Enter doesn't
        reliably submit once a file attachment is pending (ask_with_images), so every session
        method uses this one consistent path instead of two different submit mechanisms.

        On a FOLLOW-UP message (not the session's first), the Send button briefly disappears and
        gets re-created right after fill() — Angular re-rendering the composer — and a click
        aimed at that moment can land on the about-to-be-detached element and do nothing. This
        went unnoticed for a while because it fails silently: no exception, the button still
        reports enabled/visible a beat later, but the message was never actually sent. Confirmed
        by reproduction: a second, different image prompt sent this way left the composer
        non-empty, added no new message-content block, and re-downloading "the new image"
        actually returned the previous prompt's file byte-for-byte.

        Fixed by waiting for the button to be attached+enabled immediately before clicking (so
        we click the CURRENT element, not one about to be swapped out), and then verifying the
        composer actually emptied — the one unambiguous signal a click really landed — retrying
        the click a few times before giving up."""
        prior_count = page.locator("message-content").count()
        send_btn = page.locator('button[aria-label="Send message"]').first
        sent = False
        for _ in range(6):
            try:
                send_btn.wait_for(state="visible", timeout=8_000)
                send_btn.click()
            except Exception:
                # Send button not shown (Gemini sometimes drops it for a beat after fill()).
                # Enter submits fine here — ask_image()/ask() have no pending file attachment.
                page.keyboard.press("Enter")
            page.wait_for_timeout(500)
            if not box.inner_text().strip():
                sent = True
                break
        if not sent:
            raise RuntimeError("Gemini composer never emptied after clicking Send — message was not sent")
        stop_btn = page.locator('button[aria-label="Stop response"]')
        deadline = time.time() + self.timeout_s
        time.sleep(1.5)
        while stop_btn.count() > 0 and time.time() < deadline:
            time.sleep(0.5)
        while page.locator("message-content").count() <= prior_count and time.time() < deadline:
            time.sleep(0.5)

    def open_session(self) -> "GeminiAutomation":
        """Opens ONE persistent browser context/page for many follow-up calls (ask/ask_image/
        ask_with_images), all landing in the SAME chat thread — so Gemini keeps full
        conversation context (established characters, locations, style) across an entire
        production run instead of starting fresh and forgetting all of it on every call.
        Call close_session() when done."""
        self._pw_cm = sync_playwright()
        self._pw = self._pw_cm.start()
        self._ctx = self._launch(self._pw, self.headless)
        self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self._page.goto(self.url, wait_until="domcontentloaded")
        self._page.locator(self.composer_selector).first.wait_for(
            state="visible", timeout=self.timeout_s * 1000
        )
        return self

    def close_session(self) -> None:
        if getattr(self, "_ctx", None) is not None:
            self._ctx.close()
        if getattr(self, "_pw", None) is not None:
            self._pw.stop()  # .start()'s own return value has .stop(), not the context manager itself
        self._ctx = self._pw = self._pw_cm = self._page = None

    def reset_chat(self) -> "GeminiAutomation":
        """Reload the app to a fresh chat thread within the SAME persistent context. A failed
        image download / send can leave the composer wedged so every later call in the session
        fails too — call this between retries to recover without tearing down the whole browser
        (and losing the login). If the page itself is dead, rebuild the context."""
        try:
            self._page.goto(self.url, wait_until="domcontentloaded")
            self._page.locator(self.composer_selector).first.wait_for(
                state="visible", timeout=self.timeout_s * 1000
            )
            return self
        except Exception:
            try:
                self.close_session()
            except Exception:
                pass
            return self.open_session()

    def ask(self, prompt: str) -> str:
        """Follow-up text message in the open session. Call open_session() first."""
        box = self._page.locator(self.composer_selector)
        box.click()
        box.fill(prompt)
        self._send(self._page, box)
        return self._page.locator("message-content").last.inner_text()

    def ask_image(self, prompt: str, out_path: str) -> str:
        """Follow-up image-generation message in the open session — see generate_image()."""
        box = self._page.locator(self.composer_selector)
        box.click()
        box.fill(self._img_prompt(prompt))
        self._send(self._page, box)
        return self._download_last_image(self._page, out_path)

    def ask_with_images(self, prompt: str, image_paths: List[str]) -> str:
        """Attaches local image file(s) to the next message and asks about them — used for
        visual QA, where Gemini has to actually SEE a reference image or rendered frame to
        judge it, not just read a text description of it."""
        page = self._page
        page.locator('button[aria-label="Upload & tools"]').first.click()
        page.wait_for_timeout(500)
        page.locator('input[type="file"]').first.set_input_files(image_paths)
        page.wait_for_timeout(2500)
        box = page.locator(self.composer_selector)
        box.click()
        box.fill(prompt)
        self._send(page, box)
        return page.locator("message-content").last.inner_text()

    def ask_image_with_refs(self, prompt: str, ref_paths: List[str], out_path: str) -> str:
        """Attach local reference image(s) then ask Gemini to GENERATE a NEW image that
        keeps their identity / wardrobe / geometry -- the multi-reference-image
        consistency technique (character plate + environment plate + prop -> one
        composed shot frame). Persistent-session form: call open_session() first.
        Returns the saved path. Falls back to a plain text->image ask_image() if the
        upload control can't be driven."""
        page = self._page
        try:
            page.locator('button[aria-label="Upload & tools"]').first.click()
            page.wait_for_timeout(500)
            page.locator('input[type="file"]').first.set_input_files(
                [str(p) for p in ref_paths])
            page.wait_for_timeout(3000)  # let every thumbnail finish before the prompt
        except Exception:
            pass  # no refs attached -> still worth a text-only render
        box = page.locator(self.composer_selector)
        box.click()
        box.fill(self._img_prompt(prompt))
        self._send(page, box)
        return self._download_last_image(self._page, out_path)

    def send_prompt_with_files(self, prompt: str, file_paths: List[str],
                               upload_wait_s: int = 40) -> str:
        """One-shot visual QA: open a fresh chat, attach local file(s) — reference stills,
        frames sampled from a render, or an .mp4 — ask about them, return the reply text,
        then close the browser. The persistent-session form is open_session() +
        ask_with_images(). A video attachment can take a while to upload and be processed
        before Gemini re-enables Send, so this waits for the Send button (polling up to
        upload_wait_s) instead of a fixed sleep."""
        paths = [str(p) for p in file_paths]
        with self._page() as (ctx, page):
            page.locator('button[aria-label="Upload & tools"]').first.click()
            page.wait_for_timeout(500)
            page.locator('input[type="file"]').first.set_input_files(paths)
            deadline = time.time() + upload_wait_s
            while time.time() < deadline:
                try:
                    if page.locator('button[aria-label="Send message"]').first.is_enabled():
                        break
                except Exception:
                    pass
                page.wait_for_timeout(1000)
            page.wait_for_timeout(1500)  # let the last attachment settle
            box = page.locator(self.composer_selector)
            box.click()
            box.fill(prompt)
            self._send(page, box)
            reply = page.locator("message-content").last
            reply.wait_for(state="visible", timeout=self.timeout_s * 1000)
            page.wait_for_timeout(500)
            return reply.inner_text()


class QwenAutomation(BrowserLLM):
    """Drives the Qwen Chat web app (chat.qwen.ai / "Qwen Studio") - no API key,
    same persistent-profile approach as the others. Used by the H3 video-review
    node: open a fresh chat, upload the finished .mp4 (or sampled frames), run a
    brutal shot-by-shot critique prompt, return the text.

    Qwen's DOM is not a public/stable API. Selectors below are best-effort for the
    late-2025 / 2026 layout and every optional step (model switch, attach-button
    click) is wrapped in try/except - a control that's missing because the page is
    already in the right state is not fatal. If review starts failing, watch it
    live and adjust:  python browser_llm.py qwen "say OK"
    """

    provider = "qwen"
    url = "https://chat.qwen.ai/"
    # a <textarea> is the composer once the app has loaded (logged in or not).
    composer_selector = "textarea"
    # exact text shown in Qwen's model picker; overridable via QWEN_MODEL env var
    # or the constructor. "Qwen3.7-Plus" is the site default as of 2026-08.
    model_label = "Qwen3.7-Plus"

    def __init__(self, headless: bool = False, timeout_s: int = 600,
                 model: str = ""):
        # Video understanding is slower than a text turn - default the timeout up.
        super().__init__(headless=headless, timeout_s=timeout_s)
        self.model_label = (model or os.environ.get("QWEN_MODEL")
                            or type(self).model_label).strip()

    # Google's OAuth screen refuses a browser it detects as automated ("This
    # browser or app may not be secure"). Use the real installed Chrome, drop the
    # automation switches, and mask navigator.webdriver so "Continue with Google"
    # has a chance; e-mail / phone login on chat.qwen.ai works regardless.
    _STEALTH_JS = (
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        "window.chrome=window.chrome||{runtime:{}};"
        "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});"
        "Object.defineProperty(navigator,'languages',{get:()=>['en-US','en']});"
    )

    def _launch(self, p, headless: bool):
        # chat.qwen.ai's attachment uploader (OSS thumbnailing) is flaky under the
        # bundled Chrome-for-Testing build with software GL -- video thumbnails never
        # flip to their https src and the upload "never finishes". The real installed
        # Chrome (hardware media + normal fingerprint) uploads fine, so insist on it
        # and say so; only fall back to bundled Chromium if system Chrome is truly
        # absent, and make that loud instead of silent.
        try:
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_path),
                channel="chrome",
                headless=headless,
                args=["--disable-blink-features=AutomationControlled",
                      "--no-first-run", "--no-default-browser-check"],
                ignore_default_args=["--enable-automation"],
            )
            print("[qwen] launched system Chrome (channel=chrome)", file=sys.stderr)
        except Exception as exc:
            print(f"[qwen] system Chrome launch failed ({type(exc).__name__}: "
                  f"{str(exc)[:200]}); falling back to bundled Chromium -- attachment "
                  "uploads may stall", file=sys.stderr)
            ctx = super()._launch(p, headless)
        try:
            ctx.add_init_script(self._STEALTH_JS)
        except Exception:
            pass
        return ctx

    # -- helpers ------------------------------------------------------------
    def _composer(self, page):
        for sel in ("textarea[placeholder]", "textarea#chat-input",
                    "div.ProseMirror[contenteditable='true']",
                    "div[contenteditable='true']", "textarea"):
            loc = page.locator(sel).first
            try:
                loc.wait_for(state="visible", timeout=4000)
                return loc
            except Exception:
                continue
        loc = page.locator("textarea").first
        loc.wait_for(state="visible", timeout=self.timeout_s * 1000)
        return loc

    def _select_model(self, page) -> None:
        """Best-effort: make sure `self.model_label` is the active model. Any step
        that isn't found (already selected, layout differs) is skipped."""
        label = (self.model_label or "").strip()
        if not label:
            return
        try:
            switcher = page.locator(
                "button:has-text('Qwen'), [class*='model-selector'] button, "
                "[data-testid*='model'] button, [class*='ModelSelector'] button"
            ).first
            switcher.wait_for(state="visible", timeout=4000)
            if label.lower() in (switcher.inner_text(timeout=2000) or "").lower():
                return  # already active
            switcher.click(timeout=3000)
            page.wait_for_timeout(700)
        except Exception:
            return
        for opt_sel in (
            f"[role='menuitem']:has-text('{label}')",
            f"[role='option']:has-text('{label}')",
            f"li:has-text('{label}')",
            f"div[role='button']:has-text('{label}')",
            f"button:has-text('{label}')",
        ):
            try:
                page.locator(opt_sel).first.click(timeout=2500)
                page.wait_for_timeout(500)
                return
            except Exception:
                continue
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass

    def _fill_composer(self, page, box, text: str) -> None:
        box.click()
        try:
            box.fill(text, timeout=self.timeout_s * 1000)
            return
        except Exception:
            pass
        page.keyboard.insert_text(text)

    def _send(self, page, box) -> None:
        """Submit the composer, then wait for generation to finish. Enter submits
        a plain turn; with a file still attached Qwen sometimes ignores Enter, so
        try an explicit Send button first and fall back to Enter, then confirm the
        composer actually emptied (the one unambiguous "it sent" signal)."""
        sent = False
        for _ in range(6):
            for sel in ("button[aria-label*='send' i]",
                        "button[data-testid*='send' i]",
                        "button:has(svg[class*='send' i])",
                        "button[type='submit']"):
                try:
                    b = page.locator(sel).first
                    if b.is_visible(timeout=1000) and b.is_enabled():
                        b.click(timeout=2000)
                        raise StopIteration
                except StopIteration:
                    break
                except Exception:
                    continue
            else:
                try:
                    page.keyboard.press("Enter")
                except Exception:
                    pass
            page.wait_for_timeout(600)
            try:
                if not (box.input_value(timeout=1000) or "").strip():
                    sent = True
                    break
            except Exception:
                try:
                    if not (box.inner_text(timeout=1000) or "").strip():
                        sent = True
                        break
                except Exception:
                    pass
        if not sent:
            raise RuntimeError("Qwen composer never emptied after Send - message not sent")
        self._wait_generation(page)

    _STOP_SEL = ("button[aria-label*='stop' i], button[aria-label*='pause' i], "
                 "button[data-testid*='stop' i], button:has(svg[class*='stop' i])")

    def _wait_generation(self, page) -> None:
        """Wait for a complete response. A video upload delays the *start* of
        generation well past the send click, so: (1) give the stop/streaming
        control up to 150 s to appear (or the reply to gain real text),
        (2) wait while that control stays visible = actively streaming,
        (3) confirm the text has settled."""
        deadline = time.time() + self.timeout_s
        time.sleep(2.0)

        start_deadline = min(deadline, time.time() + 150)
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

    def _reply_node(self, page):
        """The locator with the most text among likely assistant-message
        containers. Qwen nests several markdown divs and re-renders them during
        streaming, so 'first visible' / '.last' can latch onto a stale or empty
        wrapper - pick the longest instead."""
        best, best_len = None, -1
        for sel in ("div[class*='assistant'] div[class*='markdown']",
                    "div[class*='markdown-body']",
                    "div[class*='markdown']",
                    "div[class*='assistant']",
                    "[data-role='assistant']",
                    "div[class*='message'][class*='bot']",
                    "div[class*='response']"):
            loc = page.locator(sel).last
            try:
                if loc.count():
                    n = len(loc.inner_text(timeout=800) or "")
                    if n > best_len:
                        best, best_len = loc, n
            except Exception:
                continue
        if best is not None and best_len > 0:
            return best
        return page.locator("main").last

    def _wait_reply_stable(self, page, deadline: float) -> None:
        """Poll the reply text until it stops growing for ~9 s AND the streaming
        control is gone. Video understanding streams in multi-second bursts, so a
        short stability window returns a truncated reply."""
        last, stable = -1, 0
        need = 6  # consecutive unchanged 1.5 s samples
        while time.time() < deadline:
            try:
                n = len(self._reply_node(page).inner_text(timeout=2500) or "")
            except Exception:
                n = last
            streaming = False
            try:
                streaming = page.locator(self._STOP_SEL).first.is_visible(timeout=400)
            except Exception:
                streaming = False
            if n > 60 and n == last and not streaming:
                stable += 1
                if stable >= need:
                    return
            else:
                stable = 0
            last = n
            time.sleep(1.5)

    def _read_reply(self, page) -> str:
        best = ""
        for _ in range(3):
            try:
                t = self._reply_node(page).inner_text(timeout=5000) or ""
            except Exception:
                t = ""
            if len(t) > len(best):
                best = t
            if best:
                break
            time.sleep(1.0)
        return best

    # -- public interface -------------------------------------------------
    def _ask(self, page, prompt: str) -> str:
        self._select_model(page)
        box = self._composer(page)
        self._fill_composer(page, box, prompt)
        page.wait_for_timeout(300)
        self._send(page, box)
        return self._read_reply(page)

    # dropdown item that opens the OS file picker, under the '+' mode menu.
    _UPLOAD_ITEM_SEL = (
        ".mode-select-dropdown-item:has-text('Upload'), "
        "[class*='dropdown-menu-item']:has-text('Upload attachment'), "
        "[class*='dropdown-item']:has-text('Upload'), "
        "[role='menuitem']:has-text('Upload')"
    )

    def _uploaded_count(self, page) -> int:
        """How many attachment thumbnails have finished uploading. Images render as
        <img class='vision-item-image' src=...>; videos as
        <video class='vision-item-video'><source src=...></video>. In both cases the
        src is a local/placeholder value while uploading and flips to an https OSS
        URL once the server has the file."""
        try:
            return page.evaluate(
                "() => {"
                "  const ok = s => (s || '').startsWith('http');"
                "  let n = 0;"
                "  document.querySelectorAll('img.vision-item-image')"
                "    .forEach(i => { if (ok(i.getAttribute('src'))) n++; });"
                "  document.querySelectorAll('video.vision-item-video').forEach(v => {"
                "    const src = v.getAttribute('src')"
                "      || (v.querySelector('source') && v.querySelector('source').getAttribute('src'));"
                "    if (ok(src)) n++;"
                "  });"
                "  return n;"
                "}"
            )
        except Exception:
            return 0

    def _attach_files(self, page, paths: List[str], upload_wait_s: int) -> None:
        """Attach local file(s) to the Qwen composer and block until every one has
        uploaded. Qwen's real attach path is: click the '+'
        (div[aria-label='Select Mode']) -> click 'Upload attachment' in the mode
        menu -> a native OS file chooser opens -> set_files. The
        <input type=file id=filesUpload> that already sits in the DOM is inert
        until that menu mounts, so driving it directly attaches nothing silently.
        Raises RuntimeError if the picker won't open or the uploads don't land, so
        the caller fails loudly instead of sending a prompt with no images."""
        want = len(paths)
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
            # preferred: the menu item raises a native file chooser
            try:
                item = page.locator(self._UPLOAD_ITEM_SEL).first
                with page.expect_file_chooser(timeout=6000) as fc:
                    item.click(timeout=3000)
                fc.value.set_files(paths, timeout=60_000)
                opened = True
                break
            except Exception:
                pass
            # fallback: the menu is open now, so #filesUpload is finally mounted
            try:
                fi = page.locator("input#filesUpload, .mode-select input[type='file'], "
                                  "input[type='file']").first
                if fi.count():
                    fi.set_input_files(paths, timeout=60_000)
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
            raise RuntimeError("Qwen: could not open the upload picker "
                               "('+' / Select Mode -> 'Upload attachment')")

        deadline = time.time() + upload_wait_s
        while time.time() < deadline:
            if self._uploaded_count(page) >= want:
                break
            page.wait_for_timeout(2000)
        got = self._uploaded_count(page)
        if got < want:
            raise RuntimeError(f"Qwen: only {got}/{want} attachment(s) finished "
                               f"uploading after {upload_wait_s}s")
        page.wait_for_timeout(1500)  # let the composer settle before typing

    def send_prompt_with_files(self, prompt: str, file_paths: List[str],
                               upload_wait_s: int = 240) -> str:
        """One-shot visual QA: open a fresh chat, attach local file(s) - reference
        stills, frames sampled from a render, or an .mp4 - ask about them, return
        the reply text, then close the browser. Attaching goes through the '+'
        mode menu ('Upload attachment') and waits for every thumbnail to finish
        uploading (up to upload_wait_s) before the prompt is typed."""
        paths = [str(p) for p in file_paths]
        with self._page() as (ctx, page):
            self._select_model(page)
            self._composer(page)
            self._attach_files(page, paths, upload_wait_s)
            box = self._composer(page)
            self._fill_composer(page, box, prompt)
            page.wait_for_timeout(300)
            self._send(page, box)
            return self._read_reply(page)

    def login(self, wait_s: int = 420) -> bool:
        with sync_playwright() as p:
            ctx = self._launch(p, headless=False)
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(self.url, wait_until="domcontentloaded")
            print(f"[qwen] Log in at chat.qwen.ai in the window. Waiting up to {wait_s}s...")
            ok = False
            deadline = time.time() + wait_s
            auth_cta = ("button:has-text('Log in'), a:has-text('Log in'), "
                        "button:has-text('Sign up'), a:has-text('Sign up'), "
                        "button:has-text('登录'), a:has-text('登录')")
            while time.time() < deadline:
                try:
                    has_box = page.locator("textarea").count() > 0
                    cta_visible = False
                    try:
                        cta_visible = page.locator(auth_cta).first.is_visible(timeout=800)
                    except Exception:
                        cta_visible = False
                    if has_box and not cta_visible:
                        ok = True
                        break
                except Exception:
                    pass
                time.sleep(2)
            print(f"[qwen] {'Login detected' if ok else 'Timed out (log in was not '
                  'confirmed)'} - session saved to {self.profile_path}")
            ctx.close()
            return ok


class FlowAutomation(BrowserLLM):
    """Drives Google Flow (labs.google/fx/tools/flow) to generate images with the
    "Nano Banana 2" image model (a.k.a. Nano Banana Pro / Gemini 3 Pro Image).

    Same no-API-key, persistent-profile approach as GeminiAutomation. Exposes the
    subset of the GeminiAutomation image interface the pipeline actually calls:
    generate_image() (one-shot) and open_session()/ask_image()/close_session()
    (persistent project, many images). Chat methods (ask/ask_with_images) are NOT
    provided — Flow can't do Q&A, so QA stays on GeminiAutomation.

    Flow's DOM is not a public/stable API. The SEL selectors below are best-effort
    for the late-2025 Flow layout and every optional step is wrapped in try/except
    (a control that's missing because it's already in the right state is not fatal).
    If image generation starts failing, watch it live and adjust SEL:
        python browser_llm.py flow "a single red apple on a plain white table"
    """

    provider = "flow"
    url = "https://labs.google/fx/tools/flow"
    composer_selector = "textarea"
    model_label = "Nano Banana 2"  # exact text shown in Flow's image-model picker

    SEL = {
        "new_project": "button:has-text('New project'), a:has-text('New project'), "
                       "button:has-text('New Project')",
        "prompt": "textarea",
        "settings_btn": "button[aria-label*='ettings' i], button[aria-label*='odel' i], "
                        "button[aria-label*='Tune' i]",
        "generate_btn": "button[aria-label*='reate' i], button[aria-label*='enerate' i], "
                        "button[aria-label*='ubmit' i], button[type='submit']",
        "result_img": "img[src*='googleusercontent'], img[src*='blob:'], img[src*='lh3.google']",
        "download_btn": "button[aria-label*='ownload' i], [role='menuitem']:has-text('Download'), "
                        "a[download]",
        "kebab_btn": "button[aria-label*='ore options' i], button[aria-label*='More' i]",
    }

    def __init__(self, headless: bool = False, timeout_s: int = 240):
        # Flow image gen is slower than a Gemini chat turn — give it a longer default.
        super().__init__(headless=headless, timeout_s=timeout_s)

    # -- shared navigation -----------------------------------------------------
    def _dismiss(self, page) -> None:
        for label in ("Got it", "No thanks", "Dismiss", "Accept all", "I agree",
                      "Continue", "Skip", "Close"):
            try:
                b = page.get_by_role("button", name=label)
                if b.count():
                    b.first.click(timeout=1200)
            except Exception:
                pass

    def _open_project(self, page) -> None:
        page.wait_for_load_state("domcontentloaded")
        self._dismiss(page)
        try:  # on the project list -> start one; already inside a project -> skip
            btn = page.locator(self.SEL["new_project"]).first
            btn.wait_for(state="visible", timeout=8000)
            btn.click()
        except Exception:
            pass
        page.locator(self.SEL["prompt"]).last.wait_for(
            state="visible", timeout=self.timeout_s * 1000)
        self._dismiss(page)

    def _select_model(self, page) -> None:
        """Best-effort: open the model/settings panel and pick Nano Banana 2. Any
        step that isn't found (panel already set, layout differs) is skipped."""
        try:
            page.locator(self.SEL["settings_btn"]).first.click(timeout=3000)
            page.wait_for_timeout(600)
        except Exception:
            pass
        for name in (self.model_label, "Nano Banana Pro", "Nano Banana"):
            try:
                opt = page.get_by_text(name, exact=False).first
                opt.wait_for(state="visible", timeout=2500)
                opt.click()
                page.wait_for_timeout(400)
                break
            except Exception:
                continue
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass

    def _submit(self, page, prompt: str) -> int:
        box = page.locator(self.SEL["prompt"]).last
        box.click()
        box.fill(prompt)
        before = page.locator(self.SEL["result_img"]).count()
        try:
            gen = page.locator(self.SEL["generate_btn"]).first
            if gen.count():
                gen.click(timeout=3000)
                return before
        except Exception:
            pass
        page.keyboard.press("Enter")
        return before

    def _wait_generation(self, page, before: int) -> None:
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if page.locator(self.SEL["result_img"]).count() > before:
                page.wait_for_timeout(1800)  # let the high-res swap in
                return
            page.wait_for_timeout(1000)

    def _download_last_image(self, ctx, page, out_path: str) -> str:
        img = page.locator(self.SEL["result_img"]).last
        img.wait_for(state="visible", timeout=self.timeout_s * 1000)
        try:  # explicit download control keeps full resolution
            try:
                page.locator(self.SEL["kebab_btn"]).last.click(timeout=2000)
                page.wait_for_timeout(400)
            except Exception:
                pass
            with page.expect_download(timeout=15000) as dl:
                page.locator(self.SEL["download_btn"]).last.click(timeout=5000)
            dl.value.save_as(out_path)
            return out_path
        except Exception:
            pass
        src = img.get_attribute("src")
        if not src:
            raise RuntimeError("Flow: generated image element has no src to fall back to")
        return self._save_image_src(ctx, page, src, out_path)

    # -- public interface (mirrors GeminiAutomation's image methods) ----------
    def generate_image(self, prompt: str, out_path: str) -> str:
        with self._page() as (ctx, page):
            self._open_project(page)
            self._select_model(page)
            before = self._submit(page, prompt)
            self._wait_generation(page, before)
            return self._download_last_image(ctx, page, out_path)

    def open_session(self) -> "FlowAutomation":
        self._pw_cm = sync_playwright()
        self._pw = self._pw_cm.start()
        self._ctx = self._launch(self._pw, self.headless)
        self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self._page.goto(self.url, wait_until="domcontentloaded")
        self._open_project(self._page)
        self._select_model(self._page)
        return self

    def close_session(self) -> None:
        if getattr(self, "_ctx", None) is not None:
            self._ctx.close()
        if getattr(self, "_pw", None) is not None:
            self._pw.stop()
        self._ctx = self._pw = self._pw_cm = self._page = None

    def ask_image(self, prompt: str, out_path: str) -> str:
        """Generate another image in the already-open Flow project. Call open_session() first."""
        before = self._submit(self._page, prompt)
        self._wait_generation(self._page, before)
        return self._download_last_image(self._ctx, self._page, out_path)

    def login(self, wait_s: int = 300) -> bool:
        with sync_playwright() as p:
            ctx = self._launch(p, headless=False)
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(self.url, wait_until="domcontentloaded")
            print(f"[flow] Log in with your Google account (needs AI Pro/Ultra). "
                  f"Waiting up to {wait_s}s...")
            ok = False
            try:
                page.locator(f"{self.SEL['new_project']}, {self.SEL['prompt']}").first.wait_for(
                    state="visible", timeout=wait_s * 1000)
                ok = True
                print(f"[flow] Login detected — session saved to {self.profile_path}")
            except Exception:
                print(f"[flow] Timed out after {wait_s}s. Whatever session state exists "
                      f"was still saved to {self.profile_path} — rerun if needed.")
            ctx.close()
            return ok


def get_provider(name: str, headless: bool = False) -> BrowserLLM:
    providers = {
        "chatgpt": ChatGPTAutomation,
        "gemini": GeminiAutomation,
        "flow": FlowAutomation,
        "qwen": QwenAutomation,
    }
    if name not in providers:
        raise ValueError(f"Unknown browser LLM provider: {name}")
    return providers[name](headless=headless)


if __name__ == "__main__":
    # Manual helpers (all use the persistent profiles under browser_profiles/):
    #   python browser_llm.py <chatgpt|gemini|flow|qwen>-login  # window to log in
    #   python browser_llm.py flow ["image prompt"]         # test a Nano Banana 2 render
    #   python browser_llm.py qwen ["prompt"] [file ...]    # test a Qwen turn / upload
    #   python browser_llm.py <chatgpt|gemini>              # verify a chat completion
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "chatgpt"
    if arg.endswith("-login"):
        get_provider(arg[:-len("-login")]).login()
        sys.exit(0)
    if arg == "flow":
        prompt = sys.argv[2] if len(sys.argv) > 2 else (
            "a single red apple on a plain white table, product photo, soft daylight")
        out = FlowAutomation(headless=False).generate_image(prompt, "flow_test.png")
        print(f"[flow] saved -> {out}")
        sys.exit(0)
    if arg == "qwen":
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Reply with just the word OK."
        files = [p for p in sys.argv[3:] if p]
        bot = QwenAutomation(headless=False)
        reply = (bot.send_prompt_with_files(prompt, files) if files
                 else bot.send_prompt(prompt))
        print(f"[qwen] reply ({len(reply)} chars):\n{reply[:2000]}")
        sys.exit(0)
    bot = get_provider(arg, headless=False)
    reply = bot.send_prompt("Reply with just the word OK.")
    print(f"[{arg}] verification reply: {reply[:200]!r}")
