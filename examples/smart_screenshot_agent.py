"""
Smart Screenshot Agent
----------------------



urls = [
   {"link": "https://veeton.com/", "name": "veeton"}, # le screen est niqué
   {"link": "https://nr2.io/", "name": "nr2"}, # le screen contient juste le header de la page
   {"link": "https://start.paa.ge/", "name":"startpaage"}, # certains images n'apparaissent pas
   {"link": "https://hcompany.ai/", "name":"hcompany"}, # site trop dynamique
   {"link": "https://rivrs.io/", "name":"rivrs"}, # images pas chargées
   {"link": "https://harmonic.ai/", "name":"harmonic"}, # images étirées
   {"link": "https://dealroom.co/", "name":"dealroom"}, # pop up de cookie qui s'enlève pas
   {"link": "https://pitchandrise.co/", "name":"pitchandrise"}, # le scroll ne marche pas
]


Captures a sequence of screenshots while scrolling through a page so that
dynamically loaded content is fully included. Works in two modes:

1. Agent mode (requires `BROWSER_USE_API_KEY`) – lets the Browser-Use agent reason
   about readiness before we take screenshots.
2. Direct local mode (default) – launches a local Chromium session and performs
   deterministic navigation + scrolling without any LLM.

Usage:
    python examples/smart_screenshot_agent.py
    python examples/smart_screenshot_agent.py https://github.com

Screenshots are saved inside `./screenshots/` as multiple files:
    20251124_103041_example.com_part_01.png
    20251124_103041_example.com_part_02.png
    ...
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

from browser_use import Agent, Browser, ChatBrowserUse, ChatOpenAI
from browser_use.browser.session import BrowserSession

load_dotenv()


def sanitize_filename(url: str) -> str:
    """Convert URL to a safe filename fragment."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path = parsed.path.strip("/").replace("/", "_")
    filename = f"{domain}_{path}" if path else domain
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filename = re.sub(r"_+", "_", filename)
    filename = filename.strip("_")[:120]
    return filename or "screenshot"


async def smart_screenshot(
    url: str,
    output_dir: str = "screenshots",
    max_segments: int = 20,
    viewport_pause: float = 2.5,
    scroll_overlap: int = 180,
) -> list[str]:
    """
    Capture multiple screenshots while scrolling through the page.

    Args:
        url: Target page.
        output_dir: Directory to save screenshots.
        max_segments: Maximum number of scroll segments to capture.
        viewport_pause: Delay (seconds) after each scroll to let dynamic content load.
        scroll_overlap: Number of pixels to overlap between consecutive shots.

    Returns:
        List of saved screenshot paths.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = sanitize_filename(url)
    base_name = f"{timestamp}_{safe_name}"

    print("📸 Smart Screenshot Agent")
    print(f"   URL: {url}")
    print(f"   Output directory: {output_path}")
    print()

    # Check for any API key (LiteLLM or Browser-Use)
    api_key_present = bool(os.getenv("OPENAI_API_KEY") or os.getenv("BROWSER_USE_API_KEY"))
    if api_key_present:
        api_key_type = "OPENAI_API_KEY (LiteLLM)" if os.getenv("OPENAI_API_KEY") else "BROWSER_USE_API_KEY"
        print(f"🤖 Detected {api_key_type} – using agent-assisted mode.")
        return await _smart_screenshot_agent_mode(
            url=url,
            base_name=base_name,
            output_dir=output_path,
            max_segments=max_segments,
            viewport_pause=viewport_pause,
            scroll_overlap=scroll_overlap,
        )

    print("⚙️  No API key found (OPENAI_API_KEY or BROWSER_USE_API_KEY) – running direct local mode (no LLM).")
    return await _smart_screenshot_direct_mode(
        url=url,
        base_name=base_name,
        output_dir=output_path,
        max_segments=max_segments,
        viewport_pause=viewport_pause,
        scroll_overlap=scroll_overlap,
    )


def _create_browser() -> BrowserSession:
    """Create a Browser session configured for visible local execution."""
    return Browser(
        headless=False,
        window_size={"width": 1920, "height": 1080},
        device_scale_factor=2.0,  # Capture sharper images on HiDPI output
        keep_alive=True,  # Keep browser open until explicitly closed
    )


async def _smart_screenshot_direct_mode(
    url: str,
    base_name: str,
    output_dir: Path,
    max_segments: int,
    viewport_pause: float,
    scroll_overlap: int,
) -> list[str]:
    browser = _create_browser()
    await browser.start()

    try:
        print("🚀 Launching local browser...")
        await browser.navigate_to(url)
        print("⏳ Waiting 3s for initial content to load...")
        await asyncio.sleep(3.0)
        # Scroll a bit to trigger any lazy-loaded banners
        await _scroll_to(browser, 100)
        await asyncio.sleep(1.0)
        await _scroll_to(browser, 0)
        print("⏳ Waiting 5s for cookie banners to appear...")
        await asyncio.sleep(5.0)
        print("🔍 Checking for cookie banners...")
        await _accept_cookies_if_present(browser)
        # One more check after a short delay in case banner appeared late
        await asyncio.sleep(2.0)
        await _accept_cookies_if_present(browser)
        await browser.get_browser_state_summary(include_screenshot=False)

        return await _capture_scrolling_screenshots(
            browser_session=browser,
            base_name=base_name,
            output_dir=output_dir,
            max_segments=max_segments,
            viewport_pause=viewport_pause,
            scroll_overlap=scroll_overlap,
        )
    finally:
        await browser.kill()


async def _smart_screenshot_agent_mode(
    url: str,
    base_name: str,
    output_dir: Path,
    max_segments: int,
    viewport_pause: float,
    scroll_overlap: int,
) -> list[str]:
    browser = _create_browser()

    # Ultra-simple task: ONLY click "Accept" button, then STOP immediately
    task = f"""
    Navigate to {url}.
    Find and click ONLY the "Accept" or "Accept All" button for cookies.
    DO NOT click "Manage", "Preferences", "Settings", or any other button.
    After clicking "Accept", immediately signal completion with done action.
    Do nothing else after clicking Accept.
    """

    # Configure LLM - use LiteLLM if available, otherwise ChatBrowserUse
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(
            model="gemini-2.5-flash-lite-preview-09-2025",
            timeout=httpx.Timeout(180.0, connect=60.0, read=180.0, write=30.0),
            max_retries=2,
            # Gemini compatibility options via LiteLLM
            add_schema_to_system_prompt=False,
            remove_min_items_from_schema=True,
            remove_defaults_from_schema=True,
            dont_force_structured_output=True,
        )
    else:
        llm = ChatBrowserUse()

    agent = Agent(
        task=task.strip(),
        llm=llm,
        browser=browser,
        max_steps=2,  # Very strict: navigate, click Accept (then stop immediately)
        flash_mode=True,
        use_thinking=False,
    )

    browser_session: BrowserSession | None = None
    try:
        browser_session = agent.browser_session
        print("🍪 Using LLM to accept cookies only...")
        history = await agent.run()
        
        # Check if agent clicked "Accept" button - look for click actions with "Accept" in the result
        clicked_accept = False
        for step in history.history:
            if hasattr(step, 'action') and step.action:
                action_name = getattr(step.action, 'action_name', '') if hasattr(step.action, 'action_name') else str(type(step.action).__name__)
                if 'click' in action_name.lower():
                    # Check if the click was on an Accept button
                    result = getattr(step, 'result', None)
                    if result and ('Accept' in str(result) or 'accept' in str(result).lower()):
                        clicked_accept = True
                        print(f"   ✅ Detected click on Accept button in step {len(history.history)}.")
                        break
        
        if clicked_accept or history.is_successful():
            print(f"   ✅ Cookies handled ({len(history.history)} steps).")
            await asyncio.sleep(1.0)  # Brief wait for banner to disappear
        else:
            print(f"   ⚠️  LLM may not have clicked Accept ({len(history.history)} steps).")
            print("   🔄 Trying pattern-based method (one attempt only, NO LLM)...")
            await asyncio.sleep(2.0)  # Wait for page to be ready
            # Use pattern-based ONLY (no LLM agent)
            await _accept_cookies_pattern_only(browser_session)
            await asyncio.sleep(1.0)  # Brief wait after click
        
        # Go directly to screenshot capture WITHOUT LLM (naive scroll method)
        print("📸 Starting screenshot capture (no LLM, direct scroll method)...")
        screenshot_paths = await _capture_scrolling_screenshots(
            browser_session=browser_session,
            base_name=base_name,
            output_dir=output_dir,
            max_segments=max_segments,
            viewport_pause=viewport_pause,
            scroll_overlap=scroll_overlap,
        )
        return screenshot_paths
    finally:
        # Close agent and browser
        try:
            await agent.close()
        except Exception as e:
            print(f"⚠️  Error closing agent: {e}")
        try:
            if browser:
                await browser.kill()
        except Exception as e:
            print(f"⚠️  Error closing browser: {e}")


async def _capture_scrolling_screenshots(
    browser_session: BrowserSession,
    base_name: str,
    output_dir: Path,
    max_segments: int,
    viewport_pause: float,
    scroll_overlap: int,
) -> list[str]:
    """Scroll through the page and capture screenshots for each viewport."""
    print("📏 Preparing to scroll and capture...")
    await _scroll_to(browser_session, 0)
    print("⏳ Waiting 1s for initial viewport to stabilize...")
    await asyncio.sleep(1.0)

    screenshot_paths: list[str] = []
    for segment in range(1, max_segments + 1):
        metrics = await _get_scroll_metrics(browser_session)
        if not metrics:
            print("⚠️  Unable to read scroll metrics, stopping.")
            break

        # No cookie checks - already handled by agent
        screenshot_file = output_dir / f"{base_name}_part_{segment:02d}.png"
        print(
            "📷 Capturing segment %s | top=%spx height=%spx"
            % (segment, int(metrics["currentScroll"]), int(metrics["viewportHeight"]))
        )

        data = await _capture_viewport_screenshot(
            browser_session=browser_session,
            metrics=metrics,
            output_path=screenshot_file,
        )
        screenshot_paths.append(str(screenshot_file.resolve()))
        print(f"   Saved {screenshot_file.name} ({len(data) / 1024:.1f} KB)")

        if not metrics["hasMoreContent"]:
            print("✅ Reached end of page.")
            break

        next_offset = metrics["currentScroll"] + metrics["viewportHeight"] - scroll_overlap
        await _scroll_to(browser_session, max(next_offset, metrics["currentScroll"] + 1))
        print(f"   ⏳ Waiting {viewport_pause}s for dynamic content to load...")
        await asyncio.sleep(viewport_pause)

    if not screenshot_paths:
        raise RuntimeError("Failed to capture any screenshots.")

    print("\n✨ Capture complete. Files:")
    for path in screenshot_paths:
        print(f"   • {path}")
    return screenshot_paths


async def _get_scroll_metrics(browser_session: BrowserSession) -> dict[str, float | bool] | None:
    """Fetch current scroll metrics via CDP."""
    script = """
    (() => {
        const doc = document.documentElement;
        const body = document.body;
        const totalHeight = Math.max(
            body.scrollHeight, body.offsetHeight,
            doc.clientHeight, doc.scrollHeight, doc.offsetHeight
        );
        const viewportHeight = window.innerHeight || doc.clientHeight;
        const viewportWidth = window.innerWidth || doc.clientWidth;
        const currentScroll = window.scrollY || window.pageYOffset || doc.scrollTop || 0;
        const remaining = totalHeight - (currentScroll + viewportHeight);
        return {
            totalHeight,
            viewportHeight,
            viewportWidth,
            currentScroll,
            devicePixelRatio: window.devicePixelRatio || 1,
            hasMoreContent: remaining > 10
        };
    })();
    """
    try:
        cdp_session = await browser_session.get_or_create_cdp_session()
        result = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": script, "returnByValue": True},
            session_id=cdp_session.session_id,
        )
        return result.get("result", {}).get("value")
    except Exception as exc:
        print(f"⚠️  Failed to read scroll metrics: {exc}")
        return None


async def _scroll_to(browser_session: BrowserSession, offset: float) -> None:
    """Scroll to a specific vertical offset via CDP."""
    script = f"window.scrollTo({{ top: {offset:.2f}, behavior: 'instant' }});"
    cdp_session = await browser_session.get_or_create_cdp_session()
    await cdp_session.cdp_client.send.Runtime.evaluate(
        params={"expression": script},
        session_id=cdp_session.session_id,
    )


async def _capture_viewport_screenshot(
    browser_session: BrowserSession,
    metrics: dict[str, float | bool],
    output_path: Path,
) -> bytes:
    """Capture only the currently visible viewport using clip coordinates."""
    clip = {
        "x": 0,
        "y": float(metrics["currentScroll"]),
        "width": float(metrics["viewportWidth"]),
        "height": float(metrics["viewportHeight"]),
        "scale": float(metrics.get("devicePixelRatio") or 1) * 1.0,
    }
    data = await browser_session.take_screenshot(
        path=str(output_path),
        full_page=False,
        format="png",
        clip=clip,
    )
    return data


async def _accept_cookies_with_llm(browser_session: BrowserSession) -> bool:
    """Use LLM-powered agent to intelligently detect and accept cookie banners.
    
    Returns True if cookies were accepted, False otherwise.
    """
    # Check for API key (can be OPENAI_API_KEY for LiteLLM or BROWSER_USE_API_KEY)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("BROWSER_USE_API_KEY")
    if not api_key:
        return False
    
    try:
        print("🤖 Using LLM to intelligently detect and accept cookie banner...")
        
        # Configure LLM - try LiteLLM first, fallback to ChatBrowserUse
        # Try to use LiteLLM if OPENAI_API_KEY is set
        if os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(
                model="gemini-2.5-flash-lite-preview-09-2025",  # Fast model via LiteLLM
                timeout=httpx.Timeout(180.0, connect=60.0, read=180.0, write=30.0),
                max_retries=2,
                # Gemini compatibility options via LiteLLM
                add_schema_to_system_prompt=False,  # Don't add schema to prompt (Gemini handles it differently)
                remove_min_items_from_schema=True,  # Gemini doesn't like minItems
                remove_defaults_from_schema=True,  # Gemini doesn't like defaults with anyOf
                dont_force_structured_output=True,  # Don't force response_format (let LiteLLM handle it)
            )
            print("   📡 Using LiteLLM (Gemini) for cookie detection...")
        else:
            # Fallback to ChatBrowserUse if only BROWSER_USE_API_KEY is set
            llm = ChatBrowserUse()
            print("   📡 Using ChatBrowserUse for cookie detection...")
        
        # Create a temporary agent just for cookie acceptance with a very specific task
        task = """
        URGENT TASK: Find and click the "Accept" or "Accept all" button on the cookie consent banner.
        
        STEP-BY-STEP INSTRUCTIONS:
        
        1. EXAMINE THE SCREENSHOT: Look at the entire page carefully. Cookie banners are usually:
           - White or light-colored popup boxes in the center or corners
           - Contain text like "This website uses cookies", "Cookies consent", "We use cookies"
           - Have buttons labeled "Accept all", "Accept", "Agree", "Allow all", etc.
        
        2. IDENTIFY THE BANNER: If you see ANY popup, modal, or banner that mentions "cookies", "consent", or "privacy", that's the cookie banner.
        
        3. FIND THE ACCEPT BUTTON: Look for buttons in the banner. The button you need to click is usually:
           - Labeled "Accept all" (THIS IS THE ONE - click it!)
           - Or "Accept", "Accept cookies", "Agree", "Allow all", "OK", "Continue"
           - Usually has a dark background (grey, black, or colored)
           - Do NOT click "Reject", "Decline", "Manage preferences", or "Reject all"
        
        4. CLICK IT: Use the click action to click on the "Accept all" or "Accept" button.
           - If you see the button in the screenshot, use its index number from the page
           - The button index is usually a small number (0, 1, 2, 3, etc.)
        
        5. VERIFY: After clicking, the banner should disappear. If it's still there, try clicking again or use done().
        
        6. IF NO BANNER: If you don't see any cookie banner in the screenshot, use done() to complete.
        
        REMEMBER: Cookie banners block the page content. You MUST accept them before proceeding.
        Look at the screenshot - if you see a popup with cookie-related text, click "Accept all" immediately!
        """
        
        agent = Agent(
            task=task,
            llm=llm,
            browser_session=browser_session,
            max_steps=8,  # Allow more steps for thorough detection
            use_vision=True,  # Ensure vision is enabled to see the banner
            flash_mode=True,  # Simplify output for Gemini compatibility
            use_thinking=False,  # Disable thinking for faster, simpler output
        )
        
        try:
            history = await agent.run()
            # Check if the agent completed successfully
            if history.is_successful():
                # Check if agent actually clicked something (not just said done)
                actions = history.action_names()
                if any('click' in action.lower() for action in actions):
                    print("🍪 LLM successfully detected and accepted cookie banner.")
                    await asyncio.sleep(1.5)  # Wait for banner to disappear
                    return True
                else:
                    print("🍪 LLM checked but found no cookie banner to accept.")
                    return False
            else:
                print("🍪 LLM did not successfully handle cookie banner.")
                return False
        finally:
            await agent.close()
    except Exception as exc:
        print(f"⚠️  LLM cookie detection failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


async def _quick_cookie_check(browser_session: BrowserSession) -> None:
    """Quick check for cookie banner - single attempt without delays."""
    try:
        cdp_session = await browser_session.get_or_create_cdp_session()
        script = """
        (() => {
            const normalize = text => (text || '').toLowerCase().replace(/\\s+/g, ' ').trim();
            const isVisible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                return style.display !== 'none' && style.visibility !== 'hidden' && 
                       style.opacity !== '0' && el.offsetParent !== null &&
                       el.offsetWidth > 0 && el.offsetHeight > 0;
            };
            
            // Look for "uses cookies" or "This website uses cookies"
            const allElements = document.querySelectorAll('*');
            for (const el of allElements) {
                const elText = normalize(el.innerText || el.textContent || '');
                if (elText.includes('uses cookies') || elText.includes('website uses cookies') ||
                    elText.includes('this website uses cookies')) {
                    if (isVisible(el)) {
                        const buttons = el.querySelectorAll('button, [role="button"]');
                        for (const btn of buttons) {
                            const btnText = normalize(btn.innerText || btn.textContent || '');
                            if ((btnText.includes('accept all') || btnText === 'accept all') && isVisible(btn)) {
                                btn.click();
                                return true;
                            }
                        }
                        for (const btn of buttons) {
                            const btnText = normalize(btn.innerText || btn.textContent || '');
                            if (btnText.includes('accept') && isVisible(btn)) {
                                btn.click();
                                return true;
                            }
                        }
                    }
                }
            }
            
            // Look for fixed/sticky popups with cookie content
            for (const el of allElements) {
                const style = window.getComputedStyle(el);
                if ((style.position === 'fixed' || style.position === 'sticky') && isVisible(el)) {
                    const elText = normalize(el.innerText || el.textContent || '');
                    if (elText.includes('cookie') || elText.includes('consent') || elText.includes('uses cookies')) {
                        const buttons = el.querySelectorAll('button, [role="button"]');
                        for (const btn of buttons) {
                            const btnText = normalize(btn.innerText || btn.textContent || '');
                            if ((btnText.includes('accept all') || btnText === 'accept all') && isVisible(btn)) {
                                btn.click();
                                return true;
                            }
                        }
                        for (const btn of buttons) {
                            const btnText = normalize(btn.innerText || btn.textContent || '');
                            if (btnText.includes('accept') && isVisible(btn)) {
                                btn.click();
                                return true;
                            }
                        }
                    }
                }
            }
            
            // Look for "Cookies consent" or similar
            for (const el of allElements) {
                const elText = normalize(el.innerText || el.textContent || '');
                if (elText.includes('cookies consent') || elText.includes('cookie consent') || 
                    (elText.includes('consent') && elText.includes('cookie'))) {
                    if (isVisible(el)) {
                        const buttons = el.querySelectorAll('button, [role="button"]');
                        for (const btn of buttons) {
                            const btnText = normalize(btn.innerText || btn.textContent || '');
                            if (btnText.includes('accept') && isVisible(btn)) {
                                btn.click();
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        })();
        """
        result = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": script, "returnByValue": True},
            session_id=cdp_session.session_id,
        )
        if result.get("result", {}).get("value", False):
            print("   🍪 Cookie banner detected and accepted during scroll.")
    except Exception:
        pass  # Silently fail on quick checks


async def _accept_cookies_pattern_only(browser_session: "BrowserSession") -> None:
    """Detect and accept cookie consent popups using ONLY pattern-based method (no LLM)."""
    print("🔍 Using pattern-based cookie detection (no LLM)...")
    script = """
    (() => {
        const keywords = [
            'accept all', 'accept cookies', 'accept', 'agree', 'allow all', 'allow cookies',
            'allow', 'consent', 'continue', 'ok', 'got it', 'i agree', 'i accept',
            "j'accepte", 'accepter', 'tout accepter', 'allow essential', 'accepter tout',
            'tous accepter', 'ok, j\'accepte', 'd\'accord', 'continuer', 'je comprends',
            'aceptar', 'aceptar todo', 'permitir', 'continuar', 'de acuerdo',
            'akzeptieren', 'alle akzeptieren', 'zustimmen', 'fortfahren',
            'accetta', 'accetta tutto', 'consenti', 'continua',
        ];

        const normalize = text => (text || '').toLowerCase().replace(/\\s+/g, ' ').trim();

        const isVisible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none' && 
                style.visibility !== 'hidden' && 
                style.opacity !== '0' &&
                el.offsetParent !== null &&
                el.offsetWidth > 0 && 
                el.offsetHeight > 0;
        };

        // Quick check: if no cookie banner is visible, don't do anything
        const hasVisibleCookieBanner = () => {
            const cookieTexts = ['cookie', 'consent', 'privacy', 'gdpr'];
            const allElements = document.querySelectorAll('*');
            for (const el of allElements) {
                const elText = normalize(el.innerText || el.textContent || '');
                if (cookieTexts.some(text => elText.includes(text)) && isVisible(el)) {
                    const buttons = el.querySelectorAll('button, [role="button"]');
                    for (const btn of buttons) {
                        const btnText = normalize(btn.innerText || btn.textContent || '');
                        if (btnText.includes('accept') && isVisible(btn)) {
                            return true;
                        }
                    }
                }
            }
            return false;
        };

        if (!hasVisibleCookieBanner()) {
            return false;
        }

        const attemptCookieAccept = (container = document) => {
            // Find banner-like elements
            const bannerSelectors = [
                '[id*="cookie" i]', '[id*="consent" i]', '[id*="privacy" i]', '[id*="gdpr" i]',
                '[class*="cookie" i]', '[class*="consent" i]', '[class*="privacy" i]', '[class*="gdpr" i]',
                '[class*="banner" i]', '[class*="popup" i]', '[class*="modal" i]',
                '[data-testid*="cookie" i]', '[data-testid*="consent" i]',
                '[aria-label*="cookie" i]', '[aria-label*="consent" i]',
                '[role="dialog"]', '[role="alertdialog"]'
            ];
            for (const selector of bannerSelectors) {
                const banners = container.querySelectorAll(selector);
                for (const banner of banners) {
                    if (!isVisible(banner)) continue;
                    const buttons = banner.querySelectorAll('button, [role="button"], input[type="submit"], a, [onclick], [class*="btn"]');
                    // Prioritize buttons matching 'accept all', 'accept', or keywords
                    for (const btn of buttons) {
                        const btnText = normalize(btn.innerText || btn.textContent || btn.value || btn.ariaLabel || btn.title || '');
                        if ((btnText === 'accept all' or btnText.includes('accept all')) and isVisible(btn)) {
                            btn.scrollIntoView({ behavior: 'instant', block: 'center' });
                            btn.click();
                            return true;
                        }
                    }
                    for (const btn of buttons) {
                        const btnText = normalize(btn.innerText || btn.textContent || btn.value || btn.ariaLabel || btn.title || '');
                        if ((btnText.includes('accept') or keywords.some(kw => btnText.includes(kw))) && isVisible(btn)) {
                            btn.scrollIntoView({ behavior: 'instant', block: 'center' });
                            btn.click();
                            return true;
                        }
                    }
                }
            }

            // Look for text-based banners
            const allElements = container.querySelectorAll('*');
            for (const el of allElements) {
                const elText = normalize(el.innerText || el.textContent || '');
                if (
                    elText.includes('cookie consent') || elText.includes('cookies consent')
                    || (elText.includes('consent') && (elText.includes('cookie') || elText.includes('cookies')))
                    || (elText.includes('uses cookies') || elText.includes('website uses cookies'))
                ) {
                    if (isVisible(el)) {
                        const buttons = el.querySelectorAll('button, [role="button"], input[type="submit"], a, [onclick], [class*="btn"]');
                        for (const btn of buttons) {
                            const btnText = normalize(btn.innerText || btn.textContent || btn.value || btn.ariaLabel || btn.title || '');
                            if (btnText.includes('accept all') && isVisible(btn)) {
                                btn.scrollIntoView({ behavior: 'instant', block: 'center' });
                                btn.click();
                                return true;
                            }
                        }
                        for (const btn of buttons) {
                            const btnText = normalize(btn.innerText || btn.textContent || btn.value || btn.ariaLabel || btn.title || '');
                            if (btnText.includes('accept') && isVisible(btn)) {
                                btn.scrollIntoView({ behavior: 'instant', block: 'center' });
                                btn.click();
                                return true;
                            }
                        }
                        for (const btn of buttons) {
                            const btnText = normalize(btn.innerText || btn.textContent || btn.value || btn.ariaLabel || btn.title || '');
                            if (btnText && keywords.some(kw => btnText.includes(kw)) && isVisible(btn)) {
                                btn.scrollIntoView({ behavior: 'instant', block: 'center' });
                                btn.click();
                                return true;
                            }
                        }
                        // As last resort, click the first visible button
                        for (const btn of buttons) {
                            if (isVisible(btn)) {
                                btn.scrollIntoView({ behavior: 'instant', block: 'center' });
                                btn.click();
                                return true;
                            }
                        }
                    }
                }
            }

            // Try direct keyword button search for all buttons
            const allButtons = container.querySelectorAll('button, [role="button"], input[type="submit"], a, [onclick], .btn');
            for (const btn of allButtons) {
                const btnText = normalize(btn.innerText || btn.textContent || btn.value || btn.ariaLabel || btn.title || '');
                if (
                    (['accept', 'accept all', 'accept cookies'].includes(btnText) ||
                    keywords.some(kw => btnText.includes(kw)))
                ) {
                    let parent = btn.parentElement;
                    for (let i = 0; i < 5 && parent; i++) {
                        const parentText = normalize(parent.innerText || parent.textContent || '');
                        if (parentText.includes('cookie') || parentText.includes('consent') || parentText.includes('privacy')) {
                            if (isVisible(btn)) {
                                btn.scrollIntoView({ behavior: 'instant', block: 'center' });
                                btn.click();
                                return true;
                            }
                            break;
                        }
                        parent = parent.parentElement;
                    }
                }
            }
            return false;
        };

        // Try in main document
        if (attemptCookieAccept(document)) {
            return true;
        }

        // Try in iframes
        const iframes = document.querySelectorAll('iframe');
        for (const iframe of iframes) {
            try {
                const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
                if (iframeDoc && attemptCookieAccept(iframeDoc)) {
                    return true;
                }
            } catch (e) { }
        }

        // Try in shadow DOM
        const walkShadowDOM = (root) => {
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null, false);
            let node;
            while (node = walker.nextNode()) {
                if (node.shadowRoot) {
                    if (attemptCookieAccept(node.shadowRoot)) {
                        return true;
                    }
                    if (walkShadowDOM(node.shadowRoot)) {
                        return true;
                    }
                }
            }
            return false;
        };
        if (walkShadowDOM(document)) {
            return true;
        }

        return false;
    })();
    """

    try:
        cdp_session = await browser_session.get_or_create_cdp_session()
        result = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": script, "returnByValue": True},
            session_id=cdp_session.session_id,
        )
        accepted = bool(result.get("result", {}).get("value", False))
        if accepted:
            print("🍪 Cookie banner detected and accepted.")
            await asyncio.sleep(1.0)
        else:
            print("🍪 No cookie banner visible (already accepted).")
    except Exception as exc:
        print(f"⚠️  Error checking cookies: {exc}")

async def main() -> int:
    """Entry point for CLI usage."""
    url = "https://example.com"

    import sys

    if len(sys.argv) > 1:
        url = sys.argv[1]

    try:
        paths = await smart_screenshot(url)
        print("\n✅ Completed capture:")
        for path in paths:
            print(f"   -> {path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\n💥 Failed to capture screenshots: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
