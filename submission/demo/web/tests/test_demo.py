"""Browser checks for the built demo (tinkerrl_demo.html), independent of the page's own claims.

Run: python3 tests/test_demo.py   (uses Playwright with the installed Google Chrome)
"""

import json
import math
import os
import shutil
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
ROOT = HERE.parents[2]
PAGE = "file://" + str(pathlib.Path(os.environ.get("DEMO_PAGE", str(HERE / "tinkerrl_demo.html"))).resolve())
DATA = json.loads(pathlib.Path(os.environ.get("DEMO_DATA", str(HERE / "src/data.json"))).read_text())
VIEWS = ["overview", "advantage", "zvf", "samestack", "platform", "answers", "stackdiff", "audit", "viva"]
CHROME = os.environ.get("CHROME_PATH") or shutil.which("chromium") or shutil.which("google-chrome")
if not CHROME:
    candidate = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    CHROME = candidate if pathlib.Path(candidate).exists() else None


def num(text):
    return float(text.replace("−", "-").replace("+", ""))


def p_pair_matches(text, raw, adjusted):
    """Compare both rendered p-values with the page's documented rounding."""
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not 0 <= v <= 1 for v in (raw, adjusted)):
        return False
    try:
        actual = [num(value.strip()) for value in text.split("/")]
    except (ValueError, AttributeError):
        return False
    expected = [float(format(value, ".1e" if value < .001 else ".3f")) for value in (raw, adjusted)]
    return len(actual) == 2 and all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
                                    for a, b in zip(actual, expected, strict=True))


def disagreement_disclosed(audit, text):
    value = audit.get("disagreements") if isinstance(audit, dict) else None
    return (type(value) is int and value >= 0
            and re.search(rf": {value} disagreements\b", text) is not None)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail and not ok else ''}")
    return ok


def main():
    from playwright.sync_api import sync_playwright

    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME)
        for width, scheme in [(1440, "light"), (1024, "dark"), (768, "light"), (375, "dark"), (320, "light")]:
            page = browser.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme)
            errors = []
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            page.on("console", lambda m, errors=errors: m.type == "error" and errors.append(m.text))
            for v in VIEWS:
                page.goto(f"{PAGE}#{v}")
                page.wait_for_timeout(150)
                overflow = page.evaluate("document.documentElement.scrollWidth") - width
                results.append(check(f"{v} renders at {width}px {scheme}, no page overflow",
                                     overflow <= 0 and page.locator("main section.view").count() == 1,
                                     f"overflow={overflow}"))
            results.append(check(f"no console or page errors at {width}px", not errors, "; ".join(errors[:3])))
            page.close()

        page = browser.new_page(viewport={"width": 1280, "height": 900})
        # The display consumes the normative artifact; it never re-pairs historical seed overlap.
        page.goto(f"{PAGE}#samestack")
        analysis = DATA.get("analysis", {})
        ref = analysis.get("primary_contrasts", {}).get("g8-ppo", {})
        if ref.get("n", 0) >= 2 and ref.get("t_ci"):
            stats = dict(zip(page.locator(".finding dl.stat dt").all_inner_texts(), page.locator(".finding dl.stat dd").all_inner_texts(), strict=True))
            lo, hi = map(num, re.findall(r"[−+-]?\d\.\d+", stats["95% difference CI"]))
            results.append(check("G8−PPO mean matches canonical artifact", abs(num(stats["Mean difference"]) - ref["mean"]) < 5e-5))
            results.append(check("95% difference CI matches artifact", abs(lo - ref["t_ci"][0]) < 5e-5 and abs(hi - ref["t_ci"][1]) < 5e-5))
            results.append(check("90% equivalence CI distinctly labeled", "90% equivalence CI" in stats))
            results.append(check("combined raw/Holm TOST matches artifact", p_pair_matches(
                stats.get("Combined TOST p (raw / Holm)", ""), ref.get("tost_p"), ref.get("holm_tost_p"))))
            results.append(check("raw/Holm sign-flip matches artifact", p_pair_matches(
                stats.get("Exact sign-flip p (raw / Holm)", ""), ref.get("p_signflip"), ref.get("holm_p_signflip"))))
        else:
            results.append(check("missing validated pairs show unavailable inference", page.locator("text=Inferential statistics unavailable").count() > 0))
            results.append(check("missing provenance never yields a final equivalence verdict", "Equivalence established" not in page.locator("main").inner_text()))
        results.append(check("fixed family and exploratory distinction visible", "Missing tests contribute p = 1" in page.locator("main").inner_text() and "G2−PPO-style is exploratory" in page.locator("main").inner_text()))
        results.append(check("v2 does not assert every arm improves", "every arm improves" not in page.locator("main").inner_text()))
        if DATA.get("samestack_v1"):
            page.get_by_role("button", name="v1", exact=True).click()
            results.append(check("v1 avoids single-factor causal attribution", "cannot identify which change caused" in page.locator("main").inner_text()))
            page.get_by_role("button", name="v2", exact=True).click()
        results.append(check("no nonfinite values in same-stack display", not re.search(r"\b(?:NaN|Infinity)\b", page.locator("main").inner_text())))

        # Preserve the separate historical platform display and all base accuracies.
        if DATA.get("platform"):
            plat = DATA["platform"]
            want = {f"{state['acc']:.4f}" for platform in plat["platforms"] for state in plat["base"][platform].values()}
            page.goto(f"{PAGE}#platform")
            text = page.locator("main section.view").inner_text()
            results.append(check("platform view shows every exported base accuracy",
                                 len(want) > 1 and all(value in text for value in want)))

        # Re-grading coverage and any disagreement remain visible without changing history.
        page.goto(f"{PAGE}#answers")
        pill = page.locator(".pill", has_text="re-graded").inner_text()
        a, b = map(int, re.match(r"(\d+)/(\d+)", pill).groups())
        results.append(check("shown re-grade coverage matches exported examples", 0 <= a <= b and b == 3 * len(DATA["examples"]["items"]), pill))
        sys.path.insert(0, str(ROOT))
        from platform_hybrid.experiments.samestack_protocol import GRADING_SPEC_ID
        audit = DATA.get("grader_check", {})
        answer_text = page.locator("main").inner_text()
        if audit.get("grading_spec_id") == GRADING_SPEC_ID:
            results.append(check("current disagreement count is disclosed", disagreement_disclosed(audit, answer_text)))
        else:
            results.append(check("stale or missing grader audit is disclosed",
                                 "Full audit for the current grading specification unavailable" in answer_text))
        # The grader probe follows the strict rule.
        for answer, gold, want in [("so \\boxed{18}", "18", "correct"), ("the answer is 18", "18", "wrong"),
                                   ("\\boxed{1,234.00}", "1234", "correct"), ("\\boxed{\\$1,234.00}", "1234", "correct"), ("\\boxed{10\\%}", "10", "correct"), ("\\boxedX{42}", "42", "wrong"), ("\\boxed{1/0}", "1", "wrong"), ("\\boxed{0.0000004}", "0", "wrong"), ("\\boxed{17}", "18", "wrong")]:
            page.fill("#probe-text", answer)
            page.fill("#probe-gold", gold)
            got = page.locator(".panel", has_text="Try the grader").locator(".pill").inner_text()
            results.append(check(f"grader probe: {answer!r} vs {gold} -> {want}", got == want, got))

        # Presenter controls.
        page.goto(f"{PAGE}#overview")
        page.keyboard.press("ArrowRight")
        results.append(check("→ moves to the next view", page.evaluate("location.hash") == "#advantage"))
        page.keyboard.press("9")
        results.append(check("number key jumps to a view", page.evaluate("location.hash") == "#viva"))
        page.keyboard.press("p")
        results.append(check("P toggles presenter mode", "present" in page.evaluate("document.documentElement.className")))

        # Navigation remains usable without global shortcuts or a pointer.
        page.close()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(f"{PAGE}#overview")
        page.keyboard.press("Tab")
        results.append(check("skip link is the first keyboard stop", page.locator(".skip-link").evaluate("el => el === document.activeElement")))
        page.keyboard.press("Enter")
        results.append(check("skip link focuses content without changing the route",
                             page.evaluate("document.activeElement.id === 'main-content' && location.hash === '#overview'")))
        page.get_by_role("button", name="Shortcuts", exact=True).click()
        page.locator("main").focus()
        page.keyboard.press("9")
        page.keyboard.press("p")
        results.append(check("disabled shortcuts leave the view and presentation unchanged",
                             page.evaluate("location.hash === '#overview' && !document.documentElement.classList.contains('present')")))
        page.get_by_role("button", name="Shortcuts", exact=True).click()
        page.keyboard.press("9")
        page.wait_for_function("document.activeElement.id === 'main-content' && location.hash === '#viva'")
        results.append(check("route changes focus the named main landmark and update the title",
                             page.locator("main").get_attribute("aria-label") == "Hard questions"
                             and page.title().startswith("Hard questions")))
        results.append(check("the active desktop link is visible in the scrollable rail", page.evaluate("""() => {
            const active = document.querySelector('.section-links [aria-current]'), rail = active.parentElement;
            const a = active.getBoundingClientRect(), r = rail.getBoundingClientRect();
            return a.left >= r.left && a.right <= r.right;
        }""")))
        page.go_back()
        page.wait_for_function("location.hash === '#overview'")
        page.go_forward()
        page.wait_for_function("location.hash === '#viva'")
        results.append(check("Back and Forward restore section selection", page.locator("main").get_attribute("aria-label") == "Hard questions"))

        page.set_viewport_size({"width": 375, "height": 900})
        page.set_viewport_size({"width": 1024, "height": 900})
        page.wait_for_function("""() => {
            const active = document.querySelector('.section-links [aria-current]'), rail = active.parentElement;
            const a = active.getBoundingClientRect(), r = rail.getBoundingClientRect();
            return a.left >= r.left && a.right <= r.right;
        }""")
        results.append(check("resizing back to desktop keeps the active section visible", True))
        page.set_viewport_size({"width": 375, "height": 900})
        page.locator("#section-select").focus()
        page.locator("#section-select").select_option("answers")
        page.wait_for_function("location.hash === '#answers'")
        results.append(check("mobile section picker navigates without interrupting selection",
                             page.locator("#section-select").input_value() == "answers"
                             and page.evaluate("document.activeElement.id") == "section-select"))
        page.keyboard.press("ArrowDown")
        page.wait_for_function("location.hash === '#stackdiff'")
        results.append(check("mobile picker supports repeated arrow-key navigation", page.evaluate("document.activeElement.id") == "section-select"))
        page.locator("#section-select").select_option("answers")
        pattern_buttons = page.locator(".pattern-choice")
        if pattern_buttons.count() > 1:
            for key, index in [("Space", 0), ("Enter", 1)]:
                button = pattern_buttons.nth(index)
                button.focus()
                page.keyboard.press(key)
                results.append(check(f"outcome pattern supports {key} with a selected state", button.get_attribute("aria-pressed") == "true"))
        page.locator("#probe-text").focus()
        page.keyboard.press("9")
        page.keyboard.press("p")
        page.keyboard.press("ArrowLeft")
        results.append(check("typing in grader controls does not invoke shortcuts", page.evaluate("location.hash") == "#answers"))
        results.append(check("textarea keyboard focus is visible", page.locator("#probe-text").evaluate("el => getComputedStyle(el).outlineStyle !== 'none'")))

        page.locator("#section-select").select_option("advantage")
        page.get_by_role("button", name="G = 16", exact=True).click()
        results.append(check("reward controls retain 44px touch targets on narrow screens", page.locator(".coin").evaluate_all("els => els.length === 16 && els.every(el => el.getBoundingClientRect().width >= 44 && el.getBoundingClientRect().height >= 44)")))
        page.emulate_media(reduced_motion="reduce")
        results.append(check("reduced motion disables reward transitions", page.locator(".coin").first.evaluate("el => getComputedStyle(el).transitionDuration === '0s'")))
        page.get_by_role("button", name="All wrong", exact=True).click()
        page.get_by_role("button", name="All correct", exact=True).click()
        results.append(check("repeated preset changes leave all reward states correct", page.locator(".coin[aria-pressed='true']").count() == 16))
        page.set_viewport_size({"width": 1440, "height": 900})
        results.append(check("sixteen rewards use two rows of eight on desktop", page.locator(".coin").evaluate_all("""els => {
            const rows = els.map(el => el.getBoundingClientRect().top);
            return rows.length === 16 && rows.slice(0, 8).every(top => top === rows[0])
                && rows.slice(8).every(top => top === rows[8]) && rows[0] !== rows[8];
        }""")))
        page.set_viewport_size({"width": 375, "height": 900})

        # Essential labels remain visible under spacing overrides and presenter mode.
        page.locator("#section-select").select_option("overview")
        page.get_by_role("button", name="Present", exact=True).click()
        page.add_style_tag(content="* { line-height: 1.5 !important; letter-spacing: .12em !important; word-spacing: .16em !important; } p { margin-bottom: 2em !important; }")
        results.append(check("presenter mode and text-spacing overrides reflow on mobile", page.evaluate("document.documentElement.scrollWidth <= innerWidth")))
        results.append(check("status badges retain their full content", page.locator(".pill").evaluate_all("els => els.every(el => el.scrollWidth <= el.clientWidth + 1 && el.scrollHeight <= el.clientHeight + 1)")))
        page.close()
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        # The advantage lab: an all-correct group has zero advantage everywhere.
        page.goto(f"{PAGE}#advantage")
        page.get_by_role("button", name="All correct").click()
        results.append(check("all-correct group is flagged as zero-variance", page.locator(".callout", has_text="zero-variance group").count() == 1))
        browser.close()

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
