"""Browser checks for the built demo (tinkerrl_demo.html), independent of the page's own claims.

Run: python3 tests/test_demo.py   (uses Playwright with the installed Google Chrome)
"""

import json
import pathlib
import re
import sys

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent.parent
ROOT = HERE.parents[2]
PAGE = "file://" + str(HERE / "tinkerrl_demo.html")
V2 = json.loads((ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2/v2_analysis.json").read_text())
PLATFORM = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2/platform_calibration.json"
VIEWS = ["overview", "advantage", "zvf", "samestack", "platform", "answers", "stackdiff", "audit", "viva"]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def num(text):
    return float(text.replace("−", "-").replace("+", ""))


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail and not ok else ''}")
    return ok


def main():
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME if pathlib.Path(CHROME).exists() else None)
        for width, scheme in [(1280, "light"), (390, "dark")]:
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
        # In-page statistics against the Python analysis.
        page.goto(f"{PAGE}#samestack")
        stats = dict(zip(page.locator(".finding dl.stat dt").all_inner_texts(), page.locator(".finding dl.stat dd").all_inner_texts()))
        ref = V2["contrasts"]["stored"]["g8-ppo"]
        lo, hi = map(num, re.findall(r"[−+-]?\d\.\d+", stats["95% t interval"]))
        results.append(check("G8−PPO mean matches v2_analysis.json", abs(num(stats["Mean difference"]) - ref["mean"]) < 5e-5))
        results.append(check("G8−PPO 95% CI matches v2_analysis.json", abs(lo - ref["t_ci"][0]) < 5e-5 and abs(hi - ref["t_ci"][1]) < 5e-5))
        results.append(check("G8−PPO TOST p matches v2_analysis.json", abs(num(stats["TOST p, margin ±0.02"]) - V2["power"]["g8-ppo"]["tost_0.02_p"]) < 5e-4))
        results.append(check("G8−PPO sign-flip p matches v2_analysis.json", abs(num(stats["Exact sign-flip p"]) - ref["p_signflip"]) < 5e-4))

        # Platform view: every stored base accuracy (per platform and Modal state) is shown.
        if PLATFORM.exists():
            plat = json.loads(PLATFORM.read_text())
            want = {f"{s['acc']:.4f}" for p in plat["platforms"] for s in plat["base"][p].values()}
            page.goto(f"{PAGE}#platform")
            text = page.locator("main section.view").inner_text()
            results.append(check("platform view shows every base accuracy from platform_calibration.json",
                                 len(want) > 1 and all(w in text for w in want), f"want {sorted(want)}"))

        # Re-grading in the page agrees with stored scores.
        page.goto(f"{PAGE}#answers")
        pill = page.locator(".pill", has_text="re-graded").inner_text()
        a, b = map(int, re.match(r"(\d+)/(\d+)", pill).groups())
        results.append(check("every shown answer re-grades to its stored score", a == b and b > 0, pill))
        results.append(check("no answer card reports a disagreement", page.locator("text=DISAGREES").count() == 0))
        # The grader probe follows the strict rule.
        for answer, gold, want in [("so \\boxed{18}", "18", "correct"), ("the answer is 18", "18", "wrong"),
                                   ("\\boxed{\\$1,234.00}", "1234", "correct"), ("\\boxed{17}", "18", "wrong")]:
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
