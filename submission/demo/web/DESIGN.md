# TinkerRL Measurement Lab — Design System

## 1. Atmosphere & Identity

TinkerRL Measurement Lab is a high-precision **Lab Instrument Panel** engineered for live academic defense and reproducible post-training auditing. Rather than decorative SaaS chrome, the interface reads like a calibrated scientific bench: one sticky section rail over one focused measurement surface per study, summary metrics at the top, and raw interactive evidence immediately below. Every displayed statistic, confidence interval, zero-variance curve, and graded completion is computed directly from stored repository records offline with zero network dependency.

The visual signature is **warm-cool analytical contrast over crisp hairline containment**: calm slate-mineral surfaces (`--bg`, `--panel`, `--line`) keep visual noise near zero so the eye immediately distinguishes the two competing reinforcement-learning regimes—deep teal (`--grpo`) for Group Relative Policy Optimization and warm amber (`--ppo`) for Proximal Policy Optimization—alongside green equivalence bands (`--band`, `--good`).

---

## 2. Color

### Palette

| Role | Token | Light (`:root`) | Dark (`[data-theme="dark"]` / OS dark) | Usage |
|------|-------|-----------------|----------------------------------------|-------|
| Surface / Primary | `--bg` | `#f3f6f7` | `#0f171c` | Main page canvas, code completion wells, input backgrounds |
| Surface / Panel | `--panel` | `#ffffff` | `#16222a` | Study panels, cards, control backgrounds, active pill text |
| Text / Primary | `--ink` | `#142029` | `#e3ecf0` | Headlines, body prose, primary table and stat values |
| Text / Secondary | `--muted` | `#5a6b76` | `#91a4ae` | Eyebrows, axis labels, secondary descriptions, inactive tabs |
| Border / Default | `--line` | `#d5dee3` | `#2a3a44` | Panel hairlines, table row dividers, segmented control frames |
| Accent / GRPO | `--grpo` | `#1d6b86` | `#5fb4d3` | Primary interactive accent, GRPO series, active tab indicator, focus rings |
| Accent / PPO | `--ppo` | `#b7791f` | `#e0a64e` | PPO / Dr. GRPO comparison series, R3 stack-diff badge |
| Surface / PPO Soft | `--ppo-soft` | `#f5e7cf` | `#3d2f18` | Warm amber fallback tint |
| Status / Good | `--good` | `#23784a` | `#5cc28a` | Equivalence pass, correct answers (`1` / `✓`), positive verification |
| Status / Warn | `--warn` | `#a86a00` | `#e3a63a` | Provisional / interim warnings, R4 stack-diff badge, caution callouts |
| Status / Bad | `--bad` | `#b3372f` | `#ec776c` | Failed checks, wrong answers (`✗`), R5 outcome-critical differences |
| Surface / Band | `--band` | `rgba(35, 120, 74, .12)` | `rgba(92, 194, 138, .14)` | Equivalence margin (`±0.02`) shaded band, selected table rows, boxed answer highlight |

### Rules
- **Three-state theme resolution (`system` | `light` | `dark`):** Stored under `localStorage["demo.theme"]` and reflected on `<html data-theme="...">`. When set to `system`, `:root:not([data-theme="light"])` inside `@media (prefers-color-scheme: dark)` tracks the OS preference automatically without a flash on reload.
- **No decorative gradients or colored left-border stripes:** Callouts and selected rows never use `border-left: 4px` stripes. State and severity use a 1px full-perimeter tonal border (`color-mix`), a soft tonal surface wash, and an explicit text/glyph label.
- **Never introduce raw hex values in components:** All component styles reference the CSS custom properties above.

---

## 3. Typography

### Scale

| Level | Size | Weight | Line Height | Tracking | Font Family | Usage |
|-------|------|--------|-------------|----------|-------------|-------|
| Display (`h1`) | `clamp(28px, 4.4vw, 44px)` | `700` | `1.08` | `-0.01em` | `--display` | Overview hero title (`text-wrap: balance`) |
| Section (`h2`) | `24px / 1.5rem` | `700` | `1.20` | `0` | `--display` | Primary study view heading |
| Brand Compact (`.brand`) | `20px / 1.25rem` | `700` | `1.20` | `0` | `--display` | Compact header title on non-overview routes |
| Metric Readout (`.big`) | `30px / 1.875rem` (`24px` mobile, `38px` present) | `600` | `1.10` | `0` | `--mono` | Headline statistical readout (`tabular-nums`) |
| Lede (`.lede`) | `17px / 1.0625rem` | `400` | `1.55` | `0` | `--body` | Overview introductory summary |
| Body (`body`, `p`) | `16px / 1rem` (`19px` present) | `400` | `1.55` | `0` | `--body` | Default narrative prose (`max-width: 68ch`) |
| Subheading (`h3`) | `15px / 0.9375rem` | `600` | `1.30` | `0` | `--body` | Panel and chart titles |
| Secondary / Table (`.small`, `table`, `.stat`) | `14px / 0.875rem` (`17px` present) | `400–600` | `1.45` | `0` | `--body` / `--mono` | Captions, controls, data tables, stat lists |
| Code Completion (`pre.completion`) | `13px / 0.8125rem` | `400` | `1.50` | `0` | `--mono` | Raw model completion transcripts |
| Eyebrow / Pill / Table Header (`.eyebrow`, `.pill`, `th`) | `12px / 0.75rem` | `600` | `1.00–1.35` | `0.04em–0.08em` | `--mono` | Uppercase metadata labels, status pills, column headers |
| Chart Axis Text (`svg text`, `.key`) | `11px / 0.6875rem` (`13px` present) | `600` | `1.00` | `0` | `--mono` | SVG axis ticks and tab shortcut badges |

### Font Stack
- **Display (`--display`):** `"Bricolage Grotesque", "Avenir Next", "Segoe UI", system-ui, sans-serif`
- **Body (`--body`):** `"Source Sans 3", "Segoe UI", system-ui, -apple-system, sans-serif`
- **Monospace (`--mono`):** `"JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace`

### Rules
- All numerical comparisons, confidence intervals, p-values, SHA-256 hashes, and seed counts use `font-variant-numeric: tabular-nums`.
- Presenter mode (`:root.present`, toggled via `P` or the `Present` button) scales the root font size to `19px` and widens `.wrap` to `1360px` for projector legibility.

---

## 4. Spacing & Layout

### Base Unit
All spacing derives from a **4px** base scale (`4px`, `6px`, `8px`, `10px`, `12px`, `14px`, `16px`, `18px`, `20px`, `24px`, `28px`, `32px`, `48px`).

| Pattern / Token | Value | Usage |
|-----------------|-------|-------|
| Tight inline gap | `4px–6px` | Tab rail gap, stat list vertical rhythm |
| Control / coin gap | `8px–10px` | `.coins` grid gap, `.row` flex gap, button padding |
| Panel padding | `18px` | `.panel` interior padding |
| Grid / View gap | `16px–20px` | `.grid3` (`16px`), `.grid2` and `section.view` (`20px`) |
| Main vertical padding | `24px` top / `48px` bottom | `main` content breathing room |

### Grid & Scroll Ownership
- **Container (`.wrap`):** `max-width: 1120px` (`1360px` in `:root.present`), centered with `padding-inline: max(16px, env(safe-area-inset-*, 0px))`.
- **Two-column layout (`.grid2`):** `grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); gap: 20px`.
- **Three-column layout (`.grid3`):** `grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: 16px`.
- **Sampled answer coins (`.coins`):** `grid-template-columns: repeat(var(--coin-cols, 8), minmax(44px, 64px)); gap: 8px`, collapsing at `<= 560px` to `repeat(min(var(--coin-cols, 8), 4), minmax(44px, 1fr))` so mobile viewports (`390px`) wrap cleanly into 4 columns with zero overflow and `>= 44px` targets.
- **Scroll ownership:** `document.documentElement` owns vertical page scroll (`scrollWidth <= clientWidth` at all breakpoints from `320px` to `1920px`). Horizontal scrolling is strictly isolated to `.table-box`, `.chart-box`, and `nav.tabs .wrap`, while `pre.completion` owns bounded vertical scroll (`max-height: 340px`).

---

## 5. Components

### `.panel` & `.finding` (Study Surface & Summary Card)
- **Structure:** `<div className="panel">` or `<a className="panel finding card-link" href="#view">`
- **Variants:** Standard container (`.panel`), vertical metric stack (`.panel.finding`), interactive overview card (`.panel.finding.card-link`), viva Q&A card (`.panel.qa`).
- **States:** Default (`1px solid var(--line)`), hover/focus on interactive cards (`border-color: var(--grpo)`, visible `2px` focus ring, `.card-go` action cue).

### `nav.tabs` (Sticky Section Rail)
- **Structure:** `<nav className="tabs" aria-label="Sections"><div className="wrap"><a href="#id"><span className="key">1</span>Label</a>...</div></nav>`
- **States:** Default (`color: var(--muted)`), hover (`color: var(--ink)`), active (`aria-current="page"`, `color: var(--ink)`, `border-bottom: 3px solid var(--grpo)`), keyboard focus (`2px solid var(--grpo)`). Auto-scrolls active tab into view on route transitions.

### `.seg` & `button` / `select` (Interactive Controls)
- **Structure:** `<div className="seg" role="group" aria-label="...">` wrapping `<button type="button" aria-pressed={...}>`.
- **Dimensions:** `min-height: 44px` across all buttons, selects, segmented controls, and `<summary>` disclosures for touch accessibility.
- **States:** Default (`background: var(--panel)`), hover (`border-color: var(--grpo)`), active/pressed (`aria-pressed="true"`, `background: var(--grpo); color: var(--panel)`), disabled (`opacity: 0.48; cursor: not-allowed`), focus-visible (`2px solid var(--grpo)`).

### `.coin` (Binary Reward Toggle in Advantage Lab)
- **Structure:** `<button type="button" className="coin on" data-coin={i} data-correct="1" aria-pressed={true} aria-label="Coin 1, answer 1: correct">`
- **Dimensions:** `min-width: 44px; min-height: 44px; aspect-ratio: 1; border-radius: 10px`.
- **States:** Wrong/`0` (`background: var(--panel); border: 2px solid var(--line)`), Correct/`1` (`.coin.on`, `background: var(--good); color: var(--panel); border-color: var(--good)`).

### `.callout` (Analytical Verdict & Caveat Banner)
- **Structure:** `<div className="callout [ok]" role="status">...</div>`
- **Styling:** `1px solid` tonal border (`color-mix` of `--warn` or `--good` with `--line`), `10%` tonal surface wash over `--panel`, `8px` concentric radius on all corners. Never uses a thick left border stripe.

### `.pill` & `.lvl` (Semantic Status Badges)
- **Variants:** `.pill.good`, `.pill.warn`, `.pill.bad`, `.pill.info` (`border: 1px solid currentColor; border-radius: 999px`); `.lvl.l0`–`.lvl.l5` (`border-radius: 6px` risk level badges in `StackDiff`).

---

## 6. Motion & Interaction

### Timing
| Type | Duration | Easing | Usage |
|------|----------|--------|-------|
| Micro state transition | `150ms–180ms` | `ease` | `.bar > span` width, `.coin` toggle, `.skip-link` reveal |
| Tab rail centering | smooth / instant | browser native | Centering active tab on route change |

### Rules
- **Compositor-friendly & reduced-motion safe:** Transitions are gated behind `@media (prefers-reduced-motion: no-preference)`. Programmatic `scrollIntoView` and `scrollTo` check `window.matchMedia("(prefers-reduced-motion: reduce)").matches` and fall back to `"auto"` (instant) when reduced motion is requested.
- **Immediate keyboard navigation:** `←` / `→` cycle views in defense order; `1`–`9` jump directly to any view; `P` toggles presenter mode; `Enter` and `Space` activate selectable pattern rows (`tr.pick`).

---

## 7. Depth & Surface

### Strategy
**Hairline Borders + Tonal Layering (`borders-and-tonal-wash`)**:
- Primary containment uses crisp `1px solid var(--line)` borders separating `--panel` cards from the `--bg` canvas.
- Sticky navigation (`nav.tabs`) uses translucent backdrop blur (`background: color-mix(in srgb, var(--bg) 85%, transparent); backdrop-filter: blur(12px) saturate(1.1)`) so scrolled content remains softly stratified beneath the rail.
- Semantic emphasis (equivalence bands, callouts, selected rows, highlighted `\boxed{}` answers) uses low-alpha tonal washes (`--band` and `color-mix` washes) paired with `1px` borders.

### Radius
| Element | Radius | Usage |
|---------|--------|-------|
| `.panel`, `.coin` | `10px` | Primary study panels and reward coins |
| `button`, `select`, `input`, `textarea`, `.callout`, `pre.completion`, `.seg` | `8px` | Interactive controls, callouts, code blocks (concentric inside `10px` panels) |
| `.lvl` | `6px` | Compact R0–R5 risk badges |
| `.bar` | `5px` | Progress and distribution bars |
| `.pill` | `999px` | Status pills |

---

## 8. Accessibility Constraints & Accepted Debt

### Constraints
- **WCAG 2.2 AA Contrast:** All text tokens (`--ink`, `--muted`, `--grpo`, `--good`, `--warn`, `--bad`) maintain >= 4.5:1 contrast against `--bg` and `--panel` in both Light and Dark themes.
- **Touch & Pointer Targets:** Every interactive control (`button`, `select`, `nav.tabs a`, `details summary`, `.coin`, `tr.pick`) enforces a minimum `44px` hit height.
- **Keyboard & Screen Reader Navigation:**
  - First focusable element is `<a className="skip-link" href="#main">Skip to content</a>` targeting `<main id="main" tabIndex={-1}>`.
  - Every SVG chart provides a `<title>` and `<desc>` summary alongside per-point `<title>` tooltips.
  - All segmented button groups carry `role="group"` and an explicit `aria-label`.

### Accepted Debt
| Item | Location | Why accepted | Owner / Exit |
|------|----------|--------------|--------------|
| Offline single-file bundle size (~590 KB) | `submission/demo/web/tinkerrl_demo.html` | Intentional architecture (`vite-plugin-singlefile`) so the defense demo embeds `src/data.json` (27 runs + stored completions) and opens from disk via `file://` with zero server or internet access. | Permanent requirement per `submission/demo/web/README.md` |
