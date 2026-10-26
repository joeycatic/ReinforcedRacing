# M7: Stadium Identity + Intro Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Restyle the game as Trackmania Stadium with Formula cars and medal opponents, and add an intro front page that tells the project's story to recruiters and engineers.

**Architecture:** Python export gains medal opponents (selected from existing training checkpoints) and two small site-data assets, under a `manifest.v2`. The web app becomes a two-page Vite build: `index.html` (intro) and `race.html` (the existing race + Watch-it-learn app, restyled). Rendering changes stay in presentation code; physics, observations and models are untouched.

**Tech Stack:** Python 3.12 / uv / PyTorch / ONNX (export side); Vite + TypeScript + PixiJS 8 + onnxruntime-web (web side); pytest, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-10-26-portfolio-v2-design.md` (section "M7"). Read it alongside this plan.

## Global Constraints

- No change to `physics.v1`, `obs.v1`, `track.v1`, `replay.v1`, `heatmap.v1` behaviour; all existing parity fixtures pass unchanged.
- No retraining. Only `artifacts/training/m3-seed2` is used; exporting more of its checkpoints is allowed.
- No external art assets and no new npm/PyPI dependencies. Fonts come from the already-installed `@fontsource/barlow-condensed` (add `latin-800-italic.css`, `latin-900.css` imports) and `@fontsource/ibm-plex-mono`.
- No external network requests from the built site (the static e2e test asserts this). No CI badge images.
- Colour families: player mint `#3fe0b0`, AI orange `#ff7a2f`, grass `#68a842`/`#5f9e3a`, tarmac `#5d6470`, kerbs `#d6312c`/`#e9e4d6`, checkpoint gates and "ahead" `#2f7cf6`, "behind" `#d6312c`, HUD boxes `#000000cc` with white heavy italic text.
- Medals: Bronze, Silver, Gold, Author (Author = `final` snapshot). Never show an opponent without a lap time on the selected track.
- Every number shown on the intro page is read from exported data, never typed into HTML.
- Commits: conventional commits, **no `Co-Authored-By` trailer**. Log every builder's-call decision in `DECISIONS.md` under a new `# M7 decisions` heading.
- `make test` green after every task; Playwright suites (`npm run test:e2e`, `npm run test:static` in `web/`) green at the end.

## Review Focus

1. **Medal order flips in the browser.** Live browser laps differ from Python by up to ~1%, so a 1% gap between medals could invert. Selection requires ≥3% per track (Task 1 test `test_select_medals_enforces_gap_on_every_track`).
2. **A medal with no lap on a track.** Its card must be absent on that track, and the result screen must not compare against it (Task 7 test `earnedMedal ignores medals without a lap on the track`; Task 6 test `medalsForTrack drops medals without a lap time`).
3. **Intro on a phone (390×844).** No horizontal scroll, buttons visible, Race button still works (Task 8 e2e `intro works at phone width`).
4. **Intro when the model or WASM fails to load.** The headline, stats and sections still render; the hero falls back to a static track (Task 8 e2e `intro survives a failed model load`; Task 10 keeps it passing).
5. **Fast clicking between track/medal cards.** Late model loads must not replace the latest choice (Task 6 e2e asserts the opponent label after a rapid click sequence).

---

### Task 1: Medal selection from training checkpoints

**Files:**
- Create: `eval/medals.py`, `tools/select_medals.py`, `tests/test_medals.py`
- Create (generated, committed): `configs/medals.yaml`, `experiments/medal-selection.json`
- Modify: `Makefile` (target `medals`)

**Interfaces:**
- Produces: `select_medals(candidates: dict[int, dict[str, float | None]], author: dict[str, float], gap: float = 0.03) -> list[MedalChoice]` and `@dataclass(frozen=True) class MedalChoice: medal: str; step: int; lap_times: dict[str, float | None]` in `eval/medals.py`. `load_medals(path: Path = ROOT / "configs/medals.yaml") -> list[MedalChoice]` (Bronze→Gold order).
- `configs/medals.yaml` shape: `run_dir: artifacts/training/m3-seed2`, `gap: 0.03`, `medals: [{medal: bronze, step: <int>, checkpoint: checkpoints/<step>.pt}, ...]`.

- [x] **Step 1: Write failing tests** in `tests/test_medals.py`:
  - `test_select_medals_picks_three_separated_finishers`: author `{"a": 20.0, "b": 30.0}`; candidates with totals spread so the expected choice is Gold=step 5 (laps 21.0/31.0), Silver=step 3 (22.0/32.5), Bronze=step 1 (23.0/34.0); assert `[c.medal for c in result] == ["bronze", "silver", "gold"]` and steps `[1, 3, 5]`.
  - `test_select_medals_enforces_gap_on_every_track`: a candidate 10% slower on track a but only 1% slower on b is rejected for Gold; assert for every consecutive pair and every track with both laps present: `slower >= faster * 1.03`.
  - `test_select_medals_prefers_full_finishers_then_partial`: only one candidate finishes all tracks; a second finishes only track a; assert two medals returned, the partial one has `lap_times["b"] is None`.
  - `test_select_medals_returns_fewer_when_none_qualify`: no candidate finishes anything → `[]`.
- [x] **Step 2: Run** `uv run --locked pytest tests/test_medals.py -v` → FAIL (module missing).
- [x] **Step 3: Implement `select_medals`.** Algorithm (the tests do not fully pin it):

```
ladder = [author]                       # fastest first
for medal in ("gold", "silver", "bronze"):
    previous = ladder[-1]
    def ok(c): every track where both c and previous have laps: c_lap >= previous_lap * (1 + gap)
    full    = [c for c in candidates if all tracks finished and ok(c)]
    partial = [c for c in candidates if any track finished and ok(c)]
    pick = min(full, key=total_lap) if full else max(partial, key=(tracks finished, -total over finished))  # if any
    stop when no pick; never reuse a step
return reversed(ladder[1:]) as MedalChoice list (bronze first)
```

- [x] **Step 4: Run** the tests → PASS.
- [x] **Step 5: Implement `tools/select_medals.py`**: for each file in `<run_dir>/checkpoints/*.pt` (sorted by step), `record_runs(checkpoint, make_track_pool("showcase"), load_export(), deterministic=True)`; lap per track = that track's `mean_lap_time` in the returned summary (`None` if not finished). Author laps from `snapshots/final.pt` the same way. Call `select_medals`, write `configs/medals.yaml` and `experiments/medal-selection.json` (every candidate's laps + the choice). Add `medals:` target to the `Makefile` (`$(UV) run --locked python -m tools.select_medals $(ARGS)`).
- [x] **Step 6: Run** `make medals` → prints a table and writes both files. Expect 3 medals; if fewer, that is the accepted fallback — record it in DECISIONS.md.
- [x] **Step 7: Commit** `feat: select medal opponents from training checkpoints` (include generated yaml/json and a DECISIONS.md entry with chosen steps and laps).

### Task 2: Export medals and site data (manifest.v2)

**Files:**
- Create: `export/site.py`
- Modify: `export/assets.py`, `tests/test_export.py`, `tools/plot_learning.py` (import shared loaders from `export/site.py`), `docs/ASSETS.md`
- Regenerated (committed): `web/public/assets/**`

**Interfaces:**
- Consumes: `load_medals()` (Task 1); `record_runs` (existing).
- Produces in `manifest.json`: `"version": "manifest.v2"`; new keys
  - `"medals": [{"medal": "bronze"|"silver"|"gold"|"author", "step": int, "checkpoint_sha256": str, "model": Asset, "lap_times": {track_id: float|null}, "replays": {track_id: Asset}}]`, ordered bronze→author. Author reuses `models/final.onnx` and the final snapshot's replays.
  - `"site": Asset` → `site.json`; `"parity_demo": Asset` → `parity-demo.json`.
- `site.json`: `{"stats": {"showcase_completion": float, "held_out_completion": float, "parity_max_error": float, "training_steps": int, "training_minutes": float}, "curves": {"<seed>": [[step, smoothed_return], ...]}, "snapshots": [[step, showcase_pct, held_out_pct], ...], "seeds": [{"seed": int, "showcase_pct": float, "held_out_pct": float, "deterministic_showcase_finishes": int}]}`. Sources: `artifacts/training/m3-seed{1,2}/metrics.jsonl` (15-update moving average, as in `tools/plot_learning.py`), `artifacts/evaluation/m4/summary.json`, `experiments/evaluation/seed{1,2}-6m-*.json`, `docs/m4-validation.json` (`parity.max_measured_absolute_error`), `experiments/m3-results.json` (`runs["2"].final_training_seconds`).
- `parity-demo.json`: from `tests/fixtures/parity/sweep.json`: `{"track_id": "sweep", "initial_state": {...}, "actions": [[steer, throttle], ...600], "python": [[x, y, psi], ...600]}`.

- [x] **Step 1: Write failing tests** in `tests/test_export.py`:
  - `test_site_data_matches_sources`: build `site.json` from the real sources; assert `stats["showcase_completion"] == pytest.approx(98.666, abs=0.01)`, `stats["held_out_completion"] == pytest.approx(85.2)`, `stats["parity_max_error"] < 1e-6`, both seeds present in `curves`, `snapshots[-1][0] == 6012928`.
  - `test_parity_demo_round_trips_fixture`: 600 actions and 600 python poses equal to the fixture's `steps[i]["action"]` and `steps[i]["state"]` x/y/psi.
  - Extend the existing full-bundle integrity test: manifest version `manifest.v2`; `[m["medal"] for m in medals][-1] == "author"`; every medal model file hash matches; each non-null medal lap on a track is ≥ the next medal's lap × 1.03.
- [x] **Step 2: Run** `uv run --locked pytest tests/test_export.py -v` → new tests FAIL.
- [x] **Step 3: Implement** `export/site.py` (`build_site_data() -> dict`, `build_parity_demo(fixture: Path) -> dict`) and extend `build_assets`: for each medal, load `<run_dir>/<checkpoint>`, verify its SHA against `experiments/medal-selection.json`, `export_policy` to `models/medal-<medal>.onnx` with the same validation samples, record deterministic replays to `replays/medal-<medal>/<track>.replay.v1.json` with `metadata.snapshot_id = "medal-<medal>"`; write `site.json` and `parity-demo.json`; bump the manifest version. Medal evaluation results are cached in `<evaluation_dir>/medals/<step>/` using the existing `fingerprint` scheme.
- [x] **Step 4: Run** `make export`, then `uv run --locked pytest tests/test_export.py -v` → PASS. Bundle grows by ~3 × 0.6 MB models plus replays.
- [x] **Step 5: Update** `docs/ASSETS.md` (manifest.v2 fields) and commit `feat: export medal opponents and site data in manifest v2` including `web/public/assets`.

### Task 3: Web asset loader for manifest.v2

**Files:**
- Modify: `web/src/assets/load.ts`, `web/tests/assets.test.ts`

**Interfaces:**
- Produces: `type MedalName = "bronze" | "silver" | "gold" | "author"`; `interface Medal { medal: MedalName; step: number; checkpoint_sha256: string; model: Asset; lap_times: Record<string, number | null>; replays: Record<string, Asset> }`; `interface SiteData` and `interface ParityDemo` mirroring Task 2's JSON; `Manifest` gains `medals: Medal[]; site: Asset; parity_demo: Asset`; `AssetLoader.site(): Promise<SiteData>`, `AssetLoader.parityDemo(): Promise<ParityDemo>`, `AssetLoader.medalReplay(medal: Medal, trackId: string)` (same checks as `replay`, identity = `medal-<name>` or `final` for author).

- [x] **Step 1: Write failing tests** in `web/tests/assets.test.ts`: `validateManifest` rejects `manifest.v1`; rejects medals not ordered bronze→author or missing `author`; rejects a medal whose `lap_times` lacks a manifest track key; accepts the real committed `web/public/assets/manifest.json`.
- [x] **Step 2: Run** `npm test --prefix web` → FAIL.
- [x] **Step 3: Implement** the types, validation and loader methods.
- [x] **Step 4: Run** `npm test --prefix web` and `npm run --prefix web typecheck` → PASS.
- [x] **Step 5: Commit** `feat: load manifest v2 medals and site data`.

### Task 4: Two-page build (intro + race)

**Files:**
- Create: `web/race.html` (the current `index.html` markup moved verbatim), `web/src/race/main.ts` (current `web/src/main.ts` moved), `web/src/intro/main.ts`, `web/src/intro/intro.css`
- Modify: `web/index.html` (becomes the intro skeleton), `web/vite.config.ts` (`build.rollupOptions.input = { intro: "index.html", race: "race.html" }`), `web/e2e/*.spec.ts` (race tests navigate to `/race.html`), `web/e2e/static.spec.ts`

**Interfaces:**
- `race.html#watch` opens directly in Watch-it-learn mode (main.ts reads `location.hash` at boot and calls `changeMode("ghosts")`).
- Intro skeleton element ids used by later tasks: `#hero`, `#hero-canvas`, `#stats`, `#ppo`, `#learning-chart`, `#parity`, `#parity-demo`, `#ship`.

- [x] **Step 1: Update e2e tests first**: race/viewer specs go to `/race.html`; static spec loads `/racing-rl/` and expects `h1` to contain "A CAR THAT TAUGHT ITSELF TO DRIVE", then clicks "Race the AI" and runs the existing race assertions on `/racing-rl/race.html`.
- [x] **Step 2: Run** `cd web && npm run test:static` → FAIL.
- [x] **Step 3: Move files and add the intro skeleton** with the three buttons: Race the AI → `./race.html`, Watch it learn → `./race.html#watch`, Draw a track → disabled with the label "Coming soon" (M9). Copy from the spec: headline "A car that taught itself to drive.", sub-copy "Reinforcement learning from scratch: a policy trained with my own PPO on 200 generated tracks, now racing live in your browser."
- [x] **Step 4: Run** `make test` and `cd web && npm run test:e2e && npm run test:static` → PASS.
- [x] **Step 5: Commit** `feat: split site into intro and race pages`.

### Task 5: Stadium track rendering and closer camera

**Files:**
- Create: `web/src/race/stadium.ts`, `web/tests/stadium.test.ts`
- Modify: `web/src/race/scene.ts` (`setTrack`, camera), `web/src/viewer/visual-config.ts`

**Interfaces:**
- Produces in `stadium.ts`: `kerbStripes(edge: Vec2[], stripe: number): { points: Vec2[]; red: boolean }[]` (alternating colours every `stripe` metres along the polyline, wrapping); `gateOrdinals(checkpointCount: number, every: number): number[]` (`[0, every, 2*every, …]`); `cameraZoom(speed: number): number` (pixels per metre).
- `visualConfig.stadium = { grass: [0x68a842, 0x5f9e3a], grassBand: 10, tarmac: 0x5d6470, kerb: [0xd6312c, 0xe9e4d6], kerbWidth: 1.2, kerbStripe: 3, gate: 0x2f7cf6, gateEvery: 4, zoom: { rest: 12, minimum: 6, speedScale: 60 } }`.

- [x] **Step 1: Write failing tests**: `kerbStripes` on a 30 m straight with stripe 3 → 10 stripes alternating starting red; on a closed loop the last stripe joins the first; `gateOrdinals(33, 4)` → `[0,4,…,32]`; `cameraZoom(0) === 12`, `cameraZoom(1000) === 6`, strictly non-increasing over 0..100 m/s.
- [x] **Step 2: Run** `npm test --prefix web` → FAIL.
- [x] **Step 3: Implement** the helpers (`cameraZoom = max(minimum, rest / (1 + speed / speedScale))`) and redraw `setTrack`: world-space grass bands (replacing the grid), kerbs on both edges, tarmac, white centre dashes, blue gate bars at `gateOrdinals` checkpoints (via `track.sample(arc)`), checkered start. Scene background becomes the darker grass colour. Race camera uses `cameraZoom`.
- [x] **Step 4: Run** `npm test --prefix web` → PASS; `cd web && npm run test:e2e` → PASS (frame-timing test still ≥57 FPS; if not, cache the static terrain with `cacheAsTexture` and note it in DECISIONS).
- [x] **Step 5: Commit** `feat: render tracks in the Stadium style`.

### Task 6: Formula cars and pre-race screen with medal cards

**Files:**
- Create: `web/src/race/car.ts`, `web/src/race/medals.ts`, `web/tests/medals.test.ts`
- Modify: `web/src/race/scene.ts` (use `formulaCar`), `web/src/race/main.ts`, `web/race.html`, `web/src/style.css`, `web/e2e/race.spec.ts`

**Interfaces:**
- `formulaCar(livery: Livery, renderer: Renderer): Sprite` in `car.ts`; `interface Livery { body: number; dark: number; light: number; accent: number; number: string }`; `LIVERIES: { player: Livery; ai: Livery }` (player body `0x3fe0b0`, number "07"; AI body `0xff7a2f`, number "AI"). Geometry follows `cars.html` option B (front wing, exposed tyres, narrow nose, cockpit, rear wing, nose number, soft shadow) at the physics car length; drawn once with `Graphics` and baked with `renderer.generateTexture` at 24 px/m with antialiasing, then used as `Sprite`s. Ghost fleet cars use the same function with the snapshot colour.
- `medalsForTrack(medals: Medal[], trackId: string): Medal[]` (drops `lap_times[trackId] === null`, keeps order); `MEDAL_COLORS: Record<MedalName, string>` (bronze `#c9824a`, silver `#c9d1d9`, gold `#f5c542`, author `#3fe0b0`).
- Pre-race screen markup in `race.html`: `#prerace` with `#track-cards` (one `button.track-card[data-track]` per track, showing minimap path, name, length, description) and `#medal-cards` (one `button.medal-card[data-medal]` per medal for the selected track, showing medal name and lap time `M:SS.mmm`), plus `#start-button`. The `<select>` sidebar is removed. Default selection: first track, Gold.

- [x] **Step 1: Write failing tests**: `medalsForTrack drops medals without a lap time`; `medalsForTrack keeps bronze→author order`.
- [x] **Step 2: Update e2e** `race.spec.ts`: wait for `.track-card` count 3 and `.medal-card` count ≥1 with one labelled "AUTHOR"; click track "technical" then medal cards rapidly bronze → gold → author → silver (if present, else bronze) and assert `#ai-label` shows the last clicked medal; existing countdown/pause/restart/respawn assertions unchanged; at 390 px no horizontal scroll.
- [x] **Step 3: Run** both suites → FAIL.
- [x] **Step 4: Implement** `car.ts`, `medals.ts`, the card UI and the `main.ts` state change (`selectedTrack: string`, `selectedMedal: MedalName`; the opponent policy is the medal's `model`, the reference ghost uses `loader.medalReplay`). Keep the existing `loadSerial` guard for late loads. Pre-race screen overlays the full-screen canvas and hides on Start; Escape-pause and the finish screen offer "Change race", which reopens it.
- [x] **Step 5: Run** `make test` and `cd web && npm run test:e2e` → PASS. Save screenshots `test-results/m7-prerace.png` and `test-results/m7-race.png` and look at them: cars must read as Formula cars at race zoom.
- [x] **Step 6: Commit** `feat: add Formula cars and medal opponent cards`.

### Task 7: Trackmania HUD and medal result

**Files:**
- Modify: `web/race.html` (HUD markup), `web/src/style.css`, `web/src/race/main.ts` (`updateHud`), `web/src/race/medals.ts`, `web/tests/medals.test.ts`, `web/e2e/race.spec.ts`

**Interfaces:**
- `earnedMedal(lap: number, medals: Medal[], trackId: string): MedalName | null` (the best medal whose lap time on the track is ≥ `lap`; ignores medals without a lap there).
- HUD: `#split` at top-centre (time + `#split-delta` with class `ahead` → background `#2f7cf6`, `behind` → `#d6312c`), updated only at gate checkpoints (`ordinal % visualConfig.stadium.gateEvery === 0`); `#lap-timer` bottom-centre; `#speed` bottom-right; heavy italic Barlow Condensed in black boxes. Result screen title: "GOLD MEDAL" (etc.) or "NO MEDAL", plus your lap vs the opponent's lap.

- [x] **Step 1: Write failing tests**: `earnedMedal(24.0, medals, "sweep")` with gold 24.25 / author 24.0 → `"author"`; lap slower than bronze → `null`; `earnedMedal ignores medals without a lap on the track`.
- [x] **Step 2: Extend e2e**: after driving past the first gate, `#split-delta` has class `ahead` or `behind` and its computed background colour is `rgb(47, 124, 246)` or `rgb(214, 49, 44)` respectively.
- [x] **Step 3: Run** → FAIL. **Step 4: Implement.** **Step 5: Run** `make test`, `npm run test:e2e` → PASS.
- [x] **Step 6: Commit** `feat: Trackmania-style HUD and medal results`.

### Task 8: Intro stats strip, sections and phone layout

**Files:**
- Modify: `web/index.html`, `web/src/intro/main.ts`, `web/src/intro/intro.css`
- Create: `web/src/intro/format.ts`, `web/tests/intro.test.ts`, `web/e2e/intro.spec.ts`

**Interfaces:**
- `formatStats(stats: SiteData["stats"]): { label: string; value: string }[]` in `format.ts`, returning exactly four tiles in this order and wording: `98.7%` "laps completed on showcase tracks"; `85.2%` "on 50 tracks it never saw"; `0 RL libs` "PPO, GAE, normalization by hand"; `5e-13` "Python ↔ browser physics error" (error formatted as `Xe-N`, one significant digit).
- Sections (copy from the spec's M7 intro bullets): `#ppo` (loop diagram as HTML: 64 parallel cars → rollout → GAE → clipped PPO update → repeat; seed table from `site.seeds` with the sentence that seed 1 regressed on one track under its deterministic mean action; link to `train/ppo.py` on GitHub), `#parity` (copy: "Trained in Python. Racing in your browser. Same car." and "Without this, the AI you race would drive worse than the one I trained."), `#ship` (CI, versioned/hashed assets, static deploy; plain links to the GitHub Actions page and repository, no badge image; **no test counts**, because a number baked into the page goes stale; this deviates from the spec's "test counts" and is logged in DECISIONS.md), footer with `joeycatic` and the GitHub link (LinkedIn omitted until the owner supplies it).

- [x] **Step 1: Write failing tests**: `formatStats` on the real `site.json` returns the four tiles above, value strings exactly `"98.7%"`, `"85.2%"`, `"0 RL libs"`, `"5e-13"`.
- [x] **Step 2: Write e2e** `intro.spec.ts`: `intro shows real stats` (four `.stat` tiles with those values); `intro works at phone width` (390×844: no horizontal scroll, the three buttons visible, Race the AI navigates to `race.html`); `intro survives a failed model load` (`page.route("**/*.onnx", r => r.abort())`: headline, stats and all sections still visible, no uncaught page errors).
- [x] **Step 3: Run** → FAIL. **Step 4: Implement** in the Stadium style (striped-grass hero, black stats strip, white sections, heavy italic headings) following `intro-page.html`. **Step 5: Run** → PASS.
- [x] **Step 6: Commit** `feat: intro page sections and stats from exported data`.

### Task 9: Interactive learning curve and live parity demo

**Files:**
- Create: `web/src/intro/chart.ts`, `web/src/intro/parity.ts`, `web/tests/parity-demo.test.ts`
- Modify: `web/src/intro/main.ts`, `web/src/intro/intro.css`, `web/e2e/intro.spec.ts`

**Interfaces:**
- `renderLearningChart(host: HTMLElement, site: SiteData): void`: inline SVG, two panels sharing the x-axis like `docs/images/learning-curve.svg` (return per seed; snapshot completion showcase vs held-out), legend for 2 seeds, crosshair + tooltip on hover/touch showing step and values, colours `#2a78d6`/`#eb6834`, single y-axis per panel.
- `replayParity(demo: ParityDemo, track: Track, config: SimulationConfig): { typescript: [number, number, number][]; maxError: number }`: runs a `DriveSession` from `initial_state` over `actions` (same sequence as `web/tests/parity.test.ts`); `maxError` = max absolute difference across x, y, psi. `renderParityDemo(host, demo, result, track)`: SVG of the track with the Python path and the TypeScript path overlaid and the measured `maxError` printed beneath, computed live on page load.

- [x] **Step 1: Write failing test**: `replayParity` on the committed `parity-demo.json` returns 600 poses and `maxError <= 1e-6`.
- [x] **Step 2: Extend e2e**: hovering the chart shows a tooltip containing "steps"; `#parity-demo` shows an error value matching `/\d(\.\d+)?e-\d+/`.
- [x] **Step 3: Run** → FAIL. **Step 4: Implement.** **Step 5: Run** → PASS.
- [x] **Step 6: Commit** `feat: interactive learning curve and live parity demo on intro`.

### Task 10: Live AI hero

**Files:**
- Create: `web/src/intro/attract.ts`
- Modify: `web/src/race/engine.ts` (`launch()`), `web/src/race/scene.ts` (camera target / player visibility), `web/src/intro/main.ts`, `web/tests/race.test.ts`, `web/e2e/intro.spec.ts`

**Interfaces:**
- `RaceEngine.launch(): void`: reset, then phase `"racing"` with no countdown.
- `RaceScene` gains `follow: "player" | "ai" = "player"` and `playerVisible = true`.
- `class AttractLoop { constructor(host: HTMLElement, loader: AssetLoader, config: SimulationConfig, track: Track); start(): Promise<void>; stop(): void }`: the Author medal drives Broad Sweep; restarts after each lap; pauses when the hero is off-screen (`IntersectionObserver`) or the tab is hidden; with `prefers-reduced-motion` it renders one static overview frame. Any failure leaves a static overview (CSS track illustration) and is logged with `console.warn` only.

- [x] **Step 1: Write failing test** in `race.test.ts`: `launch()` puts the engine in `"racing"` with `countdownTicks === 0` and the AI advances on the next `tick`.
- [x] **Step 2: Extend e2e**: `#hero-canvas canvas` exists and the label "LIVE · final AI lapping Broad Sweep" is visible; the existing failed-model test still passes (fallback).
- [x] **Step 3: Run** → FAIL. **Step 4: Implement.** **Step 5: Run** `make test`, `npm run test:e2e`, `npm run test:static` → PASS.
- [x] **Step 6: Commit** `feat: live AI lapping the track in the intro hero`.

### Task 11: Spec, docs and final verification

**Files:**
- Modify: `spec.md` (append M7–M9 to §9 from the design doc; amend §8.1 opponent picker, §8.5 visual style; note §8.2 is replaced in M8), `README.md` (new screenshot `docs/images/m7-race.png` and `docs/images/m7-intro.png` from the e2e runs; medals; site structure), `docs/DEVELOPMENT.md` (`make medals`, two-page build), `DECISIONS.md`

- [x] **Step 1:** Update the docs above; README image paths must exist.
- [x] **Step 2: Run** `make test`, `cd web && npm run test:e2e && npm run test:static`, `make web-build` → all green; note the measured FPS from the render test in DECISIONS.md.
- [x] **Step 3:** Look at the saved screenshots at 1440 px and 390 px for both pages; fix anything that reads as unpolished.
- [x] **Step 4: Commit** `docs: document M7 Stadium identity and intro page`.
