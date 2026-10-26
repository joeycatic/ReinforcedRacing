# M8: Progressive "Watch it learn" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the six-ghosts-at-once time-lapse with a ~60-second sequence: one car per training stage, played in training order. Each attempt leaves a faded trail, so the racing line visibly forms.

**Architecture:** Python evaluates the seed-2 snapshots and checkpoints on each showcase track (deterministic mean policy) and selects about 10 attempts per track. Each selected attempt must be visibly better than the last, and its replay is exported under `manifest.v3`. A new `watch.html` page plays them with a pure, tested timeline (pacing, captions, jump/skip) and a scene mode that draws trails plus exactly one moving car. Ghost mode is removed from `race.html`.

**Tech Stack:** Python 3.12 / uv / PyTorch (evaluation only); Vite + TypeScript + PixiJS 8; pytest, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-10-26-portfolio-v2-design.md` (section "M8") and `spec.md` §9 "M8". Read both alongside this plan.

## Global Constraints

- No change to `physics.v1`, `obs.v1`, `track.v1`, `replay.v1`, `heatmap.v1`; all parity fixtures pass unchanged. No retraining; no new ONNX models (attempts are replays only).
- Only `artifacts/training/m3-seed2` is used: snapshots `0, 50000, 200000, 1000000, 3000000, final` and the 36 files in `checkpoints/` (every 163,840 steps).
- No new npm/PyPI dependencies. No external network requests from the built site.
- Stadium look from M7 (`visualConfig.stadium`, Barlow Condensed heavy italic, HUD boxes `#000000cc`). Trail colours use the existing cool→warm `snapshotColor(index, count)`.
- Every caption number (steps, metres, lap time) comes from replay metadata. Hand-written headlines are allowed only after inspecting the replay, and are stored in `configs/attempts.yaml`.
- A full playthrough of each showcase track takes 50–70 s. Only one car moves at a time. The six-ghost simultaneous view is removed.
- Conventional commits, **no `Co-Authored-By` trailer**. Log builder's calls in `DECISIONS.md` under `# M8 decisions`.
- `make test` green after every task; `npm run test:e2e` and `npm run test:static` in `web/` green at the end.

## Review Focus

1. **Switching tracks while an attempt is playing or a replay is still loading.** The late replay must not appear on the new track (Task 5 e2e `rapid track switching shows only the last track`).
2. **Jumping to a segment, backwards or forwards.** Trails must be exactly the attempts before the selected one, never later ones (Task 3 test `trails are the attempts before the current one after any jump`).
3. **Hidden tab or reduced motion.** Playback pauses when the tab hides; with `prefers-reduced-motion` it starts paused on attempt 1 and never auto-advances (Task 5 e2e `reduced motion starts paused`).
4. **Phone width (390×844).** Canvas, caption and every training-progress segment fit with no horizontal scroll, and segments are tappable (Task 5 e2e `watch works at phone width`).
5. **One attempt's replay fails to load.** A notice says so and the page stays usable, not blank (Task 5 e2e `failed replay shows a notice`).

---

### Task 1: Attempt selection per showcase track

**Files:**
- Create: `eval/attempts.py`, `tools/select_attempts.py`, `tests/test_attempts.py`
- Create (generated, committed): `configs/attempts.yaml`, `experiments/attempt-selection.json`
- Modify: `Makefile` (target `attempts`)

**Interfaces:**
- Produces in `eval/attempts.py`:
  - `@dataclass(frozen=True) class Run: step: int; source: str; outcome: Literal["finished", "crashed", "stalled"]; progress: float; lap_time: float | None`. `source` is the snapshot id (`"0"`, `"50000"`, …, `"final"`) or `"checkpoint"`. `outcome` is `"stalled"` for a truncated episode.
  - `select_attempts(runs: list[Run], count: int = 10) -> list[Run]`: runs are in step order and the last one is `final`; the result is in step order.
  - `load_attempts(path: Path = ROOT / "configs/attempts.yaml") -> dict[str, list[AttemptChoice]]` with `@dataclass(frozen=True) class AttemptChoice: step: int; source: str; headline: str | None`.
- `configs/attempts.yaml` shape: `run_dir: artifacts/training/m3-seed2`, then `tracks: {sweep: [{step: 0, source: "0", headline: "Untrained policy"}, ...], ribbon: [...], technical: [...]}`.

- [ ] **Step 1: Write failing tests** in `tests/test_attempts.py`:
  - `test_score_orders_finishers_above_crashes`: `score(finished 25.0 s) > score(crashed at 800 m) > score(crashed at 100 m) > score(stalled at 40 m)`. `score(run) -> tuple` is a helper in `eval/attempts.py`: finished `(1, -lap_time)`, otherwise `(0, progress)`.
  - `test_selection_keeps_first_and_final_and_strictly_improves`: on a synthetic list of 40 runs, the result starts with the step-0 run, ends with `final`, and `score` strictly increases.
  - `test_selection_requires_visible_change`: a run only 5% further than the previous kept crash (and under 20 m) is skipped. A finished run must be ≥1% faster than the previous kept finisher.
  - `test_selection_skips_runs_faster_than_final`: a checkpoint lapping 23.88 s, when `final` laps 24.25 s, is not selected.
  - `test_selection_thins_to_count_keeping_milestones`: from a chain of 16 qualifying runs with count 10, exactly 10 come back. They include the first run, the first finisher and `final`, and the rest are spread evenly by chain index.
- [ ] **Step 2: Run** `uv run --locked pytest tests/test_attempts.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** `score` and `select_attempts`. The tests do not fully pin this algorithm:

```
final = runs[-1]; pool = [r for r in runs[:-1] if score(r) < score(final)]
chain = [pool[0]]
for r in pool[1:]:
    last = chain[-1]
    visible = (r.outcome == "finished" and (last.outcome != "finished" or r.lap_time <= last.lap_time * 0.99)) \
           or (r.outcome != "finished" and last.outcome != "finished"
               and r.progress >= max(last.progress * 1.10, last.progress + 20.0))
    if visible and score(r) > score(last): chain.append(r)
chain.append(final)
if len(chain) > count: keep first, first finisher, final; fill the remaining slots with
    chain indices from numpy.linspace(0, len(chain)-1, count), rounded, nearest unused index
```

- [ ] **Step 4: Run** the tests → PASS.
- [ ] **Step 5: Implement `tools/select_attempts.py`:**
  - Candidates are the six snapshots plus the 36 checkpoints, merged and sorted by step. Each is evaluated with `record_runs(checkpoint, make_track_pool("showcase"), load_export(), deterministic=True)`. For checkpoints, set `checkpoint.setdefault("requested_step", f"attempt-{step}")`, as `tools/select_medals.py` does.
  - Build one `Run` per track from `summary["episodes"]`.
  - Select per track, then write `configs/attempts.yaml` and `experiments/attempt-selection.json`. The JSON holds every candidate's run per track with `checkpoint_sha256`, plus the choices.
  - Each choice's default headline is computed: step 0 → "Untrained policy", first finisher → "First full lap", `final` → "Final policy". All others are `null`.
  - Add the Makefile target `attempts: $(UV) run --locked python -m tools.select_attempts $(ARGS)`.
- [ ] **Step 6: Run** `make attempts`. It prints a per-track table of step/outcome/progress/lap. Expect 7–12 attempts per track; if any track falls outside that, record the count and why in DECISIONS.md.
- [ ] **Step 7: Inspect and write headlines.** For each track, plot every selected replay's path (x/y from `eval.formats.decode_replay`) over the track to an SVG in the scratch directory, and look at what each attempt does. Write a short headline into `configs/attempts.yaml` only where the behaviour is unmistakable, e.g. "Found the throttle" (full speed, crashes at the first corner) or "First corners". Leave the rest `null`. Add `test_headlines_only_on_selected_steps`: every `headline` key in the yaml belongs to a selected step.
- [ ] **Step 8: Commit** `feat: select progressive learning attempts per showcase track`. Include the yaml/json and a DECISIONS.md entry with the selection rule (thresholds 10%/20 m and 1%, checkpoints faster than final skipped, chosen steps per track).

### Task 2: Export attempts in manifest.v3

**Files:**
- Modify: `export/assets.py`, `tests/test_export.py`, `docs/ASSETS.md`
- Regenerated (committed): `web/public/assets/**`

**Interfaces:**
- Consumes: `load_attempts()` (Task 1).
- Produces in `manifest.json`: `"version": "manifest.v3"` and a new key `"attempts": {track_id: [{"step": int, "source": str, "headline": str|null, "outcome": "finished"|"crashed"|"stalled", "progress": float, "lap_time": float|null, "replay": Asset}]}`, in step order.
  - Snapshot-sourced attempts reuse that snapshot's existing replay descriptor (no copy).
  - Checkpoint attempts are copied to `replays/attempt-<step>/<track>.replay.v1.json` with `metadata.snapshot_id = "attempt-<step>"`. Cache in `<evaluation_dir>/attempts/<step>/` with the existing `fingerprint` scheme, like medals.

- [ ] **Step 1: Write failing tests** in `tests/test_export.py`:
  - Extend the bundle integrity test: version `manifest.v3`; every showcase track has 7–12 attempts; first `step == 0` and last `source == "final"`; every attempt replay hash matches.
  - Each attempt's `outcome`/`progress`/`lap_time` must equal its replay's metadata (`finished` → `"finished"`, `crashed` → `"crashed"`, `truncated` → `"stalled"`).
  - `test_attempt_checkpoint_sha_matches_selection`: an attempt's replay `metadata.checkpoint_sha256` equals the value in `experiments/attempt-selection.json`.
- [ ] **Step 2: Run** `uv run --locked pytest tests/test_export.py -v` → FAIL.
- [ ] **Step 3: Implement** in `build_assets`. Verify each checkpoint's SHA against the selection evidence before recording, as medals do.
- [ ] **Step 4: Run** `make export` then the tests → PASS. Check the bundle growth with `du -sh web/public/assets`; expect under ~1.5 MB of new replays.
- [ ] **Step 5: Update** `docs/ASSETS.md` (manifest.v3 `attempts`) and commit `feat: export learning attempts in manifest v3`, including `web/public/assets`.

### Task 3: Watch timeline (pure TypeScript)

**Files:**
- Modify: `web/src/assets/load.ts`, `web/tests/assets.test.ts`
- Create: `web/src/watch/timeline.ts`, `web/tests/timeline.test.ts`

**Interfaces:**
- Loader: `interface Attempt { step: number; source: string; headline: string | null; outcome: "finished" | "crashed" | "stalled"; progress: number; lap_time: number | null; replay: Asset }`. `Manifest.attempts: Record<string, Attempt[]>`. `validateManifest` requires `manifest.v3`, and every manifest track to have a non-empty attempt list whose last `source` is `"final"`. `AssetLoader.attemptReplay(attempt: Attempt, trackId: string): Promise<{ data: Replay; frames: number[][] }>` checks the replay's `track_id`, and checks that its outcome/progress match the attempt.
- `timeline.ts`:
  - `interface AttemptRun { attempt: Attempt; data: Replay; frames: number[][] }`.
  - `interface Slot { index: number; rate: number; replaySeconds: number; displaySeconds: number; start: number }`, where `replaySeconds` is the replay time shown (stalled runs are cut).
  - `planTimeline(runs: AttemptRun[]): Slot[]` — pacing rules:
    - stalled runs show 3 s of replay at 1×;
    - crashed runs play at 1×;
    - finished non-final runs play at 3×;
    - `final` plays at 2×;
    - 1.5 s caption hold after every attempt.
    - Then, if the total is over 65 s, multiply the rates of all non-final, non-stalled runs by one common factor (capped at 8×) until the total is 60 s or the cap is reached.
  - `caption(run: AttemptRun, index: number, count: number): { kicker: string; headline: string | null; outcome: string }`:
    - kicker: `ATTEMPT 3 / 10 · 0.33M STEPS` (steps formatted like `snapshotLabel`);
    - outcome: `LAP 24.25 s`, `CRASHED AT 312 m` or `STALLED AT 43 m`, rounded from metadata;
    - headline: the attempt's headline.
  - `class WatchTimeline`:
    - `constructor(runs: AttemptRun[])`;
    - fields `seconds`, `playing`, `slots`, `duration`;
    - `play()`, `pause()`, `replay()`, `skipToFinal()`, `jump(index: number)`, `advance(realSeconds: number)`;
    - `current(): { index: number; replaySeconds: number; holding: boolean; frame: number[] }`;
    - `trails(): number[]`: indices of completed attempts, i.e. those before `current().index`.
    - Playback stops at the end, not looping.

- [ ] **Step 1: Write failing tests** in `web/tests/timeline.test.ts`:
  - `planTimeline cuts stalls to 3 s`: a 200 s stalled run gets `replaySeconds === 3`.
  - `final plays at 2×`.
  - `crashes play at 1× when the total fits`.
  - `real attempts fit 50–70 s`: read `web/public/assets/manifest.json` and the attempt replays from disk (as `parity.test.ts` reads fixtures); for each of the three tracks, the last slot's `start + displaySeconds` lies in [50, 70].
  - `captions match replay outcomes`: for every real attempt, `caption().outcome` is derived from replay metadata (`CRASHED AT ${Math.round(progress)} m` etc.).
  - `trails are the attempts before the current one after any jump`: jump 7 → `trails()` is `[0..6]`; jump 2 → `[0, 1]`; `skipToFinal()` → all but the last.
  - `only one attempt is current at any time`: sampling `current()` every 0.25 s over the duration never yields two indices at one time, and indices never decrease while playing.
  - Loader: `validateManifest` rejects `manifest.v2`, rejects attempts not ending in `final`, and accepts the committed manifest.
- [ ] **Step 2: Run** `npm test --prefix web` → FAIL.
- [ ] **Step 3: Implement** the loader changes and `timeline.ts`, reusing `replayFrame` and `snapshotLabel` from `web/src/viewer/playback.ts`. Those two helpers move into `timeline.ts`; `GhostPlayback` is deleted in Task 5.
- [ ] **Step 4: Run** `npm test --prefix web` and `npm run --prefix web typecheck` → PASS.
- [ ] **Step 5: Commit** `feat: pace learning attempts into a one-minute timeline`.

### Task 4: Scene support for one car plus trails

**Files:**
- Modify: `web/src/race/scene.ts`, `web/src/viewer/visual-config.ts`, `web/src/watch/timeline.ts` (read numbers from `visualConfig.watch`)
- Test: `web/e2e/watch.spec.ts` (created here, extended in Task 5). Vitest runs in Node without a renderer, so scene checks use the existing `__PIXI_APP_INIT__` hook pattern from `web/e2e/viewer.spec.ts` ("Watch Formula sprites keep their texture-to-world scale").

- Create: `web/watch.html`, `web/src/watch/main.ts` (minimal mount), plus `watch: "watch.html"` in `web/vite.config.ts`

**Interfaces:**
- Consumes: `AttemptRun`, `WatchTimeline` (Task 3).
- `RaceScene.setAttempts(runs: AttemptRun[]): void`:
  - Builds one trail `Graphics` per run: the full replay path, sampled every 6 ticks, stalled runs only up to their cut, 0.35 m wide, colour `snapshotColor(i, n)`, alpha `visualConfig.watch.trailAlpha`.
  - Builds one `formulaCar` sprite per livery colour; all start hidden.
- `RaceScene.renderAttempt(timeline: WatchTimeline): void`:
  - Shows trails for `timeline.trails()`, plus the current run's partial trail up to its current tick.
  - Shows exactly one car at `current().frame`, coloured for the current index (hidden while `holding`).
  - Uses the overview camera (`fitZoom`). Car sprites are enlarged like overview ghosts today (`visualConfig.ghost.minimumPixelsPerMeter`).
- Remove `setGhosts` and `renderGhosts`.
- `visualConfig.watch = { trailAlpha: 0.45, trailWidth: 0.35, trailEveryTicks: 6, holdSeconds: 1.5, stallSeconds: 3, maxRate: 8 }`. `timeline.ts` reads its numbers from here.

- [ ] **Step 1: Write failing e2e tests** in `web/e2e/watch.spec.ts`. Use a minimal `watch.html` that only mounts the scene and timeline on sweep, and that Task 5 fleshes out. Capture the Pixi app via `__PIXI_APP_INIT__` and count visible textured sprites, as the existing viewer spec does:
  - `renderAttempt shows exactly one visible car`: exactly one visible car sprite, 6–60 px wide;
  - `renderAttempt shows trails only for earlier attempts`: after clicking segment 4, the visible trail `Graphics` labelled `trail-<i>` are exactly 0..3 plus the current partial trail;
  - `holding hides the car`: during a caption hold, zero car sprites are visible.
  Give trails `label = "trail-<i>"` and cars `label = "attempt-car"` so the tests can find them.
- [ ] **Step 2: Run** `cd web && npx playwright test e2e/watch.spec.ts` → FAIL. **Step 3: Implement.** **Step 4: Run** it again and `npm test --prefix web` → PASS.
- [ ] **Step 5: Commit** `feat: draw learning attempts as one car with fading trails`.

### Task 5: Watch page and removal of ghost mode

**Files:**
- Create: `web/src/watch/watch.css`
- Modify (started in Task 4): `web/watch.html`, `web/src/watch/main.ts`, `web/e2e/watch.spec.ts`
- Modify: `web/index.html` (Watch it learn → `./watch.html`), `web/race.html` and `web/src/race/main.ts`:
  - remove `#mode-ghosts`, `#ghost-sidebar`, `#playback-controls` and `changeMode`;
  - `#watch` → `location.replace("./watch.html")`;
  - add a header link to Watch it learn.
- Also modify `web/src/viewer/controller.ts` (heatmaps only; delete ghost code), delete `web/src/viewer/playback.ts` (move `snapshotColor` into `timeline.ts`), update `web/tests/viewer.test.ts` (drop the `GhostPlayback` tests, keep the heatmap/colour/label tests importing from `timeline.ts`), and modify `web/e2e/viewer.spec.ts` (keep the heatmap and AI-vision assertions on the race page; drop the ghost assertions) and `web/e2e/static.spec.ts` (also load `/racing-rl/watch.html`).

**Interfaces:**
- Consumes: Task 3 loader/timeline, Task 4 scene.
- Page element ids:
  - `#watch-tracks`: three `button.track-card[data-track]`, default `sweep`;
  - `#watch-canvas`;
  - `#attempt-kicker`, `#attempt-headline`, `#attempt-outcome`;
  - `#lap-timer`: shown while a finished attempt plays, showing replay time;
  - `#progress-bar`: one `button.segment[data-attempt=i]` per attempt; width ∝ `displaySeconds`; `aria-label` = kicker + outcome; class `done`/`current`;
  - `#watch-play` (aria-pressed), `#watch-replay`, `#watch-skip`;
  - `#crash-toggle` (off by default);
  - `#notice`.
- Crash heatmap: the replays do not carry heatmaps, so the toggle shows the stochastic heatmap of the nearest snapshot at or below the current attempt's step. It is labelled `CRASHES · 1M-STEP STAGE · 50 RUNS`, and the stage changes as attempts advance. Log in DECISIONS.md: this keeps the bundle to replays only, as the spec requires.
- Late loads use a revision counter, as `ViewerController` does today. Hiding the tab pauses playback. `prefers-reduced-motion` starts paused.

- [ ] **Step 1: Write e2e** `watch.spec.ts`:
  - `plays attempts one at a time`: on load, the kicker reads `ATTEMPT 1 / N`. After clicking segment 3, the kicker reads `ATTEMPT 4 / N` and segments 0–2 have class `done`. Skip → kicker shows the last attempt and the outcome matches `/LAP \d+\.\d\d s/`.
  - `rapid track switching shows only the last track`: click ribbon → technical → sweep quickly; after settling, `#attempt-kicker` corresponds to sweep's attempt count, and no page errors occur.
  - `reduced motion starts paused`: `page.emulateMedia({ reducedMotion: "reduce" })`; after 1 s, `#watch-play` has aria-pressed false and the kicker still reads attempt 1.
  - `watch works at phone width`: 390×844, no horizontal scroll, every segment visible and clickable.
  - `failed replay shows a notice`: `page.route("**/replays/attempt-*/**", r => r.abort())`; `#notice` becomes visible and the track buttons remain enabled.
  - `full playthrough takes 50-70 s` (test timeout 120 s): on sweep at normal speed, measure the time from play until `#watch-play` returns to "Play" at the end.
  - Race page: `race.html#watch` redirects to `watch.html`; `#mode-ghosts` no longer exists.
- [ ] **Step 2: Run** `cd web && npm run test:e2e` → FAIL.
- [ ] **Step 3: Implement** the page in the Stadium style, following `.superpowers/brainstorm/135740-1793034399/content/watch-it-learn.html`: overview canvas, caption box above the bar, black training-progress bar with labels "first attempt" / "final policy". Then remove ghost mode from the race page.
- [ ] **Step 4: Run** `make test`, `cd web && npm run test:e2e && npm run test:static` → PASS. Save `test-results/m8-watch.png` mid-sequence and look at it: earlier trails must be visibly fainter and cooler than the current car.
- [ ] **Step 5: Commit** `feat: progressive Watch it learn page replaces ghost time-lapse`.

### Task 6: Spec, docs and final verification

**Files:**
- Modify:
  - `spec.md`: §8.2 now points to M8; mark M8 delivered in §9.
  - `README.md`: replace the "Watch it learn" description; screenshot `docs/images/m8-watch.png` from the e2e run.
  - `docs/DEVELOPMENT.md`: `make attempts`; the three-page build.
  - `DECISIONS.md`: `# M8 decisions`, covering selection rule, chosen steps, pacing numbers, heatmap stage mapping and measured playthrough times per track.

- [ ] **Step 1:** Update the docs; README image paths must exist.
- [ ] **Step 2: Run** `make test`, `make web-build`, `cd web && npm run test:e2e && npm run test:static` → all PASS.
- [ ] **Step 3:** Commit `docs: document M8 progressive Watch it learn`.
- [ ] **Step 4:** Summarize per AGENTS.md: what was built, how to run it, open issues. Stop; M9 has its own plan.
