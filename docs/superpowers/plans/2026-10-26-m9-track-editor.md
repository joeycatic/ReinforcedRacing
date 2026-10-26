# M9: Track Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** M8 (`docs/superpowers/plans/2026-10-26-m8-progressive-watch.md`) is finished, committed and green. The owner asked for M8 and M9 to run back to back; start M9 only after M8's final verification passes.

**Goal:** Let a visitor draw a track freehand, with crossings becoming overpasses. Then the trained final policy drives it, the visitor races it, and the track can be shared by URL.

**Architecture:**
- Python remains the simulation's source of truth. It gains an editor track builder (control points → spline → `build_track`) and crossing detection (`track.v2`), plus a raycast rule: inside a crossing zone, rays ignore the other branch's boundary segments.
- TypeScript ports exactly that and is parity-tested on generated fixtures. Tracks without crossings take an unchanged code path, so all existing fixtures and observations are identical.
- A new `editor.html` page turns a stroke into control points, validates live, runs the AI lap with the existing `RaceEngine`, and hands the track to `race.html` via the URL.

**Tech Stack:** Python 3.12 / uv / NumPy; Vite + TypeScript + PixiJS 8 + onnxruntime-web; pytest, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-10-26-portfolio-v2-design.md` (section "M9") and `spec.md` §9 "M9". Read both alongside this plan.

## Global Constraints

- `physics.v1`, `obs.v1`, `replay.v1`, `heatmap.v1` unchanged. Tracks without crossings keep `track.v1` byte-for-byte: `Track.to_dict()` emits `track.v1` when there are no crossings and `track.v2` only when there are. `VERSIONS.track` in the manifest stays `track.v1`. Python and TS loaders accept `track.v2` only for editor tracks.
- **The three existing parity fixtures (`sweep.json`, `ribbon.json`, `technical.json`) must regenerate byte-identical.** `make parity && git diff --exit-code tests/fixtures/parity/{sweep,ribbon,technical}.json` passes. `simulation_config()` must not change; editor settings live in a new `configs/editor.yaml`.
- No retraining; the AI is the shipped `final` ONNX model. No new dependencies. No server: share links encode the track in the URL fragment.
- Editor values (owner's spec plus builder's calls, all in `configs/editor.yaml`):
  - road and size: width **12 m** (training range 10–14 m); length **700 m** (training pool lengths measured 575–749 m); **24** control points; spline 40 samples per span; resample spacing 1.0 m; checkpoints every 25 m;
  - crossings: minimum crossing angle **30°**; any two crossing branches ≥ **80 m** apart along the track (more than 2 × `projection_window` 24 m); crossings ≥ **60 m** from the start line; crossing zone half-length `width / sin(angle) + 2.0` m;
  - warnings and sharing: warn above curvature **0.10** (`max_curvature`); control points quantised to **0.1 m**.
- Default bridge: the later pass over a crossing (branch 1). Clicking a crossing flips it. `over` affects rendering and the share code only, never simulation.
- Stadium look from M7. Phones can draw, submit and watch the AI lap. Racing shows the existing keyboard/gamepad notice.
- Conventional commits, **no `Co-Authored-By` trailer**. Log builder's calls in `DECISIONS.md` under `# M9 decisions`.
- `make test` green after every task; `npm run test:e2e` and `npm run test:static` in `web/` green at the end.

## Review Focus

1. **A crossing drawn near the start line or near another crossing.** It must block Submit with a reason, never reach the simulation (Task 1 fixture `near_start`, Task 4 parity of issue codes).
2. **Drawing that never closes, or a scribble.** A tiny stroke, an open stroke or a dense scribble must show a reason and keep Submit disabled without freezing the page. A rebuild stays under 50 ms (Task 5 tests `open stroke is not-closed`, `scribble is blocked`, `rebuild is fast`).
3. **A tampered or truncated share link.** It must show "This track link is damaged" and an empty editor, not throw (Task 5 test `decode rejects corrupt codes`; Task 7 e2e `bad share link`).
4. **The AI crashes or never finishes on a drawn track.** The result must say "crashed at N m" (with the tight-corner reason when it applies) or "did not finish in 90 s", and Race it must still work (Task 6 tests; Task 7 e2e with a tight track).
5. **Clicking a crossing to flip it, then sharing.** The shared link reproduces the flip; flipping never changes AI observations (Task 5 round-trip test with flips; Task 2 test `over does not change observations`).

---

### Task 1: Python crossings, editor builder and track.v2

**Files:**
- Create: `sim/crossings.py`, `sim/editor.py`, `configs/editor.yaml`, `tools/editor_shapes.py`, `tests/test_editor.py`
- Modify: `sim/__init__.py` (`TRACK_V2 = "track.v2"`), `sim/geometry.py`, `sim/track.py`, `sim/config.py` (`EditorConfig`, `load_editor()`)

**Interfaces:**
- `sim/geometry.py`: `polyline_intersections(a: FloatArray, b: FloatArray | None = None) -> list[tuple[int, int, FloatArray]]`. Same candidate and adjacency rules as `polylines_intersect`, returning segment indices and the intersection point (for collinear overlap, the midpoint of the overlap).
- `sim/track.py` defines `@dataclass(frozen=True) class Crossing` (kept there to avoid a circular import; `sim/crossings.py` imports it):
  - `Crossing: arcs: tuple[float, float]; point: tuple[float, float]; angle: float; over: int; hidden: tuple[tuple[int, ...], tuple[int, ...]]`.
    - `arcs[0] < arcs[1]`; `angle` is in degrees, between 0 and 90.
    - `hidden[b]` lists the boundary segment ids of branch `b` in BoundaryGrid order (left `i`, right `n + i`). These are the segments whose centreline index `i` has circular arc distance from `arcs[b]` ≤ `width / sin(angle) + zone_margin`.
  - `find_crossings(track: Track, config: EditorConfig, over: Sequence[int] | None = None) -> tuple[Crossing, ...]`: centreline self-intersections, with pairs whose points lie within 1 m merged into one crossing; ordered by `arcs[0]`; `over` defaults to all 1.
  - `branch(crossing: Crossing, arc: float, length: float) -> int`: whichever of `arcs` is circularly nearer.
  - `hidden_segments(track: Track, arc: float) -> frozenset[int]`: for each crossing, the other branch's `hidden`.
- `sim/editor.py`:
  - `build_editor_track(control: FloatArray, config: EditorConfig, over: Sequence[int] | None = None) -> Track`. Steps:
    1. scale the control points about their mean, so the closed control polygon's perimeter equals `config.length`;
    2. round to `config.quantum`;
    3. `catmull_rom(control, samples_per_span)`;
    4. `resample_closed(…, sample_spacing)`;
    5. `build_track(…, width, checkpoint_spacing, track_id="custom", name="Your track")`;
    6. attach `find_crossings`.

    The first control point stays the start line.
  - `editor_issues(track: Track, config: EditorConfig) -> list[str]`: sorted unique codes from `fold`, `overlap`, `shallow-crossing`, `crossing-spacing`, `crossing-start`. A boundary or centreline intersection that is not inside one crossing's hidden zones (one segment of each branch) is `overlap`.
  - `tight_sections(track: Track, limit: float) -> list[tuple[float, float]]`: merged arc ranges where `|curvature| > limit`.
- `sim/track.py`:
  - `Track` gains `crossings: tuple[Crossing, ...] = ()`.
  - `to_dict()` emits `track.v2` with `"crossings": [{"arcs", "point", "angle", "over", "hidden"}]` only when `crossings` is non-empty.
  - `from_dict` accepts `track.v2`: rebuild, recompute crossings with the supplied `over`, check they match within 1e-9; then validate with `editor_issues` instead of `validity_errors`.
- `tools/editor_shapes.py`: `shapes() -> dict[str, tuple[FloatArray, list[str]]]` maps a name to (world control points, expected issue codes):
  - `loop` → `[]`;
  - `figure8` → `[]`: a Bernoulli lemniscate sampled at 24 points with its first point at a lobe tip, so it crosses once at 90°;
  - `pretzel` → `[]`: two valid crossings;
  - `shallow` → `["shallow-crossing"]`;
  - `near_start` → `["crossing-start"]`;
  - `overlap` → `["overlap"]`: two parallel straights about 8 m apart;
  - `fold` → `["fold"]`;
  - `tight` → `[]` with non-empty `tight_sections`: curvature between 0.10 and 1/6.

- [ ] **Step 1: Write failing tests** in `tests/test_editor.py`:
  - `test_shapes_report_expected_issues`: for every shape, `editor_issues(build_editor_track(control, cfg), cfg) == expected`.
  - `test_figure8_has_one_right_angle_crossing`: one crossing, angle 90 ± 2, branch arcs ≥ 80 m apart, `over == 1`.
  - `test_hidden_zones_cover_all_crossing_boundary_hits`: on `figure8` and `pretzel`, every `polyline_intersections` hit between boundaries lies on segments of opposite branches of one crossing's `hidden`.
  - `test_track_length_and_width_in_training_range`: for `loop` and `figure8`, length is within 700 ± 35 and width is 12.
  - `test_v1_round_trip_unchanged`: every showcase track's `to_dict()["version"] == "track.v1"` and `"crossings" not in to_dict()`.
  - `test_v2_round_trip`: `Track.from_dict(figure8.to_dict())` gives equal crossings; a tampered `hidden` raises `ValueError`.
  - `test_editor_yaml_matches_web_config`: the shared keys of `configs/editor.yaml` equal `web/src/editor/editor-config.json` (the file is created in Task 5; mark this test `xfail(strict=True)` until then, and remove the mark in Task 5).
- [ ] **Step 2: Run** `uv run --locked pytest tests/test_editor.py -v` → FAIL.
- [ ] **Step 3: Implement** the modules above. Implement `polyline_intersections` with the same orientation tests as `polylines_intersect`, so both agree on what counts as touching.
- [ ] **Step 4: Run** the new tests and the full `uv run --locked pytest` → PASS. Existing track and validity tests are unchanged.
- [ ] **Step 5: Commit** `feat: add editor track builder and crossing detection (track.v2)`.

### Task 2: Python raycasts ignore the other branch at a crossing

**Files:**
- Modify: `sim/raycast.py`, `sim/pool.py`, `sim/observation.py`, `tests/test_observation.py`, `tests/test_editor.py`

**Interfaces:**
- `BoundaryGrid` gains `tags: IntArray`, one per segment slot including the sentinel. 0 means not in any zone; `1 + k` is the `k`-th (crossing, branch) pair across all tracks in the pool. It also gains `tag_count: int`.
- `BoundaryGrid.cast(track_ids, origins, directions, max_distance, hidden: NDArray[np.bool_] | None = None)`:
  - `hidden` has shape `(N, tag_count + 1)`, with column 0 always False.
  - A candidate segment is skipped when `hidden[car, tags[segment]]`.
  - `None` means nothing is hidden, and the code path must leave every float operation unchanged.
- `TrackPool.hidden_tags(track_ids: IntArray, arcs: FloatArray) -> NDArray[np.bool_] | None` returns `None` when the pool has no crossings.
- `ObservationBuilder.build` passes `self.pool.hidden_tags(track_ids, arcs)` to `cast`. The signature is unchanged.

- [ ] **Step 1: Write failing tests:**
  - `test_crossing_rays_see_no_phantom_wall`: on `figure8`, place a car on branch 0 exactly at the crossing point, heading along branch 0's tangent. Build its observation through `TrackPool([track])` and `ObservationBuilder`:
    - the forward ray (index 4) × `ray_max_distance` ≥ 25 m;
    - the ±90° rays (indices 1 and 7) are 6.0 ± 0.3 m (own walls).
    - Repeat for branch 1.
  - `test_crossing_progress_stays_on_branch`: drive `ProgressTracker` along branch 0's centreline through the crossing in 1 m steps. `projection.distance < 0.01` and the arc increases monotonically through the zone. Repeat for branch 1.
  - `test_over_does_not_change_observations`: same car, `over=(0,)` vs `(1,)` → identical observation arrays.
  - `test_pool_without_crossings_hides_nothing`: `make_track_pool("showcase").hidden_tags(...)` returns `None` and `grid.tag_count == 0`. Byte-identical observations on existing tracks are proven by the fixture check in Step 5.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `uv run --locked pytest` → PASS.
- [ ] **Step 5: Verify fixtures untouched:** `make parity && git diff --exit-code tests/fixtures/parity/` → no diff.
- [ ] **Step 6: Commit** `feat: hide the crossing branch's walls from raycasts`.

### Task 3: Parity fixtures for the editor and crossings

**Files:**
- Modify: `tools/generate_parity.py`, `tests/test_export.py` (currency test covers the new files)
- Create (generated, committed): `tests/fixtures/crossing/figure8.json` (a separate directory: `web/tests/parity.test.ts` loads every file in `parity/`, and TS cannot read track.v2 until Task 4), `tests/fixtures/editor/<shape>.json` for every shape in `tools/editor_shapes.py`

**Interfaces:**
- `figure8.json` has the same schema as the existing parity fixtures, built on the `figure8` editor track: 600 steps, progress probes, ray probes. It adds `"crossing_probes": [{"state": CarState, "arc": float, "index": int, "offset": float, "observation": [22 floats]}]`: 40 probes, 20 per branch, at arcs from −40 m to +40 m around each crossing branch, with headings along the tangent ± 0.3 rad.
- `tests/fixtures/editor/<shape>.json`: `{"control": [[x, y], ...], "over": [int, ...] | null, "config": <editor.yaml dict>, "track": Track.to_dict(), "issues": [codes], "tight": [[start, end], ...]}`.
- `generate_parity.py` gains `editor_fixtures() -> dict[str, dict]` and writes both directories. The existing `fixtures()` output for the three showcase tracks must not change.

- [ ] **Step 1: Write failing test** `test_generated_editor_fixtures_are_current` (same tolerance helper as `test_generated_fixtures_are_current`), and a matching `test_generated_crossing_fixture_is_current` for `crossing/figure8.json`.
- [ ] **Step 2: Run** → FAIL (files missing). **Step 3: Implement** and run `make parity`.
- [ ] **Step 4: Run** `uv run --locked pytest`, then `git diff --exit-code tests/fixtures/parity/{sweep,ribbon,technical}.json` → PASS, no diff.
- [ ] **Step 5: Commit** `test: add crossing and editor-builder parity fixtures`.

### Task 4: TypeScript port with parity tests

**Files:**
- Create: `web/src/sim/geometry.ts`, `web/src/sim/crossings.ts`, `web/src/editor/build.ts`, `web/tests/editor-build.test.ts`
- Modify: `web/src/sim/types.ts` (`CrossingData`, `TrackData.crossings?`), `web/src/sim/track.ts`, `web/src/sim/raycast.ts`, `web/src/sim/observation.ts`, `web/src/assets/versions.ts` (`TRACK_V2 = "track.v2"`), `web/tests/parity.test.ts`

**Interfaces:**
- `geometry.ts`: `catmullRom(control: Vec2[], samplesPerSpan: number): Vec2[]`, `resampleClosed(points: Vec2[], spacing: number): Vec2[]`, `polylineIntersections(a: Vec2[], b?: Vec2[]): [number, number, Vec2][]`. These mirror Python, including `np.interp` and `np.arange` endpoint semantics.
- `crossings.ts`: `interface Crossing { arcs: [number, number]; point: Vec2; angle: number; over: number; hidden: [number[], number[]] }`, `findCrossings(data: TrackData, config: EditorConfig, over?: number[]): Crossing[]`, `branchOf(c: Crossing, arc: number, length: number): 0 | 1`, `hiddenSegments(track: Track, arc: number): Set<number>`.
- `build.ts`: `interface EditorConfig` (fields of `editor-config.json`), `buildTrackData(centerline: Vec2[], width: number, checkpointSpacing: number, id: string, name: string): TrackData` (port of `build_track`), `buildEditorTrack(control: Vec2[], config: EditorConfig, over?: number[]): { track: Track; issues: string[]; tight: [number, number][] }`, `editorIssues(...)`, `tightSections(...)`.
- `Track` constructor accepts `track.v1`, and `track.v2` with `crossings`. `BoundaryGrid` gains `tags: number[]`. `cast(origin, direction, maxDistance, hidden?: ReadonlySet<number>)` skips segments whose id is in `hidden`. `ObservationBuilder.build` passes `hiddenSegments(track, p.arc)` only when the track has crossings.

- [ ] **Step 1: Write failing tests:**
  - `editor-build.test.ts`, for every `tests/fixtures/editor/*.json`: `buildEditorTrack(control, config, over)` yields track data matching `fixture.track` (numbers ≤ 1e-9, integers and strings exact), `issues` equal, and `tight` equal within 1e-9.
  - `rebuild is fast`: `figure8` builds in < 50 ms averaged over 20 runs in Node.
  - `parity.test.ts` also loads `tests/fixtures/crossing/*.json`; when a fixture has `crossing_probes`, `ObservationBuilder.build(probe.state, {index, arc, offset, distance: 0}, previous action from the probe)` matches `probe.observation` within the existing 1e-6 bound. `figure8.json` passes the existing 600-step test unchanged.
- [ ] **Step 2: Run** `npm test --prefix web` → FAIL.
- [ ] **Step 3: Implement.** Use a uniform grid bucket (cell = 2 × width) in `polylineIntersections` so a 700-point track stays well under 50 ms. Python can stay brute-force.
- [ ] **Step 4: Run** `npm test --prefix web`, `npm run --prefix web typecheck` → PASS, with the existing three fixtures' max error still ≤ the recorded 5.4e-13.
- [ ] **Step 5: Commit** `feat: port editor builder and crossing raycasts to TypeScript`.

### Task 5: Stroke to control points, issues UI model and share codes

**Files:**
- Create: `web/src/editor/editor-config.json`, `web/src/editor/stroke.ts`, `web/src/editor/share.ts`, `web/src/editor/issues.ts`, `web/tests/editor.test.ts`
- Modify: `tests/test_editor.py` (remove the `xfail` from `test_editor_yaml_matches_web_config`)

**Interfaces:**
- `editor-config.json` holds every key of `configs/editor.yaml`, plus `"ui": {"closeDistancePx": 40, "minStrokePx": 300, "rebuildMs": 100}`.
- `stroke.ts`: `strokeToControls(points: Vec2[], config: EditorConfig): { control: Vec2[] | null; problem: "too-short" | "not-closed" | null }`.
  - A stroke shorter than `minStrokePx` is `too-short`.
  - An end farther than `closeDistancePx` from the start is `not-closed`.
  - Otherwise: drop the end segment that overlaps the start; take `control_points` points evenly by arc length along the closed stroke, starting at the first point; flip y (screen-down → world-up).
  - Scaling and quantisation happen in `buildEditorTrack`.
- `share.ts`: `encodeTrack(control: Vec2[], over: number[]): string` → `"1." + base64url(varint stream)`. The stream is `[controlCount, crossingCount, overBits…, zigzag deltas of round(x / quantum), round(y / quantum)…]`. `decodeTrack(code: string): { control: Vec2[]; over: number[] } | null` returns `null` for any malformed, truncated or out-of-range input.
- `issues.ts`: `MESSAGES: Record<string, string>` with these exact strings:
  - `not-closed`: "Finish near the start to close the loop"
  - `too-short`: "Draw a bigger loop"
  - `fold`: "The road folds over itself in a corner"
  - `overlap`: "Roads overlap — cross them clearly or keep them apart"
  - `shallow-crossing`: "Roads cross at too shallow an angle"
  - `crossing-spacing`: "Crossings are too close together"
  - `crossing-start`: "A crossing is too close to the start line"
  - `tight`: "Too tight — the AI never trained on corners this sharp"

  `canSubmit(problem, issues): boolean` is true only when there is no problem and no issue codes (tight is a warning only).

- [ ] **Step 1: Write failing tests** in `web/tests/editor.test.ts`:
  - `open stroke is not-closed`; `tiny stroke is too-short`.
  - `closed circle stroke gives 24 controls starting at the first point`.
  - `scribble is blocked`: a dense zig-zag stroke gives a non-empty issues list and `canSubmit === false`.
  - `share round trip reproduces the track`: for `figure8` and `pretzel` controls with flipped `over`, `buildEditorTrack(decode(encode(c, o)))` deep-equals `buildEditorTrack(c, o)`.
  - `decode rejects corrupt codes`: wrong prefix, bad base64, truncated, oversized counts, `over` bit count mismatch → `null`.
  - `tight warns but allows submit`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `make test` → PASS (including the un-xfailed Python config test).
- [ ] **Step 5: Commit** `feat: stroke capture, editor issues and shareable track codes`.

### Task 6: Bridges in the scene and the AI trial lap

**Files:**
- Create: `web/src/editor/trial.ts`, `web/tests/trial.test.ts`
- Modify: `web/src/race/scene.ts`, `web/src/race/stadium.ts`, `web/tests/stadium.test.ts`

**Interfaces:**
- `stadium.ts`: `carAboveBridges(track: Track, arc: number): boolean`. True when, for some crossing, the car's branch is `over` and the arc is within that branch's hidden-zone half-length.
- `RaceScene.setTrack` draws each crossing's upper-branch zone as a separate bridge layer above the cars layer: tarmac, kerbs and a 0.35-alpha drop shadow offset 1.5 m. Each frame, cars where `carAboveBridges` is true are drawn in a top layer above the bridges; others stay beneath, so a car on the lower branch passes under. Tracks without crossings create no bridge layer.
- `trial.ts`: `class TrialLap { constructor(engine: RaceEngine, tight: [number, number][], limitSeconds = 90); tick(): Promise<void>; result: TrialResult | null }`.
  - `type TrialResult = { kind: "lap"; time: number } | { kind: "crashed"; at: number; tight: boolean } | { kind: "timeout" }`.
  - It uses `engine.launch()` and advances `engine.tick([0, 0])` (the player stays parked and hidden). It stops at the first AI crash (`engine.ai.crashes > 0`; `at` = `engine.ai.progress.unwrapped` rounded to metres), at the first AI lap, or after `limitSeconds` of simulated time.
  - `tight` is true when `at` lies within 15 m of a tight section.
  - `describe(result): string`: `LAP 0:31.482`, `CRASHED AT 412 M` (plus " — this corner is tighter than anything in training" when tight), `DID NOT FINISH IN 90 S`.

- [ ] **Step 1: Write failing tests:**
  - `stadium.test.ts`: on `figure8` with `over=[1]`, `carAboveBridges` is true at branch 1's crossing arc and false at branch 0's crossing arc; flipping `over` swaps them; false 100 m from any crossing.
  - `trial.test.ts`, using a fake `Policy`:
    - full throttle with zero steer on `loop` → `crashed`, `at` > 0;
    - a crash inside a `tight` section → `tight: true` and the description contains "tighter than anything in training";
    - zero throttle → `timeout` after 90 s;
    - the `describe` formats above.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `npm test --prefix web` → PASS.
- [ ] **Step 5: Commit** `feat: render overpasses and run an AI trial lap`.

### Task 7: Editor page

**Files:**
- Create: `web/editor.html`, `web/src/editor/main.ts`, `web/src/editor/preview.ts`, `web/src/editor/editor.css`, `web/e2e/editor.spec.ts`
- Modify: `web/vite.config.ts` (input `editor: "editor.html"`), `web/index.html` (enable "Draw a track" → `./editor.html`), `web/e2e/static.spec.ts` (load `/racing-rl/editor.html`), `web/e2e/intro.spec.ts` (button enabled)

**Interfaces:**
- Page ids:
  - drawing: `#draw-canvas` (2D canvas, `touch-action: none`, pointer events); `#track-stats` (`1,040 M · 1 OVERPASS`); `#issues` (list of `MESSAGES`, warnings styled differently);
  - buttons: `#clear`, `#submit` (disabled unless `canSubmit`), `#share`, `#race-it`;
  - trial: `#trial` (Pixi host for the AI lap), `#trial-result`.
- `preview.ts`: `drawPreview(ctx: CanvasRenderingContext2D, built: ReturnType<typeof buildEditorTrack>, view: { scale: number; offset: Vec2 }): void`. It draws grass, the road at true width, tight sections in red, bridges with shadow over the lower branch, the start line and a direction arrow. `crossingAt(built, view, point: Vec2): number | null` hit-tests 20 px around each crossing point.
- Flow:
  1. **Drawing:** pointerdown starts a new stroke, replacing any previous one. While drawing, rebuild at most every `rebuildMs` and preview as if closed. On pointerup, run the final validation.
  2. **Flip:** clicking a crossing (no drag) flips its `over`.
  3. **Clear** empties everything.
  4. **Submit:** load `final.onnx` through `AssetLoader`, create `RaceEngine` + `TrialLap` with `RaceScene` (`follow = "ai"`, `playerVisible = false`), show a live lap timer, then `describe(result)` in `#trial-result`. Submit is disabled while the trial runs; Clear during a trial stops it.
  5. **Race it** → `./race.html#track=<code>`.
  6. **Share** writes `editor.html#t=<code>` to `location.hash` and copies it (`navigator.clipboard`, with a selectable text fallback).
  7. **Shared link:** opening `#t=<code>` rebuilds the track. An undecodable code shows "This track link is damaged" in `#issues`.
- Follow `.superpowers/brainstorm/135740-1793034399/content/track-editor.html`: the drawing on the left, steps 1–3 on the right; stacked on phones.

- [ ] **Step 1: Write e2e** `editor.spec.ts`:
  - `draw a loop and the AI laps it`: draw a circle with `page.mouse` (60 points, closing at the start); `#submit` enabled; click; `#trial-result` matches `/LAP \d:\d\d\.\d{3}/` within 60 s.
  - `figure-eight AI lap crosses the overpass`: open `editor.html#t=<code>`, with the code generated in the test from the `figure8` fixture via `encodeTrack`. `#track-stats` contains "1 OVERPASS". Submit → `#trial-result` shows a lap, not a crash. This is the spec's phantom-wall acceptance.
  - `invalid drawing is blocked`: an open stroke shows "Finish near the start to close the loop" and `#submit` is disabled; drawing the `shallow` shape via share code shows its message.
  - `tight corner warns but allows`: the `tight` shape shows the tight message and `#submit` is enabled.
  - `flip survives sharing`: click the crossing, click Share; reload the URL from `location.hash`, and the bridge branch is flipped (`#track-stats` reads the same; the share code differs from the unflipped code).
  - `bad share link`: `editor.html#t=1.!!!!` shows "This track link is damaged" with no page error.
  - `draw on a phone`: 390×844 with `hasTouch`; draw a loop with touch events; no horizontal scroll; Submit enabled.
  - `clear wipes the drawing`.
- [ ] **Step 2: Run** `cd web && npx playwright test e2e/editor.spec.ts` → FAIL.
- [ ] **Step 3: Implement.** If the final policy fails the figure-eight for reasons unrelated to the crossing (check where it crashes), change only the `figure8` shape's lobe size in `tools/editor_shapes.py`, regenerate the fixtures, and log it in DECISIONS.md. Never change the crossing logic to make the test pass.
- [ ] **Step 4: Run** `make test`, `cd web && npm run test:e2e && npm run test:static` → PASS. Save `test-results/m9-editor.png` (a figure-eight with a bridge and the tight-corner warning) and look at it: the bridge must clearly sit on top with a shadow.
- [ ] **Step 5: Commit** `feat: track editor with overpasses, validation, AI trial lap and sharing`.

### Task 8: Race a drawn track

**Files:**
- Modify: `web/src/race/main.ts`, `web/race.html`, `web/e2e/race.spec.ts`

**Interfaces:**
- `race.html#track=<code>` decodes with `decodeTrack` and builds with `buildEditorTrack`.
  - The pre-race screen shows one track card ("Your track", length, minimap) and one opponent card "AI · FINAL POLICY" (no medal times); the opponent is the Author model.
  - Heatmap, ghost and AI-vision toggles that need exported data are disabled with the title "Not available on drawn tracks". AI vision stays available, since it is computed live.
  - A "Back to editor" link goes to `./editor.html#t=<code>`. Splits compare against the live AI, as on showcase tracks.
  - An undecodable code shows the notice "This track link is damaged" and falls back to the normal showcase setup.

- [ ] **Step 1: Write e2e**:
  - `race a drawn track`: open `race.html#track=<loop code>`; one `.track-card` named "YOUR TRACK"; Start; the countdown and race run; `#ai-time` eventually shows a lap time.
  - `damaged race link falls back`: shows the notice and three showcase track cards.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `make test`, `cd web && npm run test:e2e` → PASS.
- [ ] **Step 5: Commit** `feat: race the AI on a drawn track`.

### Task 9: Spec, docs and final verification

**Files:**
- Modify:
  - `spec.md`: §3.3 notes `track.v2` crossings; §7 mentions crossing parity; mark M9 delivered in §9.
  - `sim/OBSERVATION.md`: the ray exclusion rule at crossings; obs.v1 is otherwise unchanged.
  - `README.md`: the Draw a track section and a screenshot `docs/images/m9-editor.png`.
  - `docs/ASSETS.md`: the track.v2 format.
  - `docs/DEVELOPMENT.md`: the editor fixtures and `make parity`.
  - `DECISIONS.md`: `# M9 decisions`, covering every value in Global Constraints with its reason, the share-code format, the 90 s trial limit, and any figure-eight shape change.

- [ ] **Step 1:** Update the docs; README image paths must exist.
- [ ] **Step 2: Run** `make test`, `make web-build`, `make parity && git diff --exit-code tests/fixtures/parity/`, and `cd web && npm run test:e2e && npm run test:static` → all PASS.
- [ ] **Step 3:** Commit `docs: document M9 track editor`.
- [ ] **Step 4:** Summarize per AGENTS.md: what was built, how to run it, open issues.
