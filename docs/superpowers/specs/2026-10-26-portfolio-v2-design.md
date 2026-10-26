# Reinforced Racing v2: Stadium look, intro page, progressive learning, track editor

Date: 2026-10-26. Status: approved by owner.

## Intent

Reinforced Racing works and is deployed (https://joeycatic.github.io/ReinforcedRacing/), but it
looks like a generic dark dashboard, and "Watch it learn" shows six cars at once instead of a
learning story. The owner is putting this on their CV. Two audiences open the link:

- **Recruiters** skimming for ~30 seconds. They must leave thinking "understands RL and ships
  finished things".
- **Engineers** who scroll. They must find the substance, above all **PPO written from scratch**
  and **Python ↔ browser physics parity**.

v2 gives the project a distinct identity (Trackmania Stadium), a front page that tells the story,
a "Watch it learn" that reads as progressive learning, and a track editor that demonstrates
generalisation: a car that never saw your track drives it.

## Goals

1. Trackmania Stadium visual identity across the whole app, with high-quality Formula cars.
2. An intro/front page that works at both reading depths.
3. "Watch it learn" as a ~60-second sequence: one car per attempt, improving each time.
4. A track editor: draw a loop (overpasses allowed), the trained AI drives it, then you race it.

## Non-goals

- Training in the browser, or any retraining. Only the existing seed-2 run is used; exporting
  more of its saved checkpoints (M7 medals, M8 replays) is in scope.
- Touch driving controls, sound, online leaderboards, accounts, any backend.
- Car-to-car collision, 3D, new surfaces.
- Changing physics, observation semantics, or the trained models for existing tracks.

## Anti-goals (would be a failure even if it technically works)

- Looking like a template or SaaS dashboard (sidebar of `<select>`s, generic dark cards).
- Any claim on the page that the data doesn't back. Every number and caption comes from real
  evaluation output.
- Breaking parity: an existing track must produce bit-identical observations and the existing
  parity fixtures must keep passing unchanged.
- "Watch it learn" becoming a long wait or a clutter of cars again.

## Constraints

- Static site on GitHub Pages; no server. Vite + TypeScript + PixiJS + onnxruntime-web, as today.
- `AGENTS.md`: one milestone at a time, `make test` green before finishing each, ambiguous calls
  logged in `DECISIONS.md`, conventional commits. **No Claude co-author trailer in commits.**
- No external art assets: cars and environment are drawn in code (vector → PixiJS textures).
- Python is the source of truth for simulation; any simulation change lands in Python and
  TypeScript together, with golden-fixture parity tests.
- Phone visitors must get a working experience (intro, Watch it learn, drawing), even though
  racing needs a keyboard or gamepad.

## Delivery: three milestones, in order

These are appended to `spec.md` §9 as M7–M9 and amend §8 where noted. M7 comes first so M8 and
M9 are built directly in the new style.

### M7: Stadium identity + intro page

Amends spec §8.5 (visual style) and the §8.1 opponent picker.

**Visual identity (owner's choices: Trackmania Stadium look, Formula car)**
- Environment: daylight; striped grass; grey tarmac; red/white kerbs on both edges; blue
  checkpoint gates across the road at each checkpoint; checkered start/finish.
- Cars: open-wheel Formula cars, top-down, drawn in code with body shading, soft drop shadow,
  front/rear wings, visible tyres, nose number. Player = mint (`#3fe0b0` family), AI = orange
  (`#ff7a2f` family). Reference: `cars.html` option B in the brainstorm mockups.
- HUD in Trackmania's language: heavy italic type in black boxes; checkpoint split shown at
  top-centre with the delta in **blue when ahead, red when behind**; race time bottom-centre;
  large speed readout bottom-right.
- Camera sits closer than today so the car detail reads in play; zoom still eases with speed.
  *(Builder's call within that: exact zoom levels.)*

**Race setup (owner approved)**
- Opponents are the training stages framed as Trackmania medals: **Bronze, Silver, Gold, Author**
  (Author = final policy).
- Today only two shipped snapshots (3M, final) can finish a lap, so the medals come from the
  **existing training checkpoints** (owner's decision): evaluate the deterministic mean policy of
  the 36 checkpoints in `artifacts/training/m3-seed2/checkpoints/` on each showcase track, pick
  three that finish **all three** showcase tracks with clearly separated lap times
  (Bronze slowest), and export ONNX for them through the existing export/validation path. No
  retraining. The same three checkpoints are used on every track.
- Fallback (owner's decision): if fewer than three checkpoints finish a given track, that track
  shows fewer medals. Never show an opponent that cannot finish.
- Medal cards show the medal's real lap time on that track, from the evaluation output.
  Chosen checkpoint steps and the selection evidence go in DECISIONS.md.
- Pre-race screen with track cards and medal-opponent cards replaces the sidebar of dropdowns.
  The race itself is full-screen.

**Intro page (owner approved the structure in `intro-page.html`)**
- It is the site's front page. Race, Watch it learn and the editor are separate screens reached
  from it.
- Hero: headline "A car that taught itself to drive.", one sentence on what it is, three
  buttons (Race the AI / Watch it learn / Draw a track). Background is the **real final policy
  lapping a Stadium track live** (ONNX in the browser), not a video.
- Stats strip: 98.7% showcase completion · 85.2% on 50 unseen tracks · 0 RL libraries ·
  5e-13 physics parity error. Values read from committed result files, not hand-typed.
- Engineer sections, in order:
  1. **PPO from scratch**: loop diagram (64 cars → rollout → GAE → clipped update), interactive
     learning curve (hover tooltips), the two-seed comparison including seed 1's regression,
     links into `train/ppo.py`.
  2. **Physics parity**: a Python-recorded trajectory and the live TypeScript simulation driven
     by the same inputs, drawn overlapping; error figures; one line on why it matters ("without
     this, the AI you race would drive worse than the one I trained").
  3. **Built to ship**: CI status, test counts, versioned/hashed assets, static deploy.
- Footer: owner's name, GitHub, LinkedIn. *(Owner to supply the name/LinkedIn URL; placeholder
  until then.)*
- Phones: full intro; buttons route to screens that work without a keyboard; Race shows the
  existing keyboard/gamepad notice.

**Acceptance**: every screen uses the Stadium look; no `<select>`-sidebar remains; intro renders
correctly at phone width; race behaviour, lap times and parity results unchanged; `make test`
and the e2e/static suites green.

### M8: Progressive "Watch it learn"

Replaces spec §8.2 (ghost time-lapse). §8.3 heatmaps stay, as an optional layer.

**Experience (owner's choices: one car at a time, improving each attempt, ~1 minute total)**
- About 10 attempts on one showcase track, played in training order, one car at a time.
- Each completed attempt leaves a faded trail, coloured cool (early) → warm (final), so the
  racing line visibly forms.
- Pacing to fit ~60 s: crashes play at 1× (short anyway); stalls are cut after ~3 s and labelled
  "stalled"; middle attempts fast-forward at 3×; the final clean lap gets the most time at 2× with
  a live lap timer. *(Builder's call: exact per-attempt speeds, as long as total ≈ 60 s.)*
- Bottom bar is **training progress**, one segment per attempt; play/pause, replay, skip to
  final, click any segment to jump.
- Captions are generated from data: attempt number, training steps, outcome ("crashed at
  312 m", "stalled", "lap 24.25 s"). A few milestone attempts get a short headline ("Found the
  throttle", "First full lap"), written only after inspecting what that checkpoint actually does.
- Crash heatmap for the current attempt's stage: optional toggle, off by default.
- Track picker for the three showcase tracks.

**Data**
- Attempts are deterministic mean-policy runs (as the current ghosts are) of selected
  checkpoints. Available: snapshots 0 / 50K / 200K / 1M / 3M / final plus 36 checkpoints every
  163,840 steps in `artifacts/training/m3-seed2/checkpoints/`.
- Selection: ~10 checkpoints, denser early in training, each showing a visible behaviour change
  on that track; selection may differ per track. *(Builder's call: the selection method; record
  the chosen steps in DECISIONS.md.)*
- Export adds replays for the selected checkpoints only. No ONNX models are needed for them,
  so the bundle grows only by replay size. Asset manifest/versioning extended accordingly.

**Acceptance**: a full playthrough on each showcase track takes 50–70 s; one moving car at a
time; captions match the replay outcomes; the six-ghost simultaneous view is gone.

### M9: Track editor

New screen; adds a `track.v2` capability (crossings) to the simulation.

**Experience (owner's choices: draw freehand, crossings become over/underpasses, Clear to
start over, Submit when done, then the AI drives it, then you race it)**
- One continuous freehand stroke with mouse or finger. The loop closes automatically when the
  stroke ends near its start. The start line is the stroke's first point; driving direction is the
  drawing direction.
- The stroke is smoothed into a spline, resampled, and scaled to a track length in the range of
  the training tracks. Road width is fixed inside the training range (10–14 m).
- Crossings become overpasses: the later pass over a crossing is the bridge by default; clicking
  a crossing flips it. Bridge drawn on top with a shadow. Crossings shallower than a minimum angle
  are rejected (the roads would overlap too long). *(Builder's call: the exact minimum angle and
  minimum spacing between crossings / from the start line.)*
- Live validation while drawing:
  - **Blocks Submit**: geometry that cannot exist (road folds inside a corner, boundaries
    overlapping other than at a valid crossing, too short).
  - **Warns but allows**: corners sharper than the training limit (`max_curvature` 0.10) are
    painted red with a "too tight — the AI never trained on corners this sharp" note.
- **Clear** wipes the drawing.
- **Submit** → the final policy drives one lap with a follow camera and lap timer. Result: lap
  time, or "crashed at N m" with the reason when known (e.g. the crash is inside a red section).
- **Race it** → the normal race screen on the drawn track, AI lap as the target, splits vs AI.
- **Share** (my addition, owner did not object): the track is encoded in the URL; opening the
  link loads it. No server.

**Simulation change: crossings**
- Crash detection already uses a local-window centreline projection, so it stays on the correct
  branch at a crossing.
- Raycasts must not see the other branch's boundaries inside a crossing zone (otherwise the car
  "sees" a wall across its road). Add per-crossing boundary-segment exclusion keyed on the car's
  current branch.
- Implemented in **both** Python (`sim/`) and TypeScript (`web/src/sim/`), with new golden
  fixtures on a crossing track. Tracks without crossings must produce identical observations; all
  existing fixtures must pass unchanged. This also means the editor's track builder is a
  TypeScript port of `build_track` / `validity_errors`, parity-tested against Python on the same
  control points.

**Acceptance**: a drawn simple loop can be driven by the AI and raced; a figure-eight with one
overpass is driven by the AI without phantom-wall crashes at the crossing; invalid drawings are
blocked with a reason; too-tight corners warn; shared URLs reproduce the same track;
Python/TS crossing parity fixtures pass; existing parity untouched.

## Left to the builder

Exact colours beyond the named families, typography choice within "heavy italic", camera zoom
values, medal-to-snapshot mapping, checkpoint selection method for M8, per-attempt playback
speeds, crossing angle/spacing thresholds, URL encoding format, and module structure. Record
each in DECISIONS.md.

## Reference mockups

`.superpowers/brainstorm/135740-1793034399/content/` (git-ignored, local only):
`visual-direction.html` (A chosen), `cars.html` (B chosen), `intro-page.html`,
`watch-it-learn.html`, `track-editor.html`.
