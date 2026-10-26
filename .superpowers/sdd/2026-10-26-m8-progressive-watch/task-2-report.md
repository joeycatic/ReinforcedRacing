# M8 Task 2 report — export learning attempts

## Scope and implementation

- Implemented manifest.v3 `attempts` with selected per-track entries, actual metadata steps, source/headline, outcome/progress/lap time and hashed replay descriptors.
- Snapshot choices reuse the exact existing descriptor. Checkpoint choices are SHA-256 checked against selection evidence before recording; checkpoint replay metadata uses `attempt-<step>` and files publish only for selected tracks.
- Deterministic attempt caches live under evaluation `attempts/<step>/`, using the existing configuration/source/geometry/checkpoint fingerprint and exact expected file-set plus per-file integrity checks. Corrupt and incomplete caches regenerate; verified caches avoid recording.
- Every output's metadata checkpoint hash and actual step are checked against selection before its manifest entry is emitted.
- Updated asset documentation and DECISIONS.md. Per coordinator ruling, the minimal browser version gate and synthetic fixture move to v3 atomically; v2 is explicitly rejected. Attempt validation/API and timeline remain Task 3.
- No dependencies, retraining, simulation/observation changes or new ONNX model identities.

## Test-first evidence

Initial tests were written before implementation, then:

```text
uv run --locked pytest tests/test_export.py -v
4 failed, 7 passed, 2 warnings in 7.04s
FAILED test_complete_export_integrity_and_cache_repair
FAILED test_committed_medal_bundle_integrity
FAILED test_committed_attempt_bundle_integrity
FAILED test_attempt_checkpoint_sha_matches_selection
```

Failures were missing attempt export hook, old manifest version and absent attempts. After implementation but before export, synthetic export passed while three committed-bundle tests correctly remained red. The browser integrity test also correctly rejected the still-v2 committed bundle through the new v3 gate (11 passed, 1 failed).

Additional unit coverage verifies corrupt/incomplete cache repair, cache hits avoiding recording, changed checkpoint SHA rejection before recording, and snapshot SHA/step evidence rejection (3 passed, 11 deselected in 1.27s). Existing non-committed export tests passed (9 passed, 3 deselected in 7.35s); Ruff and formatting checks pass.

## Export and final validation

`make export` completed successfully (exit 0). `uv run --locked pytest tests/test_export.py -v` then passed: **14 passed, 6 warnings in 7.33s**. Warnings are the existing legacy Torch ONNX deprecations.

- Counts: Sweep 10, Ribbon 7, Technical 9 (26 entries).
- Six snapshot entries reuse the existing initial/final replay descriptors; 20 new replay files occupy **447,383 bytes**, below the ~1.5 MB target.
- Total regular-file bundle size: **6,645,611 → 7,100,434 bytes**, +454,823 bytes including manifest growth. `du -sh web/public/assets`: **6.9M** (before 6.5M).
- Compared every pre-existing public file against Git HEAD bytes: **only manifest.json changed**. All existing ONNX model bytes, tracks, simulation, replays, heatmaps, site and parity demo are exact. Existing snapshots/medals/version/track/simulation/site/parity manifest values are also equal.
- `git diff --check` clean. Final `make test` result below.

The new Task 1 eval module invalidates old evaluation fingerprints because the existing fingerprint includes all `eval/*.py`. Full showcase/stochastic/held-out evaluation is therefore genuinely regenerated; no cache bypass or fingerprint rewrite is used.

## Reproduction

With retained seed-2 checkpoints/logs installed, run `make export`, then `make test`. The reference bundle is committed for fresh-clone playback. Task 3 will add attempt consumer validation/loading; this task deliberately leaves existing viewer behavior intact.

## Independent cache prewarming

At coordinator suggestion, paused the main exporter with `kill -STOP 179889` while it
was evaluating snapshot 50000, then started four processes with the existing evaluator:

```python
report = evaluate_snapshot(
    ROOT / config.run_dir / "snapshots" / f"{step}.pt",
    load_export(), make_track_pool("showcase"), make_track_pool("held_out"),
)
```

For each of `200000`, `1000000`, `3000000`, `final`:
`PYTHONPATH=. uv run --locked python .superpowers/sdd/2026-10-26-m8-progressive-watch/scratch/task2-prewarm/prewarm.py <step>`.
Each process writes its disjoint existing snapshot cache directory. Main export resumes
with `kill -CONT 179889` after every prewarm succeeds, so no cache has concurrent writers.
Logs are `/tmp/m8-task2-prewarm-<step>.log`; no fingerprint/config/source modifications.

## Open issues and scope boundary

No Task 2 blockers. Browser attempt schema validation and `attemptReplay` remain Task 3 as planned; the existing viewer continues loading snapshots. M8 UI/playback/e2e acceptance remains later tasks. No push or deployment performed.

## Final GREEN output

```text
make test  # exit 0
All checks passed!
66 files already formatted
106 passed, 6 warnings in 21.83s
tsc --noEmit  # passed
Test Files  8 passed (8)
Tests  44 passed (44)
Duration  1.12s
```

Evidence logs: `/tmp/m8-task2-red.log`, `/tmp/m8-task2-green.log`,
`/tmp/m8-task2-export.log`, `/tmp/m8-task2-test.log`, and prewarm logs above.
