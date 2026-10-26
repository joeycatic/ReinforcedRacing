from eval.attempts import Run, score, select_attempts


def crash(step: int, progress: float) -> Run:
    return Run(step, str(step), "crashed", progress, None)


def finish(step: int, lap: float, source: str | None = None) -> Run:
    return Run(step, source or str(step), "finished", 1000, lap)


def test_score_orders_finishers_above_crashes() -> None:
    assert (
        score(finish(4, 25))
        > score(crash(3, 800))
        > score(crash(2, 100))
        > score(Run(1, "1", "stalled", 40, None))
    )


def test_selection_keeps_first_and_final_and_strictly_improves() -> None:
    runs = [crash(i, i * 30) for i in range(39)] + [finish(40, 25, "final")]
    selected = select_attempts(runs)
    assert selected[0] == runs[0]
    assert selected[-1] == runs[-1]
    assert all(score(a) < score(b) for a, b in zip(selected, selected[1:], strict=False))


def test_selection_requires_visible_change() -> None:
    runs = [
        crash(0, 100),
        crash(1, 105),
        crash(2, 120),
        finish(3, 30),
        finish(4, 29.9),
        finish(5, 29),
        finish(6, 28, "final"),
    ]
    assert [r.step for r in select_attempts(runs)] == [0, 2, 3, 5, 6]


def test_selection_skips_runs_faster_than_final() -> None:
    runs = [crash(0, 0), finish(1, 26), finish(2, 23.88), finish(3, 24.25, "final")]
    assert [r.step for r in select_attempts(runs)] == [0, 1, 3]


def test_selection_thins_to_count_keeping_milestones() -> None:
    runs = [crash(i, i * 30) for i in range(8)] + [finish(i, 40 - i) for i in range(8, 15)]
    runs.append(finish(15, 24, "final"))
    selected = select_attempts(runs)
    assert len(selected) == 10
    assert [r.step for r in selected] == [0, 2, 3, 5, 7, 8, 10, 12, 13, 15]


def test_headlines_only_on_selected_steps() -> None:
    import json

    from eval.attempts import load_attempts
    from sim.config import ROOT

    evidence = json.loads((ROOT / "experiments/attempt-selection.json").read_text())
    for track, choices in load_attempts().items():
        runs = [Run(**candidate["runs"][track]) for candidate in evidence["candidates"]]
        selected = select_attempts(runs)
        assert [(c.step, c.source) for c in choices] == [(r.step, r.source) for r in selected]
        assert all(c.headline is None or c.step in {r.step for r in selected} for c in choices)
        assert choices[0].headline == "Untrained policy"
        assert choices[-1].headline == "Final policy"
