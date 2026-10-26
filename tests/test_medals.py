from eval.medals import select_medals


def test_select_medals_picks_three_separated_finishers() -> None:
    candidates = {1: {"a": 23.0, "b": 34.0}, 3: {"a": 22.0, "b": 32.5}, 5: {"a": 21.0, "b": 31.0}}
    result = select_medals(candidates, {"a": 20.0, "b": 30.0})
    assert [c.medal for c in result] == ["bronze", "silver", "gold"]
    assert [c.step for c in result] == [1, 3, 5]


def test_select_medals_enforces_gap_on_every_track() -> None:
    author = {"a": 20.0, "b": 30.0}
    result = select_medals(
        {1: {"a": 22.0, "b": 30.3}, 2: {"a": 23.0, "b": 32.0}, 3: {"a": 25.0, "b": 34.0}}, author
    )
    assert 1 not in [c.step for c in result]
    laps = [c.lap_times for c in result] + [author]
    for slower, faster in zip(laps, laps[1:], strict=False):
        for track in author:
            if slower[track] is not None and faster[track] is not None:
                assert slower[track] >= faster[track] * 1.03


def test_select_medals_prefers_full_finishers_then_partial() -> None:
    result = select_medals(
        {1: {"a": 22.0, "b": 33.0}, 2: {"a": 24.0, "b": None}}, {"a": 20.0, "b": 30.0}
    )
    assert len(result) == 2
    assert result[0].lap_times["b"] is None


def test_select_medals_returns_fewer_when_none_qualify() -> None:
    assert select_medals({1: {"a": None}}, {"a": 20.0}) == []


def test_partial_medals_keep_gap_after_filtering_each_tracks_missing_laps() -> None:
    author = {"a": 20.0, "b": 30.0}
    result = select_medals(
        {
            1: {"a": 22.0, "b": 33.0},
            2: {"a": 24.0, "b": None},
            3: {"a": None, "b": 25.0},
            4: {"a": None, "b": 36.0},
        },
        author,
    )
    assert [choice.step for choice in result] == [4, 2, 1]
    for track in author:
        visible = [c.lap_times[track] for c in result if c.lap_times[track] is not None]
        visible.append(author[track])
        for slower, faster in zip(visible, visible[1:], strict=False):
            assert slower >= faster * 1.03


def test_committed_selection_still_matches_all_recorded_candidate_laps() -> None:
    import json

    from sim.config import ROOT

    evidence = json.loads((ROOT / "experiments/medal-selection.json").read_text())
    candidates = {
        int(step): candidate["lap_times"] for step, candidate in evidence["candidates"].items()
    }
    result = select_medals(candidates, evidence["author"]["lap_times"], evidence["gap"])
    assert [(c.medal, c.step) for c in result] == [
        (c["medal"], c["step"]) for c in evidence["choices"]
    ]
