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
