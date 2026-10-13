"""Every recording in tests/replays/ is a regression test (v0.1.20).

Drop a ``.awayrec`` there (from ``away agent``'s record_session, a diagnostics bundle's ``last-seconds.awayrec``,
or an auto-detector bundle) and run the tests once with ``AWAY_UPDATE_REPLAYS=1`` to pin what the gesture pipeline
does with it (``<file>.expected.json``). From then on any change in the detected gestures or mouse calls fails here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from away.agentlink.replay import assert_replay

REPLAYS = sorted((Path(__file__).parent / "replays").glob("*.awayrec"))


@pytest.mark.parametrize("path", REPLAYS, ids=[p.stem for p in REPLAYS])
def test_recorded_session_still_does_the_same(path: Path) -> None:
    assert_replay(path, update=os.environ.get("AWAY_UPDATE_REPLAYS") == "1")


def test_there_is_at_least_one_replay() -> None:
    assert REPLAYS, "tests/replays/ should hold at least one recording"
