"""End-to-end test against the real CHTC LLM service.

Run with `uv run pytest -m live`. It needs CHTC_LLM_API_KEY and sends one request.
"""

import json
from pathlib import Path

import pytest

from rockd_live.classify import main

pytestmark = pytest.mark.live

ROCK_IMAGE = Path(__file__).parents[1] / "images" / "rock.jpg"


def test_classifies_outcrop_as_in_situ(tmp_path: Path) -> None:
    main([str(ROCK_IMAGE), "--out", str(tmp_path)])

    result = json.loads((tmp_path / "rock.json").read_text())
    assert result["label"] == "IN_SITU_ROCKS", result
    assert 0 <= result["confidence"] <= 1, result
