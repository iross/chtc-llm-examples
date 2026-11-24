"""End-to-end tests against the real CHTC LLM service.

Run with `uv run pytest -m live`. They need CHTC_LLM_API_KEY and send two requests.
"""

from pathlib import Path

import pytest

from pdf_ocr.ocr import main

pytestmark = pytest.mark.live

SAMPLE_PDF = Path(__file__).parent / "data" / "field_report.pdf"
EXPECTED_TEXT = ["Annual Field Report", "Wisconsin River", "21.4", "Site summary", "A-1", "6.9"]


@pytest.mark.parametrize(
    ("flags", "reasoning_off"), [([], False), (["--no-thinking"], True)], ids=["thinking", "off"]
)
def test_transcribes_sample_pdf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flags: list[str], reasoning_off: bool
) -> None:
    main([str(SAMPLE_PDF), "--out", str(tmp_path), *flags])

    text = (tmp_path / "field_report.md").read_text()
    missing = [s for s in EXPECTED_TEXT if s not in text]
    assert not missing, f"Missing {missing} from transcription:\n{text}"
    assert "|" in text, f"Expected the table as Markdown:\n{text}"
    assert ("(0 chars of reasoning)" in capsys.readouterr().err) == reasoning_off
