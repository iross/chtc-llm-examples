import json
import re
from collections.abc import Callable
from pathlib import Path

import httpx2
import pypdfium2 as pdfium
import pytest
from openai import OpenAI

from pdf_ocr.ocr import BASE_URL, Settings, find_pdfs, main, resolve_model, run

Handler = Callable[[httpx2.Request], httpx2.Response]
M72 = Settings("m", dpi=72)


def chat_reply(content: str | None, finish_reason: str = "stop") -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
        },
    )


def models_handler(ids: list[str]) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/models"
        data = [{"id": i, "object": "model", "created": 0, "owned_by": "x"} for i in ids]
        return httpx2.Response(200, json={"object": "list", "data": data})

    return handler


def make_client(handler: Handler) -> OpenAI:
    transport = httpx2.MockTransport(handler)
    return OpenAI(
        base_url=BASE_URL,
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.Client(transport=transport),
    )


def write_pdf(path: Path, pages: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(200, 100)
    pdf.save(path)
    pdf.close()
    return path


def never_called(request: httpx2.Request) -> httpx2.Response:
    raise AssertionError(f"unexpected request to {request.url}")


class TestFindPdfs:
    def test_recurses_and_filters_by_suffix(self, tmp_path: Path) -> None:
        a = write_pdf(tmp_path / "a.pdf")
        b = write_pdf(tmp_path / "sub" / "deeper" / "b.PDF")
        (tmp_path / "notes.txt").write_text("skip me")
        assert find_pdfs([tmp_path]) == [a, b]

    def test_accepts_individual_files(self, tmp_path: Path) -> None:
        pdf = write_pdf(tmp_path / "a.pdf")
        assert find_pdfs([pdf]) == [pdf]

    def test_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="nope"):
            find_pdfs([tmp_path / "nope"])

    def test_duplicate_names_rejected(self, tmp_path: Path) -> None:
        write_pdf(tmp_path / "x" / "same.pdf")
        write_pdf(tmp_path / "y" / "same.pdf")
        with pytest.raises(ValueError, match="Duplicate PDF name 'same'"):
            find_pdfs([tmp_path])


class TestResolveModel:
    def test_requested_model_available(self) -> None:
        client = make_client(models_handler(["other", "qwen3.8-27b"]))
        assert resolve_model(client, "qwen3.8-27b") == "qwen3.8-27b"

    def test_requested_model_unavailable(self) -> None:
        client = make_client(models_handler(["a", "b"]))
        with pytest.raises(SystemExit, match=r"'nonsense' is not available.*Available: a, b"):
            resolve_model(client, "nonsense")


class TestRun:
    def test_ocrs_each_page_and_combines(self, tmp_path: Path) -> None:
        requests: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            requests.append(request)
            return chat_reply(f"\n\n# Page text {len(requests)}\n")

        pdf = write_pdf(tmp_path / "in" / "report.pdf", pages=2)
        out = tmp_path / "out"

        assert run(make_client(handler), M72, [pdf], out) == 0
        assert (out / "report" / "page-0001.md").read_text() == "# Page text 1\n"
        assert (out / "report.md").read_text() == (
            "<!-- page 1 -->\n\n# Page text 1\n\n<!-- page 2 -->\n\n# Page text 2\n"
        )
        assert len(requests) == 2
        assert requests[0].url.path == "/api/chat/completions"
        assert requests[0].headers["authorization"] == "Bearer test-key"
        content = json.loads(requests[0].content)["messages"][0]["content"]
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,iVBOR")

    def test_skips_finished_pdfs(self, tmp_path: Path) -> None:
        pdf = write_pdf(tmp_path / "report.pdf")
        out = tmp_path / "out"
        out.mkdir()
        (out / "report.md").write_text("done")
        assert run(make_client(never_called), M72, [pdf], out) == 0

    def test_failed_page_is_retried_on_rerun(self, tmp_path: Path) -> None:
        replies = iter([chat_reply("one"), httpx2.Response(500, json={"detail": "boom"})])
        pdf = write_pdf(tmp_path / "report.pdf", pages=2)
        out = tmp_path / "out"

        assert run(make_client(lambda r: next(replies)), M72, [pdf], out) == 1
        assert (out / "report" / "page-0001.md").exists()
        assert not (out / "report" / "page-0002.md").exists()
        assert not (out / "report.md").exists()

        assert run(make_client(lambda r: chat_reply("two")), M72, [pdf], out) == 0
        assert (out / "report.md").read_text().endswith("<!-- page 2 -->\n\ntwo\n")

    @pytest.mark.parametrize(
        ("content", "finish_reason"), [(None, "stop"), ("  ", "stop"), ("partial", "length")]
    )
    def test_empty_or_truncated_reply_fails(
        self, tmp_path: Path, content: str | None, finish_reason: str
    ) -> None:
        pdf = write_pdf(tmp_path / "report.pdf")
        client = make_client(lambda r: chat_reply(content, finish_reason))
        assert run(client, M72, [pdf], tmp_path / "out") == 1
        assert not (tmp_path / "out" / "report" / "page-0001.md").exists()

    def test_unreadable_pdf_fails_and_continues(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.pdf"
        bad.write_text("not a pdf")
        good = write_pdf(tmp_path / "good.pdf")
        out = tmp_path / "out"

        assert run(make_client(lambda r: chat_reply("ok")), M72, [bad, good], out) == 1
        assert not (out / "bad").exists()
        assert (out / "good.md").exists()


class TestMain:
    def test_missing_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CHTC_LLM_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="CHTC_LLM_API_KEY is not set"):
            main([str(tmp_path)])

    def test_no_pdfs_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHTC_LLM_API_KEY", "key")
        with pytest.raises(SystemExit, match="No PDF files found"):
            main([str(tmp_path)])

    @pytest.mark.parametrize("dpi", ["0", "601"])
    def test_dpi_out_of_range(self, tmp_path: Path, dpi: str) -> None:
        with pytest.raises(SystemExit):
            main([str(tmp_path), "--dpi", dpi])

    def test_rejected_key_aborts_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(401, json={"detail": "Not authenticated"})

        monkeypatch.setenv("CHTC_LLM_API_KEY", "bad")
        monkeypatch.setattr("pdf_ocr.ocr.OpenAI", lambda **kw: make_client(handler))
        write_pdf(tmp_path / "a.pdf")
        with pytest.raises(SystemExit, match=r"HTTP 401\): Not authenticated"):
            main([str(tmp_path), "--out", str(tmp_path / "out")])

    def test_unknown_model_stops_before_any_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHTC_LLM_API_KEY", "key")
        monkeypatch.setattr(
            "pdf_ocr.ocr.OpenAI", lambda **kw: make_client(models_handler(["qwen3.8-27b"]))
        )
        write_pdf(tmp_path / "a.pdf")
        with pytest.raises(SystemExit, match="'nonsense' is not available"):
            main([str(tmp_path), "--model", "nonsense", "--out", str(tmp_path / "out")])


class TestThinking:
    @staticmethod
    def capture(requests: list[dict], reasoning_field: str = "reasoning_content") -> Handler:
        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.url.path.endswith("/models"):
                return models_handler(["qwen3.8-27b"])(request)
            requests.append(json.loads(request.content))
            body = json.loads(chat_reply("text").content)
            body["choices"][0]["message"][reasoning_field] = "Let me read the page."
            body["usage"] = {"prompt_tokens": 500, "completion_tokens": 42, "total_tokens": 542}
            return httpx2.Response(200, json=body)

        return handler

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [([], None), (["--no-thinking"], {"enable_thinking": False})],
    )
    def test_flag_controls_chat_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: object
    ) -> None:
        requests: list[dict] = []
        monkeypatch.setenv("CHTC_LLM_API_KEY", "key")
        monkeypatch.setattr("pdf_ocr.ocr.OpenAI", lambda **kw: make_client(self.capture(requests)))
        write_pdf(tmp_path / "a.pdf")
        main([str(tmp_path / "a.pdf"), "--out", str(tmp_path / "out"), *flags])
        (body,) = requests
        assert body.get("chat_template_kwargs") == expected

    @pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
    def test_logs_time_tokens_and_reasoning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str
    ) -> None:
        pdf = write_pdf(tmp_path / "a.pdf")
        assert run(make_client(self.capture([], field)), M72, [pdf], tmp_path / "out") == 0
        err = capsys.readouterr().err
        assert re.search(r"page 1/1 \.\.\. \d+\.\ds, 42 tokens \(21 chars of reasoning\)", err)

    def test_no_reasoning_logged_as_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pdf = write_pdf(tmp_path / "a.pdf")
        assert run(make_client(lambda r: chat_reply("text")), M72, [pdf], tmp_path / "out") == 0
        assert "(0 chars of reasoning)" in capsys.readouterr().err


class TestBaseUrl:
    OLLAMA = "http://localhost:11434/v1"

    def test_ollama_needs_no_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict = {}

        def fake_openai(**kw: object) -> OpenAI:
            seen.update(kw)
            return make_client(TestThinking.capture([], "reasoning"))

        monkeypatch.delenv("CHTC_LLM_API_KEY", raising=False)
        monkeypatch.setattr("pdf_ocr.ocr.OpenAI", fake_openai)
        write_pdf(tmp_path / "a.pdf")
        main([str(tmp_path), "--base-url", self.OLLAMA, "--out", str(tmp_path / "out")])
        assert seen["base_url"] == self.OLLAMA
        assert seen["api_key"] == "unused"
        assert (tmp_path / "out" / "a.md").exists()

    def test_server_not_running(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("Connection refused", request=request)

        monkeypatch.setattr("pdf_ocr.ocr.OpenAI", lambda **kw: make_client(refuse))
        write_pdf(tmp_path / "a.pdf")
        with pytest.raises(SystemExit, match=r"Could not connect to http://localhost:11434/v1"):
            main([str(tmp_path), "--base-url", self.OLLAMA, "--out", str(tmp_path / "out")])

    def test_ollama_style_error_body(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(404, json={"error": {"message": "model not found"}})

        monkeypatch.setattr("pdf_ocr.ocr.OpenAI", lambda **kw: make_client(handler))
        write_pdf(tmp_path / "a.pdf")
        with pytest.raises(SystemExit, match=r"HTTP 404\): .*model not found"):
            main([str(tmp_path), "--base-url", self.OLLAMA, "--out", str(tmp_path / "out")])
