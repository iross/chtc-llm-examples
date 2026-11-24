import json
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from openai import OpenAI

from rockd_live.classify import (
    BASE_URL,
    find_images,
    main,
    parse_json_response,
    resolve_model,
    run,
)

Handler = Callable[[httpx2.Request], httpx2.Response]
GOOD_ANSWER = {"reasoning": "Layered outcrop.", "label": "IN_SITU_ROCKS", "confidence": 0.9}


def chat_reply(content: str) -> httpx2.Response:
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
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
        },
    )


def make_client(handler: Handler) -> OpenAI:
    transport = httpx2.MockTransport(handler)
    return OpenAI(
        base_url=BASE_URL,
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.Client(transport=transport),
    )


def write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8fake-jpeg")
    return path


class TestParseJsonResponse:
    def test_plain_json(self) -> None:
        assert parse_json_response(json.dumps(GOOD_ANSWER)) == GOOD_ANSWER

    def test_json_wrapped_in_text(self) -> None:
        reply = f"<think>hmm</think>\n```json\n{json.dumps(GOOD_ANSWER)}\n```"
        assert parse_json_response(reply) == GOOD_ANSWER

    @pytest.mark.parametrize("reply", ["", "no json here", "} backwards {"])
    def test_no_json_raises(self, reply: str) -> None:
        with pytest.raises(ValueError, match="No JSON object"):
            parse_json_response(reply)


class TestFindImages:
    def test_recurses_and_filters_by_suffix(self, tmp_path: Path) -> None:
        a = write_image(tmp_path / "a.jpg")
        b = write_image(tmp_path / "sub" / "deeper" / "b.JPEG")
        c = write_image(tmp_path / "c.png")
        (tmp_path / "notes.txt").write_text("skip me")
        assert find_images([tmp_path]) == [a, c, b]

    def test_accepts_individual_files(self, tmp_path: Path) -> None:
        image = write_image(tmp_path / "a.jpg")
        assert find_images([image]) == [image]

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert find_images([tmp_path]) == []

    def test_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="nope"):
            find_images([tmp_path / "nope"])

    def test_duplicate_names_rejected(self, tmp_path: Path) -> None:
        write_image(tmp_path / "x" / "same.jpg")
        write_image(tmp_path / "y" / "same.jpg")
        with pytest.raises(ValueError, match="Duplicate image name 'same'"):
            find_images([tmp_path])


class TestResolveModel:
    @staticmethod
    def models_handler(ids: list[str]) -> Handler:
        def handler(request: httpx2.Request) -> httpx2.Response:
            assert request.url.path == "/api/models"
            data = [{"id": i, "object": "model", "created": 0, "owned_by": "x"} for i in ids]
            return httpx2.Response(200, json={"object": "list", "data": data})

        return handler

    def test_requested_model_available(self) -> None:
        client = make_client(self.models_handler(["other", "qwen3-vl:8b"]))
        assert resolve_model(client, "qwen3-vl:8b") == "qwen3-vl:8b"

    def test_requested_model_unavailable(self) -> None:
        client = make_client(self.models_handler(["a", "b"]))
        with pytest.raises(SystemExit, match=r"'nonsense' is not available.*Available: a, b"):
            resolve_model(client, "nonsense")


class TestRun:
    def test_classifies_and_writes_json(self, tmp_path: Path) -> None:
        requests: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            requests.append(request)
            return chat_reply(json.dumps(GOOD_ANSWER))

        image = write_image(tmp_path / "in" / "123.jpg")
        out = tmp_path / "out"
        failures = run(make_client(handler), "m", [image], out, "PROMPT")

        assert failures == 0
        assert json.loads((out / "123.json").read_text()) == GOOD_ANSWER
        (request,) = requests
        assert request.url.path == "/api/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        content = json.loads(request.content)["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "PROMPT"}
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_skips_images_with_existing_results(self, tmp_path: Path) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            raise AssertionError("should not call the API")

        image = write_image(tmp_path / "123.jpg")
        out = tmp_path / "out"
        out.mkdir()
        (out / "123.json").write_text("{}")
        assert run(make_client(handler), "m", [image], out, "p") == 0

    def test_bad_reply_counts_as_failure_and_continues(self, tmp_path: Path) -> None:
        replies = iter(["I cannot help with that.", json.dumps(GOOD_ANSWER)])

        def handler(request: httpx2.Request) -> httpx2.Response:
            return chat_reply(next(replies))

        images = [write_image(tmp_path / "1.jpg"), write_image(tmp_path / "2.jpg")]
        out = tmp_path / "out"
        failures = run(make_client(handler), "m", images, out, "p")

        assert failures == 1
        assert not (out / "1.json").exists()
        assert (out / "2.json").exists()

    def test_server_error_counts_as_failure(self, tmp_path: Path) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(500, json={"detail": "boom"})

        image = write_image(tmp_path / "1.jpg")
        assert run(make_client(handler), "m", [image], tmp_path / "out", "p") == 1


class TestMain:
    def test_missing_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CHTC_LLM_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="CHTC_LLM_API_KEY is not set"):
            main([str(tmp_path)])

    def test_rejected_key_aborts_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(401, json={"detail": "Not authenticated"})

        monkeypatch.setenv("CHTC_LLM_API_KEY", "bad")
        monkeypatch.setattr("rockd_live.classify.OpenAI", lambda **kw: make_client(handler))
        write_image(tmp_path / "1.jpg")
        with pytest.raises(SystemExit, match=r"HTTP 401\): Not authenticated"):
            main([str(tmp_path), "--model", "m", "--out", str(tmp_path / "out")])

    def test_unknown_model_stops_before_any_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.url.path == "/api/chat/completions":
                raise AssertionError("no image should be sent for an unknown model")
            return TestResolveModel.models_handler(["real-model"])(request)

        monkeypatch.setenv("CHTC_LLM_API_KEY", "key")
        monkeypatch.setattr("rockd_live.classify.OpenAI", lambda **kw: make_client(handler))
        write_image(tmp_path / "1.jpg")
        with pytest.raises(SystemExit, match="'nonsense' is not available"):
            main([str(tmp_path), "--model", "nonsense", "--out", str(tmp_path / "out")])
