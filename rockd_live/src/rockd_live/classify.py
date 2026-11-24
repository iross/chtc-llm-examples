"""Classify Rockd images with the vision model hosted on CHTC's Open WebUI service."""

import argparse
import base64
import json
import os
import sys
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI

BASE_URL = "https://llm.chtc.wisc.edu/api"
API_KEY_ENV = "CHTC_LLM_API_KEY"
MODEL_ENV = "CHTC_LLM_MODEL"
DEFAULT_MODEL = "qwen3.8-27b"
IMAGE_SUFFIXES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
REQUEST_TIMEOUT_SECONDS = 180.0

FATAL_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)


def default_prompt() -> str:
    return files("rockd_live").joinpath("prompts/classify.txt").read_text()


def parse_json_response(content: str) -> dict[str, Any]:
    """Parse the model's reply, tolerating text (e.g. reasoning) around the JSON object.

    Raises:
        ValueError: If no JSON object can be recovered from the reply.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object in model reply: {content[:200]!r}") from None
        return json.loads(content[start : end + 1])


def classify_image(client: OpenAI, model: str, image_path: Path, prompt: str) -> dict[str, Any]:
    """Send one image with the prompt and return the model's parsed JSON answer."""
    mime_type = IMAGE_SUFFIXES[image_path.suffix.lower()]
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    resp = client.chat.completions.create(
        model=model,
        temperature=1.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        ],
    )
    content = resp.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError(f"Model returned no text content (got {type(content).__name__})")
    return parse_json_response(content)


def find_images(inputs: Iterable[Path]) -> list[Path]:
    """Expand files and directories (recursively) into a sorted list of image paths.

    Raises:
        FileNotFoundError: If an input path does not exist.
        ValueError: If two images share a filename stem, since outputs are keyed by stem.
    """
    images: list[Path] = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        images.extend(p for p in candidates if p.suffix.lower() in IMAGE_SUFFIXES)

    seen: dict[str, Path] = {}
    for image in images:
        if image.stem in seen:
            raise ValueError(
                f"Duplicate image name {image.stem!r}: {seen[image.stem]} and {image}. "
                "Outputs are written as <name>.json, so names must be unique."
            )
        seen[image.stem] = image
    return images


def resolve_model(client: OpenAI, requested: str) -> str:
    """Return the requested model if the service offers it.

    Raises:
        SystemExit: If the service doesn't offer the requested model.
    """
    available = sorted(m.id for m in client.models.list())
    if requested in available:
        return requested
    raise SystemExit(
        f"Model {requested!r} is not available. Choose one with --model or ${MODEL_ENV}. "
        f"Available: {', '.join(available) or 'none'}"
    )


def run(client: OpenAI, model: str, images: list[Path], out_dir: Path, prompt: str) -> int:
    """Classify each image not already in out_dir; return the number of failures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for image in images:
        out_path = out_dir / f"{image.stem}.json"
        if out_path.exists():
            continue
        try:
            result = classify_image(client, model, image, prompt)
        except FATAL_ERRORS:
            raise
        except (openai.OpenAIError, ValueError) as exc:
            failures += 1
            print(f"FAILED {image}: {exc}", file=sys.stderr)
            continue
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"image": str(image), **result}), flush=True)
    return failures


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify Rockd images with the model hosted at "
            f"{BASE_URL}. Requires ${API_KEY_ENV}. Images that already have a result "
            "in --out are skipped, so the command can be re-run as new images arrive."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files or directories")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Result directory")
    parser.add_argument(
        "--model",
        default=os.environ.get(MODEL_ENV, DEFAULT_MODEL),
        help=f"Or ${MODEL_ENV} (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--prompt", type=Path, help="Prompt file (default: bundled classifier)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise SystemExit(
            f"${API_KEY_ENV} is not set. Create a key under Settings > Account at "
            "https://llm.chtc.wisc.edu and export it."
        )
    prompt = args.prompt.read_text() if args.prompt else default_prompt()
    images = find_images(args.inputs)
    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        model = resolve_model(client, args.model)
        failures = run(client, model, images, args.out, prompt)
    except FATAL_ERRORS as exc:
        detail = exc.body.get("detail") if isinstance(exc.body, dict) else exc.body
        raise SystemExit(
            f"Request rejected by {BASE_URL} (HTTP {exc.status_code}): {detail}"
        ) from exc
    if failures:
        raise SystemExit(f"{failures} of {len(images)} images failed; re-run to retry them.")
