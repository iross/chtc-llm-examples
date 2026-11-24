"""OCR PDFs to Markdown with the vision model hosted on CHTC's Open WebUI service."""

import argparse
import base64
import io
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import openai
import pypdfium2 as pdfium
from openai import OpenAI

BASE_URL = "https://llm.chtc.wisc.edu/api"
OLLAMA_EXAMPLE_URL = "http://localhost:11434/v1"
API_KEY_ENV = "CHTC_LLM_API_KEY"
MODEL_ENV = "CHTC_LLM_MODEL"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_DPI = 150
REQUEST_TIMEOUT_SECONDS = 300.0

PROMPT = """\
Transcribe all text on this page as Markdown, in reading order.
- Use Markdown headings, lists, and tables where the page uses them.
- Keep the original wording, spelling, and numbers exactly. Do not summarize or correct.
- Describe figures and images in one short line, like: [Figure: map of Wisconsin counties]
- Skip page headers, footers, and page numbers.
Output only the transcription, with no commentary."""

FATAL_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)


@dataclass(frozen=True)
class Settings:
    """How each page is rendered and sent to the model."""

    model: str
    dpi: int = DEFAULT_DPI
    thinking: bool = True


@dataclass(frozen=True)
class PageResult:
    """A page's transcription and how much the model generated to produce it.

    reasoning_chars is the length of the reasoning text the server returned alongside the
    answer. It's counted in characters because Ollama doesn't report reasoning tokens.
    """

    text: str
    completion_tokens: int
    reasoning_chars: int


def reasoning_text(message: object) -> str:
    """Return the reasoning a server attached to a reply message, or "" if none.

    vLLM uses `reasoning_content`; Ollama uses `reasoning`.
    """
    for field in ("reasoning_content", "reasoning"):
        value = getattr(message, field, None)
        if isinstance(value, str) and value:
            return value
    return ""


def render_page(pdf: pdfium.PdfDocument, index: int, dpi: int) -> bytes:
    """Render one zero-indexed page of an open PDF to PNG bytes."""
    image = pdf[index].render(scale=dpi / 72).to_pil()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def ocr_page(client: OpenAI, png: bytes, settings: Settings) -> PageResult:
    """Send one rendered page to the model and return its Markdown transcription.

    With settings.thinking off, asks vLLM's chat template to skip Qwen's reasoning step.
    Servers that don't use that setting (Ollama) ignore it and the model reasons anyway.

    Raises:
        ValueError: If the reply is empty or was cut off by the token limit.
    """
    image_b64 = base64.b64encode(png).decode("ascii")
    extra_body = None if settings.thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    resp = client.chat.completions.create(
        model=settings.model,
        temperature=0.0,
        extra_body=extra_body,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    )
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("Reply was cut off by the token limit; try a lower --dpi")
    content = (choice.message.content or "").strip()
    if not content:
        raise ValueError(f"Model returned no text (finish_reason={choice.finish_reason})")
    return PageResult(
        text=content,
        completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        reasoning_chars=len(reasoning_text(choice.message)),
    )


def find_pdfs(inputs: Iterable[Path]) -> list[Path]:
    """Expand files and directories (recursively) into a list of PDF paths.

    Raises:
        FileNotFoundError: If an input path does not exist.
        ValueError: If two PDFs share a filename stem, since outputs are keyed by stem.
    """
    pdfs: list[Path] = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        pdfs.extend(p for p in candidates if p.suffix.lower() == ".pdf")

    seen: dict[str, Path] = {}
    for pdf in pdfs:
        if pdf.stem in seen:
            raise ValueError(
                f"Duplicate PDF name {pdf.stem!r}: {seen[pdf.stem]} and {pdf}. "
                "Outputs are written as <name>.md, so names must be unique."
            )
        seen[pdf.stem] = pdf
    return pdfs


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


def ocr_pdf(client: OpenAI, settings: Settings, pdf_path: Path, out_dir: Path) -> int:
    """OCR each page of one PDF not already in out_dir; return the number of failed pages.

    Each page is saved as out_dir/<name>/page-NNNN.md. Once every page has a file, they are
    joined into out_dir/<name>.md. Each page's time and token use are logged to stderr.
    """
    pdf = pdfium.PdfDocument(pdf_path)
    pages_dir = out_dir / pdf_path.stem
    pages_dir.mkdir(parents=True, exist_ok=True)
    try:
        page_paths = [pages_dir / f"page-{i + 1:04d}.md" for i in range(len(pdf))]
        failures = 0
        for index, page_path in enumerate(page_paths):
            if page_path.exists():
                continue
            label = f"{pdf_path} page {index + 1}/{len(page_paths)}"
            print(f"{label} ...", end=" ", file=sys.stderr, flush=True)
            start = time.monotonic()
            try:
                result = ocr_page(client, render_page(pdf, index, settings.dpi), settings)
            except FATAL_ERRORS:
                print(file=sys.stderr)
                raise
            except (openai.OpenAIError, ValueError) as exc:
                failures += 1
                print(f"FAILED: {exc}", file=sys.stderr)
                continue
            print(
                f"{time.monotonic() - start:.1f}s, {result.completion_tokens} tokens "
                f"({result.reasoning_chars} chars of reasoning)",
                file=sys.stderr,
            )
            page_path.write_text(result.text + "\n")
    finally:
        pdf.close()

    if failures == 0:
        sections = [f"<!-- page {i + 1} -->\n\n{p.read_text()}" for i, p in enumerate(page_paths)]
        combined = out_dir / f"{pdf_path.stem}.md"
        combined.write_text("\n".join(sections))
        print(combined, flush=True)
    return failures


def run(client: OpenAI, settings: Settings, pdfs: list[Path], out_dir: Path) -> int:
    """OCR every PDF without a finished result in out_dir; return the number of failed pages.

    A PDF that pdfium can't open counts as one failure.
    """
    failures = 0
    for pdf_path in pdfs:
        if (out_dir / f"{pdf_path.stem}.md").exists():
            continue
        try:
            failures += ocr_pdf(client, settings, pdf_path, out_dir)
        except pdfium.PdfiumError as exc:
            failures += 1
            print(f"FAILED {pdf_path}: could not read PDF ({exc})", file=sys.stderr)
    return failures


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"OCR PDFs to Markdown with the model hosted at {BASE_URL} (needs ${API_KEY_ENV}), "
            "or any OpenAI-compatible server such as Ollama via --base-url. Pages that already "
            "have a result in --out are skipped, so re-running retries only what failed."
        )
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help=f"OpenAI-compatible API root, e.g. {OLLAMA_EXAMPLE_URL} (default: {BASE_URL})",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="PDF files or directories")
    parser.add_argument("--out", type=Path, default=Path("ocr"), help="Output directory")
    parser.add_argument(
        "--model",
        default=os.environ.get(MODEL_ENV, DEFAULT_MODEL),
        help=f"Or ${MODEL_ENV} (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI, help=f"Render resolution (default: {DEFAULT_DPI})"
    )
    parser.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Turn off the model's reasoning step (faster; compare output quality)",
    )
    args = parser.parse_args(argv)
    if not 36 <= args.dpi <= 600:
        parser.error("--dpi must be between 36 and 600")
    return args


def api_key_for(base_url: str) -> str:
    """Return the API key to send. Only the CHTC service requires one; Ollama ignores it.

    Raises:
        SystemExit: If base_url is the CHTC service and ${API_KEY_ENV} is unset.
    """
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key and base_url == BASE_URL:
        raise SystemExit(
            f"${API_KEY_ENV} is not set. Create a key under Settings > Account at "
            "https://llm.chtc.wisc.edu and export it."
        )
    return api_key or "unused"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    api_key = api_key_for(args.base_url)
    pdfs = find_pdfs(args.inputs)
    if not pdfs:
        raise SystemExit("No PDF files found in the given inputs.")
    client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        settings = Settings(resolve_model(client, args.model), args.dpi, args.thinking)
        failures = run(client, settings, pdfs, args.out)
    except openai.APIConnectionError as exc:
        raise SystemExit(
            f"Could not connect to {args.base_url} ({exc}). Check --base-url; for Ollama, "
            "make sure `ollama serve` is running."
        ) from exc
    except FATAL_ERRORS as exc:
        body = exc.body if isinstance(exc.body, dict) else {}
        detail = body.get("detail") or body.get("error") or exc.body
        raise SystemExit(
            f"Request rejected by {args.base_url} (HTTP {exc.status_code}): {detail}"
        ) from exc
    if failures:
        raise SystemExit(f"{failures} failures (see above); re-run to retry them.")
