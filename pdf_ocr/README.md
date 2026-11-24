# pdf-ocr: turn PDFs into text

This example turns PDFs into Markdown text. It works on scanned documents and photos of pages
that have no text you can select. It doesn't read text out of the PDF file directly; it turns
each page into an image and asks the `qwen3.8-27b` model on CHTC's LLM service to read it.
Because the model sees the page, it can also keep headings, lists, and tables.

**New to the CHTC LLM service?** Do the [setup steps in the main README](../README.md) first.
You need an API key in `CHTC_LLM_API_KEY` and [uv](https://docs.astral.sh/uv/) installed.

## Try it

From this folder, run it on a PDF:

```bash
uv run pdf-ocr path/to/document.pdf --out ocr/
```

The first run takes a little longer while uv installs the packages. As it goes, it prints one
line per page:

```text
path/to/document.pdf page 1/2 ... 2.1s, 93 tokens (252 chars of reasoning)
path/to/document.pdf page 2/2 ... 2.9s, 123 tokens (288 chars of reasoning)
ocr/document.md
```

The result is `ocr/document.md`. Each page starts with a `<!-- page N -->` marker, so you can
find your place:

```markdown
<!-- page 2 -->

Table 1. Site summary

| Site | Depth (m) | pH |
|------|-----------|----|
| A-1  | 2.3       | 7.1 |
```

You can also give it a folder. It finds every `.pdf` inside, including in subfolders.

## How it works

The code is in [`src/pdf_ocr/ocr.py`](src/pdf_ocr/ocr.py). For each page it:

1. Draws the page as a PNG image, using [pypdfium2](https://pypi.org/project/pypdfium2/).
2. Sends one chat request with the image and the instructions in `PROMPT` (near the top of
   `ocr.py`), which ask for an exact Markdown transcription.
3. Saves the reply as `ocr/<name>/page-0001.md`, `page-0002.md`, and so on.

When every page is done, it joins them into `ocr/<name>.md`.

To change what the model is asked to do (for example, to keep page numbers or skip figure
descriptions), edit `PROMPT`.

## Running it again

Finished pages are saved one at a time and skipped on the next run. If a page fails (the
service returns an error, the reply is empty, or the reply was cut off), the error is printed,
the run continues, and the command exits with an error at the end. Run the same command again
to retry only the failed pages. The combined `ocr/<name>.md` is written once every page has
succeeded.

If the API key is rejected, the run stops right away. If the model doesn't exist, it stops and
lists the ones you can use.

## Options

```bash
uv run pdf-ocr --help
```

| Option | Default | What it does |
|---|---|---|
| `--out` | `ocr/` | Folder for the results |
| `--model` | `qwen3.8-27b` | Model to use (or set `CHTC_LLM_MODEL`). It must accept images. |
| `--dpi` | `150` | How sharp the page images are. Raise it for small print or poor scans; lower it if pages fail with a token-limit error. |
| `--no-thinking` | off | Skip the model's reasoning step (see below) |
| `--base-url` | CHTC | Use a different server, such as Ollama on your own computer (see below) |

## Comparing settings and models

Each run writes only into its `--out` folder. To compare two settings, give each its own
folder, then compare the folders:

```bash
uv run pdf-ocr scans/ --out ocr-a/
uv run pdf-ocr scans/ --out ocr-b/ --no-thinking
diff -r ocr-a/ ocr-b/
```

Don't reuse a folder for a different setting: pages that are already there are skipped, not
redone.

### Thinking on or off

Qwen models normally "think" (write out their reasoning) before they answer. The progress line
shows how much reasoning text came back for each page. `--no-thinking` asks the model to skip
that step. It's faster and uses fewer tokens. Compare the outputs to see whether quality
changes for your documents.

### Local models with Ollama

`--base-url` points the tool at another server that speaks the same API, such as
[Ollama](https://ollama.com) on your own computer. Start `ollama serve`, then:

```bash
uv run pdf-ocr scans/ --out ocr-local/ \
    --base-url http://localhost:11434/v1 --model qwen3-vl:8b
```

No API key is needed for Ollama. Run `ollama list` to see your models. The model has to accept
images.

Notes from testing with Ollama 0.40.6 on a Mac:

- It's much slower than CHTC. On a 2-page test PDF, `qwen3-vl:8b` took 27s and 91s per page,
  against 2–8s for `qwen3.8-27b` on CHTC.
- `--no-thinking` has no effect with Ollama, and `qwen3-vl:8b` always thinks. To compare
  without thinking locally, use a model that doesn't think (for example an `-instruct` tag).
- If output stops partway down a page, the model ran out of room for the page image plus the
  reply. Lower `--dpi`, or use a model set up with a larger context (`num_ctx`).

## Development

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run ty check
```

`pytest` doesn't call the service. The live test does, and needs `CHTC_LLM_API_KEY`:

```bash
uv run pytest -q -m live
```

It sends two requests: it transcribes `tests/data/field_report.pdf`, a one-page scan with
no text layer, with thinking on and off, and checks that the expected text comes back.
