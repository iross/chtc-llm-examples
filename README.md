nb: READMEs and documentation are Claude-written as of 2026-11-24.

# CHTC LLM examples

CHTC runs large language models (LLMs) at <https://llm.chtc.wisc.edu>. You can chat with them
in your browser, or call them from your own code. These examples show how to call them from
Python. You don't need a GPU, and the model doesn't run on your computer.

If you've never done this before, follow the four steps below in order. They take about ten
minutes.

## 1. Sign in

Go to <https://llm.chtc.wisc.edu> and sign in with **CILogon**. Pick your institution (for
example, University of Wisconsin-Madison) and log in with your usual campus account (your
NetID). There's no separate password to create.

Once you're in, you can chat with the models in your browser. That's a good way to try a
prompt before you write code for it.

> If you sign in but can't see any models, your account may still need to be activated. Please be patient or ping Ian if it's been more than a day or two.

## 2. Create an API key

Your code proves who you are with an **API key**: a long password that starts with `sk-`.

1. Click your name or picture (bottom left), then **Settings**.
2. Go to **Account**, find **API Keys**, and create a new key.
3. Copy the key. You may not be able to see it again later, so save it somewhere safe, like a
   password manager.

Treat the key like a password. Don't email it, paste it into shared documents, or commit it to
Git. If it leaks, delete it on the same page and create a new one.

## 3. Install uv and save your key

The examples use [uv](https://docs.astral.sh/uv/), which installs Python and the packages each
example needs. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                  # macOS or Linux
```

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

Then put your key in an environment variable named `CHTC_LLM_API_KEY`. Every example reads it
from there, so it never has to be written into the code.

```bash
export CHTC_LLM_API_KEY=sk-...                                   # macOS or Linux
```

```powershell
$env:CHTC_LLM_API_KEY = "sk-..."                                 # Windows PowerShell
```

This lasts until you close the terminal. To make it permanent on macOS or Linux, add the
`export` line to `~/.zshrc` (macOS) or `~/.bashrc` (Linux).

## 4. Run your first request

```bash
cd quickstart
uv run hello.py
```

You should see something like:

```text
Models you can use:
   gpt-oss-20b
   qwen3-32b
   qwen3.8-27b
   ...

Answer: High-throughput computing is a style of distributed computing that processes a large
number of independent tasks by running many jobs in parallel across many computers.
```

If that worked, you're set up. If not, see [Troubleshooting](#troubleshooting).

## How the code works

The service speaks the same API as OpenAI, so you use the standard
[`openai`](https://pypi.org/project/openai/) Python package and point it at CHTC instead. Here's
all of [`quickstart/hello.py`](quickstart/hello.py) that matters:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://llm.chtc.wisc.edu/api",       # CHTC instead of OpenAI
    api_key=os.environ["CHTC_LLM_API_KEY"],         # the key from step 2
)

response = client.chat.completions.create(
    model="qwen3.8-27b",                             # any id from client.models.list()
    messages=[{"role": "user", "content": "In one sentence, what is high-throughput computing?"}],
)
print(response.choices[0].message.content)
```

- `base_url` is the only CHTC-specific part. Any tutorial or library written for OpenAI's
  "chat completions" API works the same way once you change it.
- `model` picks which model answers. `client.models.list()` returns the ones your key can use.
  The examples here use `qwen3.8-27b`, which reads both text and images.
- `messages` is the conversation so far. To add instructions, put a
  `{"role": "system", "content": "..."}` message first. To continue a conversation, append the
  model's reply as `{"role": "assistant", ...}` and then your next question.

To send an image, make the message content a list with the text and the image. The image is
sent inline, base64-encoded:

```python
import base64

image_b64 = base64.b64encode(open("photo.jpg", "rb").read()).decode()
response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this photo in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ],
    }],
)
```

To use the package in your own project, install it with `uv add openai` (or
`pip install openai`).

## Examples

Each folder is a self-contained project with its own README. Start with `quickstart`.

| Folder | What it does |
|---|---|
| [`quickstart`](quickstart) | One small script: lists your models and asks one question. |
| [`pdf_ocr`](pdf_ocr) | Turns PDFs, including scans, into Markdown text by showing each page to the model. |
| [`rockd_live`](rockd_live) | Sorts photos into categories and saves a JSON answer for each one. |

## Troubleshooting

| What you see | What it means |
|---|---|
| `Set your API key first` or `CHTC_LLM_API_KEY is not set` | The environment variable isn't set in this terminal. Repeat the `export` from step 3. |
| `401` or `Not authenticated` | The key is wrong, was deleted, or got cut off when you copied it. Create a new one. |
| `Model '...' is not available` / `404` | That model id doesn't exist for your key. Use one from the list `hello.py` prints. |
| `command not found: uv` | Open a new terminal after installing uv, or see the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/). |

## Before you send real data

The service is shared with other CHTC users and runs on CHTC's servers. If your data is
restricted (for example, health or student records), check with CHTC before sending it.

## Tests and CI

Each example has its own tests (see the Development section of its README). GitHub Actions
runs them on every push and pull request:

- **checks** runs lint, type checks, and unit tests for each example. These don't call the
  service.
- **live tests** run `quickstart/hello.py` and each example's `pytest -m live` against the real
  service. They use a `CHTC_LLM_API_KEY` secret stored in the repository's `live` environment
  (**Settings → Environments → live**), and are skipped for pull requests from forks.

If you're changing the examples, install [prek](https://prek.j178.dev) and run `prek install`
once in your clone. It then checks each commit with the same ruff and ty checks CI uses, plus
whitespace, file-size, and GitHub Actions security checks (configured in `prek.toml`). To check
everything without committing, run `prek run --all-files`. To update the hook versions, run
`prek update --cooldown-days 7`.
