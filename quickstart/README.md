# quickstart

[`hello.py`](hello.py) is the smallest useful program for CHTC's LLM service. It prints the
models your API key can use, then asks `qwen3.8-27b` one question and prints the answer.

Do the [setup steps in the main README](../README.md) first, then:

```bash
uv run hello.py
```

There's no project file here. The block at the top of `hello.py` lists the one package it needs
(`openai`), and `uv run` installs it automatically.

## Make it your own

Copy `hello.py` and change it:

- Change the question in `messages`.
- Change `model` to another id from the printed list.
- Add instructions with a system message, before the question:

  ```python
  messages=[
      {"role": "system", "content": "Answer like a pirate."},
      {"role": "user", "content": "In one sentence, what is high-throughput computing?"},
  ]
  ```

The [main README](../README.md#how-the-code-works) explains each part and shows how to send an
image. For bigger programs built the same way, see [`pdf_ocr`](../pdf_ocr) and
[`rockd_live`](../rockd_live).
