# rockd-live: sort photos into categories

[Rockd](https://rockd.org) is a geology app where people post photos from the field. This
example shows each photo to a vision model on CHTC's LLM service and asks it which of four
categories the photo belongs to:

| Label | Meaning |
|---|---|
| `IN_SITU_ROCKS` | Rocks or geological features in their natural setting, as seen in the field |
| `INDOOR_ROCKS` | Specimens taken out of their setting: in a hand, on a table, in a collection |
| `ANTHROPIC_STONE_USE` | Stone used by people: buildings, sculpture, decoration, landscaping |
| `OFF_TOPIC` | Anything without a clear geological subject, such as selfies |

The same pattern works for any "look at each image and answer in a fixed format" task. Change
the prompt and you have a different classifier.

**New to the CHTC LLM service?** Do the [setup steps in the main README](../README.md) first.
You need an API key in `CHTC_LLM_API_KEY` and [uv](https://docs.astral.sh/uv/) installed.

## Try it

From this folder, classify the public-domain sample photo of a rock outcrop:

```bash
uv run rockd-live images/rock.jpg --out my-results/
```

The first run takes a little longer while uv installs the packages. Each answer is printed as
one line and saved as `my-results/<photo name>.json`:

```json
{
  "reasoning": "The image shows a large natural rock outcrop in an outdoor field setting...",
  "label": "IN_SITU_ROCKS",
  "confidence": 0.97
}
```

To classify your own photos (`.jpg`, `.jpeg`, or `.png`), put them in a folder and pass the
folder instead, for example `uv run rockd-live photos/ --out my-results/`. Each photo is one
request, so start with a handful before running a large folder.

## How it works

The code is in [`src/rockd_live/classify.py`](src/rockd_live/classify.py). For each photo it:

1. Reads the image file and base64-encodes it.
2. Sends one chat request with two parts: the instructions in
   [`src/rockd_live/prompts/classify.txt`](src/rockd_live/prompts/classify.txt), and the image.
3. The prompt asks the model to reply with only a JSON object. The code pulls that object out
   of the reply, even if the model wrapped it in extra text, and saves it.

## Options

```bash
uv run rockd-live --help
```

- **Inputs** can be image files or folders. Folders are searched, including subfolders, for
  `.jpg`, `.jpeg`, and `.png` files. Each photo needs a different file name, since results are
  saved by name.
- **`--out`** is the folder for results (default `results/`).
- **`--model`** picks the model (default `qwen3.8-27b`). You can also set `CHTC_LLM_MODEL`. If
  the model doesn't exist, the tool stops and lists the ones that do.
- **`--prompt`** uses your own prompt file instead of the bundled one. Ask for JSON in the
  prompt, since that's what gets saved:

  ```bash
  uv run rockd-live photos/ --out descriptions/ --prompt my_prompt.txt
  ```

## Running it again

Photos that already have a result file are skipped. That means:

- If a photo fails (the service returns an error, or the model's reply isn't JSON), the error is
  printed, the run continues, and the command exits with an error at the end. Run the same
  command again to retry only the failed photos.
- When new photos arrive in a folder, run the same command again and only the new ones are
  sent. Running it on a schedule (for example, from cron) turns it into a simple live service.

If the API key is rejected, the run stops right away.

## Development

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run ty check
```

`pytest` doesn't call the service. The live test does, and needs `CHTC_LLM_API_KEY`:

```bash
uv run pytest -q -m live
```

It sends one request: it classifies `images/rock.jpg` and checks the label is
`IN_SITU_ROCKS`.
