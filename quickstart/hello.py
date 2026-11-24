# /// script
# requires-python = ">=3.13"
# dependencies = ["openai==3.55.1"]
# ///
"""List the models your API key can use, then ask one of them a question.

Run it with:  uv run hello.py
"""

import os
import sys

from openai import OpenAI

api_key = os.environ.get("CHTC_LLM_API_KEY")
if not api_key:
    sys.exit("Set your API key first:  export CHTC_LLM_API_KEY=sk-...")

client = OpenAI(base_url="https://llm.chtc.wisc.edu/api", api_key=api_key)

print("Models you can use:")
for model in client.models.list():
    print("  ", model.id)

response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "In one sentence, what is high-throughput computing?"}],
)
answer = response.choices[0].message.content or ""
print("\nAnswer:", answer.strip())
