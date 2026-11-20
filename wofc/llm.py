"""The language-model classifier, and the checks that decide whether its answer can be used without a person.

The model must return JSON with a component, a failure and the words from the note that show it. The answer is
accepted only when both labels are in the taxonomy, the pair is a real one, and the quoted evidence is actually in
the note. Anything else goes to "needs review" instead of being stored.
"""
from __future__ import annotations

import json
import os
import re

from .taxonomy import COMPONENTS, FAILURES, VALID, is_valid

PROMPT = """Classify this maintenance work order.

Components: {components}
Failure modes: {failures}
Allowed pairs: {pairs}

Work order: "{text}"

Reply with JSON only, in this form:
{{"component": "<one component>", "failure": "<one failure mode>", "evidence": "<the exact words from the work order that show it>"}}"""


def prompt(text: str) -> str:
    pairs = "; ".join(f"{c}: {', '.join(fs)}" for c, fs in VALID.items())
    return PROMPT.format(components=", ".join(COMPONENTS), failures=", ".join(FAILURES), pairs=pairs,
                         text=text.replace('"', "'"))


def parse(raw: str) -> dict | None:
    """The first JSON object in the reply, or None."""
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate(text: str, data: dict | None) -> tuple[dict, list[str]]:
    """Returns the labels to use (None where unusable) and the reasons a person must look."""
    if data is None:
        return {"component": None, "failure": None}, ["reply was not JSON"]
    component = str(data.get("component", "")).strip().lower()
    failure = str(data.get("failure", "")).strip().lower()
    evidence = " ".join(str(data.get("evidence", "")).lower().split())
    problems = []
    if component not in COMPONENTS:
        problems.append(f"unknown component '{component}'")
    if failure not in FAILURES:
        problems.append(f"unknown failure '{failure}'")
    if not problems and not is_valid(component, failure):
        problems.append(f"'{component}' cannot have '{failure}'")
    if not evidence or evidence not in " ".join(text.lower().split()):
        problems.append("evidence is not quoted from the work order")
    if problems:
        return {"component": None, "failure": None}, problems
    return {"component": component, "failure": failure}, []


class LocalModel:
    """A Hugging Face chat model on this machine, greedy decoding."""

    def __init__(self, name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        # bfloat16 on CPU halves memory (a 1.5B model needs about 3 GB instead of 6).
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16).eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def __call__(self, text: str) -> str:
        import torch

        chat = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class GeminiModel:
    """Gemini with JSON output and temperature 0, through Google Cloud or an API key."""

    def __init__(self, model: str | None = None, client=None):
        self.name = model or os.getenv("MODEL", "gemini-3.5-flash")
        if client is None:
            from google import genai

            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            client = (genai.Client(vertexai=True, project=project, location=os.getenv("MODEL_LOCATION", "global"))
                      if project else genai.Client(api_key=os.environ["GOOGLE_API_KEY"]))
        self.client = client

    def __call__(self, text: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
        return self.client.models.generate_content(model=self.name, contents=text, config=config).text or ""


class LLMClassifier:
    def __init__(self, model):
        self.model = model
        self._cache: dict[str, dict] = {}  # greedy decoding: the same note always gets the same answer

    def classify(self, text: str) -> dict:
        if text not in self._cache:
            raw = self.model(prompt(text))
            labels, problems = validate(text, parse(raw))
            self._cache[text] = {**labels, "problems": problems, "raw": raw.strip()}
        return self._cache[text]


def cascade(first, then):
    """Rules first, where they are exact; the model only for notes the rules cannot label."""
    def classify(text: str) -> dict:
        result = first(text)
        if result.get("component") and result.get("failure"):
            return {**result, "problems": [], "raw": "", "by": "rules"}
        return {**then(text), "by": "model"}
    return classify
