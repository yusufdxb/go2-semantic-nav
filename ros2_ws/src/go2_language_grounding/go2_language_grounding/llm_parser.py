"""Optional LLM-backed parser for queries that fall outside the rule grammar.

This is a lazy-loaded fallback — queries that the rule-based parser fails on
can be escalated here. The default implementation uses a small instruction-tuned
model (Qwen2-0.5B-Instruct) via transformers; the caller is expected to keep it
off the hot path.

Graceful degradation: if transformers/torch aren't importable, the parser
returns None and the caller falls through to rule-based behavior.
"""

from __future__ import annotations

import json
from typing import Optional

from .query_parser import ParsedQuery

_SYSTEM_PROMPT = """You parse natural-language robot navigation queries into structured JSON.
The output MUST be valid JSON with EXACTLY these keys:
  target_noun         (string, required, noun the robot should approach)
  attribute           (string, may be "")
  relation            (string, one of: "", "near", "left_of", "right_of",
                       "in_front_of", "behind", "above", "below", "on", "beside")
  reference_noun      (string, may be "")
  reference_attribute (string, may be "")
  stand_off_m         (number or null; null means use default)

Only output the JSON object. No prose, no markdown fences.

Examples:
Q: go stand next to the red chair
{"target_noun": "chair", "attribute": "red", "relation": "near", "reference_noun": "", "reference_attribute": "", "stand_off_m": null}

Q: the lamp on the wooden dining table
{"target_noun": "lamp", "attribute": "", "relation": "on", "reference_noun": "table", "reference_attribute": "wooden dining", "stand_off_m": null}

Q: approach the whiteboard, stay 0.5 m back
{"target_noun": "whiteboard", "attribute": "", "relation": "", "reference_noun": "", "reference_attribute": "", "stand_off_m": 0.5}
"""


class LlmParser:
    """Thin wrapper around a small instruction-tuned model.

    Thread-safe only if the caller serializes `parse()` calls. Designed to be
    constructed once at node startup and invoked on the async grounding path.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2-0.5B-Instruct", device: str = "cuda:0"):
        self._model_id = model_id
        self._device = device
        self._model = None
        self._tokenizer = None

    def load(self) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return False
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id).to(self._device).eval()
        return True

    def parse(self, query: str, timeout_s: float = 5.0) -> Optional[ParsedQuery]:
        if self._model is None or self._tokenizer is None:
            return None
        import torch

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {query}"},
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"{_SYSTEM_PROMPT}\n\nQ: {query}\n"

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Extract the first {...} JSON blob.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

        return ParsedQuery(
            raw=query,
            target_noun=str(obj.get("target_noun", "")).strip(),
            attribute=str(obj.get("attribute", "")).strip(),
            relation=(obj.get("relation") or None) or None,
            reference_noun=str(obj.get("reference_noun", "")).strip(),
            reference_attribute=str(obj.get("reference_attribute", "")).strip(),
            stand_off_m=(obj.get("stand_off_m") if obj.get("stand_off_m") is not None else None),
        )
