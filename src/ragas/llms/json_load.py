from __future__ import annotations

import asyncio
import json
import logging
import typing as t
from dataclasses import dataclass
from functools import partial
import re

from ragas.run_config import RunConfig, add_async_retry, add_retry

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from ragas.llms.base import BaseRagasLLM

def preprocess_json_string(s: str) -> str:
    s = s.strip()
    if not s.startswith('{'): s = '{' + s
    if not s.endswith('}'): s = s + '}'
    s = s.replace("'", '"')
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    s = re.sub(r'(\w+)(?=\s*:)', r'"\1"', s)
    return s

def load_as_json(text) -> t.Dict:
    """
    Validate and return given text as json
    """
    try:
        return json.loads(preprocess_json_string(text))
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid json after preprocessing: {e}")
        return {}

JSON_PROMPT = """\
Rewrite the input into valid json

Input:
{{
    "name": "John Doe",
    "age": 30,
    "isStudent": false
    "address": {{
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA",
    }}
    "hobbies": ["reading", "swimming", "cycling"]
}}
Output:
{{
    "name": "John Doe",
    "age": 30,
    "isStudent": false,
    "address": {{
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA"
    }},
    "hobbies": ["reading", "swimming", "cycling"]
}}


Input:
{{
    "statement": "The Earth is also known as "Terra" "
}}
Output:
{{
    "statement": "The Earth is also known as 'Terra'"
}}

Input:
{input}

Output:
"""

@dataclass
class JsonLoader:
    max_retries: int = 2

    def _safe_load(self, text: str, llm: BaseRagasLLM, callbacks: Callbacks = None):
        retry = 0
        while retry <= self.max_retries:
            try:
                _json = self._load_all_jsons(text)
                return _json[0] if len(_json) == 1 else _json
            except ValueError:
                from ragas.llms.prompt import PromptValue
                results = llm.generate_text(
                    PromptValue(prompt_str=JSON_PROMPT.format(input=text)),
                    n=1,
                    callbacks=callbacks,
                )
                text = results.generations[0][0].text
            retry += 1
        return {}

    async def _asafe_load(self, text: str, llm: BaseRagasLLM, callbacks: Callbacks = None):
        retry = 0
        while retry <= self.max_retries:
            try:
                _json = self._load_all_jsons(text)
                return _json[0] if len(_json) == 1 else _json
            except ValueError:
                from ragas.llms.prompt import PromptValue
                results = await llm.agenerate_text(
                    PromptValue(prompt_str=JSON_PROMPT.format(input=text)),
                    n=1,
                    callbacks=callbacks,
                )
                text = results.generations[0][0].text
            retry += 1
        return {}

    async def safe_load(
        self,
        text: str,
        llm: BaseRagasLLM,
        callbacks: Callbacks = None,
        is_async: bool = True,
        run_config: RunConfig = RunConfig(),
    ) -> t.Union[t.Dict, t.List]:
        if is_async:
            _asafe_load_with_retry = add_async_retry(self._asafe_load, run_config)
            return await _asafe_load_with_retry(text=text, llm=llm, callbacks=callbacks)
        else:
            _safe_load_with_retry = add_retry(self._safe_load, run_config)
            loop = asyncio.get_event_loop()
            safe_load = partial(_safe_load_with_retry, text=text, llm=llm, callbacks=callbacks)
            return await loop.run_in_executor(None, safe_load)

    def _load_all_jsons(self, text):
        start, end = self._find_outermost_json(text)
        try:
            _json = json.loads(text[start:end])
        except json.JSONDecodeError:
            _json = load_as_json(text[start:end])
        
        text = text.replace(text[start:end], "", 1)
        start, end = self._find_outermost_json(text)
        if (start, end) == (-1, -1):
            return [_json]
        else:
            return [_json] + self._load_all_jsons(text)

    def _find_outermost_json(self, text):
        stack = []
        start_index = -1
        in_string = False
        escape_char = False

        for i, char in enumerate(text):
            if not in_string:
                if char in "{[":
                    if len(stack) == 0:
                        start_index = i
                    stack.append(char)
                elif char in "}]":
                    if len(stack) > 0:
                        last = stack.pop()
                        if (char == "}" and last != "{") or (char == "]" and last != "["):
                            break
                    if len(stack) == 0 and start_index != -1:
                        return start_index, i + 1
            if char == '"' and not escape_char:
                in_string = not in_string
            escape_char = char == '\\' and not escape_char

        return -1, -1

json_loader = JsonLoader()
