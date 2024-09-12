from __future__ import annotations
import logging
import typing as t
import os
import shutil
import fcntl  # For Unix-based file locking
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from tenacity import retry, stop_after_attempt, wait_exponential

from ragas.llms.json_load import json_loader
from ragas.testset.prompts import keyphrase_extraction_prompt

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM
    from ragas.llms.prompt import Prompt
    from ragas.testset.docstore import Node

logger = logging.getLogger(__name__)

@dataclass
class Extractor(ABC):
    llm: BaseRagasLLM

    @abstractmethod
    async def extract(self, node: Node, is_async: bool = True) -> t.Any:
        ...

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the extractor to a different language.
        """
        raise NotImplementedError("adapt() is not implemented for {} Extractor")

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the extractor prompts to a path.
        """
        raise NotImplementedError("adapt() is not implemented for {} Extractor")

@dataclass
class KeyphraseExtractor(Extractor):
    extractor_prompt: Prompt = field(
        default_factory=lambda: keyphrase_extraction_prompt
    )

    async def extract(self, node: Node, is_async: bool = True) -> t.List[str]:
        try:
            prompt = self.extractor_prompt.format(text=node.page_content)
            logger.info(f"Generated prompt for LLM: {prompt}")
            results = await self.llm.generate(prompt=prompt, is_async=is_async)

            # Check if the LLM returned valid results
            if not results or not results.generations or not results.generations[0]:
                logger.error("LLM returned an empty or invalid response")
                raise ValueError("LLM returned an empty or invalid response")

            generated_text = results.generations[0][0].text.strip()
            logger.info(f"LLM generated output: {generated_text}")

            # Ensure non-empty response
            if not generated_text:
                logger.error("LLM returned an empty response")
                raise ValueError("LLM returned an empty response")

            # Try to load the JSON safely
            keyphrases = await json_loader.safe_load(generated_text, llm=self.llm, is_async=is_async)

            # Ensure the returned keyphrases are a valid dict
            if not isinstance(keyphrases, dict):
                logger.error(f"Invalid keyphrases format received: {keyphrases}")
                raise ValueError("LLM returned invalid keyphrases format")

            logger.debug(f"Extracted keyphrases: {keyphrases}")
            return keyphrases.get("keyphrases", [])

        except Exception as e:
            logger.error(f"Error in extracting keyphrases: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        cache_lock_file = os.path.join(cache_dir, "cache.lock")

        # Ensure the cache is handled atomically with a lock
        with open(cache_lock_file, 'w') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)  # Acquire exclusive lock
            try:
                logger.info(f"Adapting keyphrase extraction to {language}")

                self.extractor_prompt = keyphrase_extraction_prompt.adapt(language, self.llm, cache_dir)
                self.extractor_prompt.save(cache_dir)

                logger.info(f"Successfully adapted keyphrase extraction to {language}")
            except Exception as e:
                logger.error(f"Error during adaptation: {str(e)}")
                raise
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)  # Release lock

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the extractor prompts to a path.
        """
        try:
            self.extractor_prompt.save(cache_dir)
            logger.info(f"Successfully saved extractor prompt to {cache_dir}")
        except Exception as e:
            logger.error(f"Error saving extractor prompt: {str(e)}")
            raise
