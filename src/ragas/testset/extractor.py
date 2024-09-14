from __future__ import annotations
import logging
import typing as t
import os
import shutil
import tempfile
from filelock import FileLock # type: ignore
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from tenacity import retry, stop_after_attempt, wait_exponential # type: ignore

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
        pass

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the extractor to a different language.
        """
        raise NotImplementedError(f"adapt() is not implemented for {self.__class__.__name__} Extractor")

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the extractor prompts to a path.
        """
        raise NotImplementedError(f"save() is not implemented for {self.__class__.__name__} Extractor")

@dataclass
class KeyphraseExtractor(Extractor):
    extractor_prompt: Prompt = field(
        default_factory=lambda: keyphrase_extraction_prompt
    )

    async def extract(self, node: Node, is_async: bool = True) -> t.List[str]:
        try:
            prompt = self.extractor_prompt.format(text=node.page_content)
            results = await self.llm.generate(prompt=prompt, is_async=is_async)

            # Check if the LLM returned valid results
            if not results or not results.generations or not results.generations[0]:
                logger.error("LLM returned an empty or invalid response")
                return []

            generated_text = results.generations[0][0].text.strip()

            # Ensure non-empty response
            if not generated_text:
                logger.error("LLM returned an empty response")
                return []

            # Try to load the JSON safely
            keyphrases = await json_loader.safe_load(generated_text, llm=self.llm, is_async=is_async)

            # Ensure the returned keyphrases are a valid dict
            if not isinstance(keyphrases, dict):
                logger.error(f"Invalid keyphrases format received: {keyphrases}")
                return []

            return keyphrases.get("keyphrases", [])

        except Exception as e:
            logger.error(f"Error in extracting keyphrases: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()

        cache_lock_file = os.path.join(cache_dir, "cache.lock")

        # Use FileLock for platform-independent file locking
        with FileLock(cache_lock_file):
            try:
                logger.info(f"Adapting keyphrase extraction to {language}")

                self.extractor_prompt = keyphrase_extraction_prompt.adapt(language, self.llm, cache_dir)
                self.extractor_prompt.save(cache_dir)

                logger.info(f"Successfully adapted keyphrase extraction to {language}")
            except Exception as e:
                logger.error(f"Error during adaptation: {str(e)}")
                raise

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
