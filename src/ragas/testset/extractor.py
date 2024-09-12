from __future__ import annotations
import logging
import typing as t
import os
import shutil
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
        prompt = self.extractor_prompt.format(text=node.page_content)
        results = await self.llm.generate(prompt=prompt, is_async=is_async)
        keyphrases = await json_loader.safe_load(
            results.generations[0][0].text.strip(), llm=self.llm, is_async=is_async
        )
        keyphrases = keyphrases if isinstance(keyphrases, dict) else {}
        logger.debug("topics: %s", keyphrases)
        return keyphrases.get("keyphrases", [])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the extractor to a different language.
        """
        try:
            logger.info(f"Adapting keyphrase extraction to {language}")
    
            # Ensure prompt is reset before adaptation
            self.extractor_prompt = keyphrase_extraction_prompt  # Reset or reinitialize
            
            # Optional: Clear the cache directory for this language
            if cache_dir:
                language_cache_dir = os.path.join(cache_dir, language)
                if os.path.exists(language_cache_dir):
                    logger.info(f"Clearing cache for {language}")
                    shutil.rmtree(language_cache_dir)
                os.makedirs(language_cache_dir, exist_ok=True)
    
            # Now adapt the prompt with the new language and LLM
            self.extractor_prompt = self.extractor_prompt.adapt(language, self.llm, cache_dir)
            self.extractor_prompt.save(cache_dir)
            
            logger.info(f"Successfully adapted keyphrase extraction to {language}")
        except Exception as e:
            logger.error(f"Error during keyphrase extraction adaptation: {str(e)}")
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
