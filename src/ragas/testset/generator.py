from __future__ import annotations

import os
import logging
import typing as t
from dataclasses import dataclass
from random import choices, sample
import pandas as pd
import shutil

from datasets import Dataset

from ragas._analytics import TestsetGenerationEvent, track, track_was_completed
from ragas.embeddings.base import (
    BaseRagasEmbeddings,
    LangchainEmbeddingsWrapper,
    LlamaIndexEmbeddingsWrapper,
)
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper, LlamaIndexLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset.docstore import Document, DocumentStore, InMemoryDocumentStore
from ragas.testset.evolutions import (
    ComplexEvolution,
    CurrentNodes,
    DataRow,
    Evolution,
    multi_context,
    reasoning,
    simple,
)
from ragas.testset.extractor import KeyphraseExtractor
from ragas.testset.filters import EvolutionFilter, NodeFilter, QuestionFilter
from ragas.utils import check_if_sum_is_close, deprecated, get_feature_language, is_nan

if t.TYPE_CHECKING:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.embeddings import Embeddings as LangchainEmbeddings
    from langchain_core.language_models import BaseLanguageModel as LangchainLLM
    from llama_index.core.base.embeddings.base import (
        BaseEmbedding as LlamaIndexEmbeddings,
    )
    from llama_index.core.base.llms.base import BaseLLM as LlamaindexLLM
    from llama_index.core.schema import Document as LlamaindexDocument

logger = logging.getLogger(__name__)

Distributions = t.Dict[t.Any, float]
DEFAULT_DISTRIBUTION = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}


class AdaptationError(Exception):
    """Custom exception for adaptation errors."""
    pass


@dataclass
class TestDataset:
    """
    TestDataset class
    """

    test_data: t.List[DataRow]

    def _to_records(self) -> t.List[t.Dict]:
        data_samples = []
        for data in self.test_data:
            data_dict = dict(data)
            data_dict["episode_done"] = True
            data_samples.append(data_dict)
        return data_samples

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._to_records())

    def to_dataset(self) -> Dataset:
        return Dataset.from_list(self._to_records())


@dataclass
class TestsetGenerator:
    generator_llm: BaseRagasLLM
    critic_llm: BaseRagasLLM
    embeddings: BaseRagasEmbeddings
    docstore: DocumentStore

    @classmethod
    def from_langchain(
        cls,
        generator_llm: LangchainLLM,
        critic_llm: LangchainLLM,
        embeddings: LangchainEmbeddings,
        docstore: t.Optional[DocumentStore] = None,
        run_config: RunConfig = RunConfig(),
        chunk_size: int = 1024,
    ) -> "TestsetGenerator":
        generator_llm_model = LangchainLLMWrapper(generator_llm)
        critic_llm_model = LangchainLLMWrapper(critic_llm)
        embeddings_model = LangchainEmbeddingsWrapper(embeddings)

        keyphrase_extractor = KeyphraseExtractor(llm=generator_llm_model)
        if docstore is None:
            from langchain.text_splitter import TokenTextSplitter

            splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
            docstore = InMemoryDocumentStore(
                splitter=splitter,
                embeddings=embeddings_model,
                extractor=keyphrase_extractor,
                run_config=run_config,
            )
        return cls(
            generator_llm=generator_llm_model,
            critic_llm=critic_llm_model,
            embeddings=embeddings_model,
            docstore=docstore,
        )

    @classmethod
    def from_llama_index(
        cls,
        generator_llm: LlamaindexLLM,
        critic_llm: LlamaindexLLM,
        embeddings: LlamaIndexEmbeddings,
        docstore: t.Optional[DocumentStore] = None,
        run_config: RunConfig = RunConfig(),
    ) -> "TestsetGenerator":
        generator_llm_model = LlamaIndexLLMWrapper(generator_llm)
        critic_llm_model = LlamaIndexLLMWrapper(critic_llm)
        embeddings_model = LlamaIndexEmbeddingsWrapper(embeddings)
        keyphrase_extractor = KeyphraseExtractor(llm=generator_llm_model)
        if docstore is None:
            from langchain.text_splitter import TokenTextSplitter

            splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=0)
            docstore = InMemoryDocumentStore(
                splitter=splitter,
                embeddings=embeddings_model,
                extractor=keyphrase_extractor,
                run_config=run_config,
            )
        return cls(
            generator_llm=generator_llm_model,
            critic_llm=critic_llm_model,
            embeddings=embeddings_model,
            docstore=docstore,
        )

    @classmethod
    @deprecated("0.1.4", removal="0.2.0", alternative="from_langchain")
    def with_openai(
        cls,
        generator_llm: str = "gpt-4o-mini",
        critic_llm: str = "gpt-4o",
        embeddings: str = "text-embedding-ada-002",
        docstore: t.Optional[DocumentStore] = None,
        chunk_size: int = 1024,
    ) -> "TestsetGenerator":
        generator_llm_model = ChatOpenAI(model=generator_llm)
        critic_llm_model = ChatOpenAI(model=critic_llm)
        embeddings_model = OpenAIEmbeddings(model=embeddings)

        return cls.from_langchain(
            generator_llm=generator_llm_model,
            critic_llm=critic_llm_model,
            embeddings=embeddings_model,
            docstore=docstore,
            chunk_size=chunk_size,
        )

    def generate_with_llamaindex_docs(
        self,
        documents: t.Sequence[LlamaindexDocument],
        test_size: int,
        distributions: t.Optional[Distributions] = None,
        with_debugging_logs=False,
        is_async: bool = True,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ):
        distributions = distributions or {}
        # chunk documents and add to docstore
        self.docstore.add_documents(
            [Document.from_llamaindex_document(doc) for doc in documents]
        )

        return self.generate(
            test_size=test_size,
            distributions=distributions,
            with_debugging_logs=with_debugging_logs,
            is_async=is_async,
            run_config=run_config,
            raise_exceptions=raise_exceptions,
        )

    # if you add any arguments to this function, make sure to add them to
    # generate_with_langchain_docs as well
    def generate_with_langchain_docs(
        self,
        documents: t.Sequence[LCDocument],
        test_size: int,
        distributions: t.Optional[Distributions] = None,
        with_debugging_logs=False,
        is_async: bool = True,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ):
        distributions = distributions or {}
        # chunk documents and add to docstore
        self.docstore.add_documents(
            [Document.from_langchain_document(doc) for doc in documents]
        )

        return self.generate(
            test_size=test_size,
            distributions=distributions,
            with_debugging_logs=with_debugging_logs,
            is_async=is_async,
            raise_exceptions=raise_exceptions,
            run_config=run_config,
        )

    def init_evolution(self, evolution: Evolution) -> None:
        evolution.docstore = self.docstore

        if evolution.generator_llm is None:
            evolution.generator_llm = self.generator_llm

            if evolution.question_filter is None:
                evolution.question_filter = QuestionFilter(llm=self.critic_llm)
            if evolution.node_filter is None:
                evolution.node_filter = NodeFilter(llm=self.critic_llm)

            if isinstance(evolution, ComplexEvolution):
                if evolution.evolution_filter is None:
                    evolution.evolution_filter = EvolutionFilter(llm=self.critic_llm)

    @track_was_completed
    def generate(
        self,
        test_size: int,
        distributions: t.Optional[Distributions] = None,
        with_debugging_logs=False,
        is_async: bool = True,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ) -> TestDataset:
        distributions = distributions or DEFAULT_DISTRIBUTION
        # validate distributions
        if not check_if_sum_is_close(list(distributions.values()), 1.0, 3):
            raise ValueError(
                f"distributions passed do not sum to 1.0 [got {sum(list(distributions.values()))}]. Please check the "
                f"distributions."
            )

        # configure run_config for docstore
        if run_config is None:
            run_config = RunConfig(max_retries=15, max_wait=90)
        self.docstore.set_run_config(run_config)

        # init filters and evolutions
        for evolution, probability in distributions.items():
            self.init_evolution(evolution)
            evolution.init(is_async=is_async, run_config=run_config)

        if with_debugging_logs:
            from ragas.utils import patch_logger

            patch_logger("ragas.testset.evolutions", logging.DEBUG)
            patch_logger("ragas.testset.extractor", logging.DEBUG)
            patch_logger("ragas.testset.filters", logging.DEBUG)
            patch_logger("ragas.testset.docstore", logging.DEBUG)
            patch_logger("ragas.llms.prompt", logging.DEBUG)

        exec = Executor(
            desc="Generating",
            keep_progress_bar=True,
            raise_exceptions=raise_exceptions,
            run_config=run_config,
        )

        current_nodes = [
            CurrentNodes(root_node=n, nodes=[n])
            for n in self.docstore.get_random_nodes(k=test_size)
        ]
        total_evolutions = 0
        for evolution, probability in distributions.items():
            for i in sample(range(test_size), round(probability * test_size)):
                exec.submit(
                    evolution.evolve,
                    current_nodes[i],
                    name=f"{evolution.__class__.__name__}-{i}",
                )
                total_evolutions += 1
        if total_evolutions <= test_size:
            filler_evolutions = choices(
                list(distributions), k=test_size - total_evolutions
            )
            for evolution in filler_evolutions:
                exec.submit(
                    evolution.evolve,
                    current_nodes[total_evolutions],
                    name=f"{evolution.__class__.__name__}-{total_evolutions}",
                )
                total_evolutions += 1

        try:
            test_data_rows = exec.results()
            if not test_data_rows:
                raise ExceptionInRunner()

        except ValueError as e:
            raise e
        # make sure to ignore any NaNs that might have been returned
        # due to failed evolutions. MaxRetriesExceeded is a common reason
        test_data_rows = [r for r in test_data_rows if not is_nan(r)]
        test_dataset = TestDataset(test_data=test_data_rows)
        evol_lang = [get_feature_language(e) for e in distributions]
        evol_lang = [e for e in evol_lang if e is not None]
        track(
            TestsetGenerationEvent(
                event_type="testset_generation",
                evolution_names=[e.__class__.__name__.lower() for e in distributions],
                evolution_percentages=[distributions[e] for e in distributions],
                num_rows=len(test_dataset.test_data),
                language=evol_lang[0] if len(evol_lang) > 0 else "",
                is_experiment=False,
            )
        )

        return test_dataset

    def adapt(self, language: str, evolutions: t.List[Evolution], cache_dir: t.Optional[str] = None) -> None:
        for attempt in range(3):
            try:
                assert isinstance(self.docstore, InMemoryDocumentStore), "Must be an instance of in-memory DocumentStore"
                assert self.docstore.extractor is not None, "Extractor is not set"

                logger.info(f"Adapting TestsetGenerator to language: {language}")
                logger.info(f"Cache directory: {cache_dir}")

                if not cache_dir:
                    logger.warning("No cache directory provided. Adaptation will not be cached.")
                    self._perform_adaptation(language, evolutions)
                    return

                language_cache_dir = os.path.join(cache_dir, language)

                # Initial cache validation
                if not self.is_cache_valid(cache_dir, language):
                    logger.info(f"Invalid or incomplete cache for {language}. Deleting cache directory.")
                    if os.path.exists(language_cache_dir):
                        shutil.rmtree(language_cache_dir)
                    # Proceed with adaptation without re-checking cache
                    self._perform_adaptation(language, evolutions, cache_dir)
                else:
                    logger.info(f"Valid cache exists for {language}, skipping adaptation.")
                    return

                # Post-adaptation cache validation
                if not self.is_cache_valid(cache_dir, language):
                    raise AdaptationError("Cache validation failed after adaptation")
                else:
                    # Cache is valid after adaptation
                    return
            except AdaptationError as e:
                logger.error(f"Adaptation attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                else:
                    logger.info("Retrying adaptation...")
                    continue

    def _perform_adaptation(self, language: str, evolutions: t.List[Evolution], cache_dir: t.Optional[str] = None) -> None:
        logger.info("Adapting docstore extractor")
        self.docstore.extractor.adapt(language=language, cache_dir=cache_dir)
        self.docstore.extractor.save(cache_dir=cache_dir)
        logger.info(f"Extractor adapted and saved for {language}")

        for evolution in evolutions:
            self.init_evolution(evolution)
            evolution.init()
            evolution.adapt(language=language, cache_dir=cache_dir)
            evolution.save(cache_dir=cache_dir)

    def is_cache_valid(self, cache_dir: str, language: str) -> bool:
        """Check if the cache is valid by ensuring all necessary files exist."""
        missing_files = self.get_missing_files(cache_dir, language)
        if missing_files:
            logger.info(f"Cache is incomplete, missing: {', '.join(missing_files)}")
            return False
        return True

    def get_missing_files(self, cache_dir: str, language: str) -> t.List[str]:
        """Get a list of missing required files in the cache."""
        language_cache_dir = os.path.join(cache_dir, language)
        required_files = [
            'answer_formulate.json',
            'compress_question.json',
            'conditional_question.json',
            'evolution_elimination.json',
            'filter_question.json',
            'find_relevant_context.json',
            'keyphrase_extraction.json',
            'multi_context_question.json',
            'reasoning_question.json',
            'rewrite_question.json',
            'score_context.json',
            'seed_question.json'
        ]
        return [file for file in required_files if not os.path.exists(os.path.join(language_cache_dir, file))]
