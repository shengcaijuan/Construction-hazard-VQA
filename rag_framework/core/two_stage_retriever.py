"""
Two-Stage Retriever with Agent Decision-Making for COHS VQA RAG Framework

This module implements the dual-stage retrieval mechanism that uses an intelligent agent
to filter retrieved knowledge based on the user's question.

The two stages:
1. First Stage: Retrieve image-relevant text using visual semantic anchors
2. Second Stage: Agent filters for question-relevant content

This corresponds to "Two-Stage Retrieval with Agent Decision" module of the paper's framework:
"""

import json
import logging
from typing import Dict, List, Optional
from ..models import LLMClient
from .knowledge_base import COHSKnowledgeBase
from .temp_vector_db_builder import TempVectorDBBuilder
from ..prompts import SystemPrompts

logger = logging.getLogger(__name__)


class TwoStageRetriever:
    """
    Two-Stage Retriever with Agent Decision-Making.

    Implements intelligent knowledge retrieval with two stages:
    1. Anchor-based retrieval from knowledge base
    2. Agent-based filtering for question relevance
    """

    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_base: COHSKnowledgeBase,
        use_temp_db: bool = True,
        embeddings_model: str = "text-embedding-3-large",
    ):
        """
        Initialize the two-stage retriever.

        Args:
            llm_client: LLM client for agent decision-making
            knowledge_base: COHS knowledge base for retrieval
            use_temp_db: Whether to use temporary vector database (default: True)
            embeddings_model: Embedding model for temp DB (must match KB)
        """
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base
        self.use_temp_db = use_temp_db
        self.embeddings_model = embeddings_model
        self.system_prompts = SystemPrompts()

        # Initialize temporary database builder if enabled
        self.temp_db_builder: Optional[TempVectorDBBuilder] = None
        if use_temp_db:
            self.temp_db_builder = TempVectorDBBuilder(
                knowledge_base=knowledge_base,
                embeddings_model=embeddings_model,
            )
            logger.info("TwoStageRetriever initialized with temporary database support")
        else:
            logger.info("TwoStageRetriever initialized without temporary database")

    def first_stage_retrieve(
        self, object_queries: List[str], operation_queries: List[str]
    ) -> Dict[str, str]:
        """
        First Stage: Retrieve image-relevant text using visual semantic anchors.

        This stage uses the extracted anchors to retrieve safety guidelines
        from the knowledge base. The results are relevant to the image content.

        Args:
            object_queries: List of object anchors
            operation_queries: List of operation anchors

        Returns:
            First-stage retrieval results:
            {
                "Scaffolding": "Safety guidelines for scaffolding...",
                "Welding operation": "Safety guidelines for welding..."
            }
        """
        return self.knowledge_base.retrieve_all_knowledge(
            object_queries, operation_queries
        )

    def _first_stage_retrieve_from_temp(
        self,
        object_queries: List[str],
        operation_queries: List[str],
        image_path: str,
    ) -> Dict[str, str]:
        """
        First Stage: Retrieve from unified temporary vector database.

        This method retrieves from the cached unified temporary database instead of
        the main knowledge base. Used when temp DB is enabled and cached.

        The unified temp DB contains both object and operation chunks re-embedded into a single knowledge base.

        Args:
            object_queries: List of object anchors
            operation_queries: List of operation anchors
            image_path: Image file path (cache key)

        Returns:
            First-stage retrieval results from unified temp DB:
            {
                "Scaffolding": "Safety guidelines for scaffolding...",
                "Welding operation": "Safety guidelines for welding..."
            }
        """
        if self.temp_db_builder is None:
            logger.warning("Temp DB builder not initialized, falling back to main KB")
            return self.first_stage_retrieve(object_queries, operation_queries)

        results = {}

        # All queries (both object and operation) are retrieved from the unified temp DB
        all_queries = object_queries + operation_queries

        for query in all_queries:
            docs = self.temp_db_builder.retrieve_from_temp(
                query=query,
                image_path=image_path,
                top_k=1,
            )
            if docs:
                results[query] = str(docs[0].page_content)

        return results

    def second_stage_filter(
        self, user_question: str, first_stage_results: Dict[str, str]
    ) -> List[str]:
        """
        Second Stage: Agent filters for question-relevant content.

        This stage uses an intelligent agent to analyze the user's question
        and select only the relevant knowledge entries from the first-stage results.

        Key logic:
        - Questions about "unsafe behaviors" -> only select operation-related text
        - Questions about "general hazards" -> select all text

        Args:
            user_question: The user's question
            first_stage_results: Results from first stage {anchor: knowledge_text}

        Returns:
            List of selected anchor names (e.g., ["Scaffolding", "Welding operation"])
        """
        if not first_stage_results:
            return []

        # Build agent prompt
        system_prompt = self.system_prompts.get_two_stage_agent_prompt(
            user_question, first_stage_results
        )

        try:
            response = self.llm_client.invoke(
                messages=[{"role": "system", "content": system_prompt}],
                max_tokens=300,
            )

            # Parse response
            response = response.strip()

            if response.lower() == "none":
                return []

            selected_anchors = json.loads(response)
            return selected_anchors

        except json.JSONDecodeError as e:
            logger.error(f"Agent response JSON parsing failed: {e}")
            # Fallback: return all keys
            return list(first_stage_results.keys())
        except Exception as e:
            logger.error(f"Second stage filtering failed: {e}")
            # Fallback: return all keys
            return list(first_stage_results.keys())

    def retrieve(
        self,
        user_question: str,
        object_queries: List[str],
        operation_queries: List[str],
        image_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Execute complete two-stage retrieval.

        Combines both stages:
        1. Retrieve image-relevant knowledge using anchors
        2. Filter for question-relevant knowledge using agent

        If temporary database is enabled and image_path is provided:
        - First call for an image: builds temp DB from main KB
        - Subsequent calls: uses cached temp DB (faster)

        Args:
            user_question: The user's question
            object_queries: List of object anchors
            operation_queries: List of operation anchors
            image_path: Image file path (for temp DB caching)

        Returns:
            Final filtered retrieval results:
            {
                "Scaffolding": "Safety guidelines for scaffolding...",
                "Welding operation": "Safety guidelines for welding..."
            }
        """
        # First stage: retrieve using anchors
        if (
            self.use_temp_db
            and image_path
            and self.temp_db_builder is not None
            and self.temp_db_builder.has_cache(image_path)
        ):
            # Use cached temporary database
            logger.debug(f"Using cached temp DB for {image_path}")
            first_stage = self._first_stage_retrieve_from_temp(
                object_queries, operation_queries, image_path
            )
        else:
            # Build temp DB if enabled, or use main KB directly
            if self.use_temp_db and image_path and self.temp_db_builder is not None:
                logger.debug(f"Building temp DB for {image_path}")
                # Build and cache temp DB, then retrieve from it
                self.temp_db_builder.get_or_build(
                    image_path=image_path,
                    object_queries=object_queries,
                    operation_queries=operation_queries,
                )
                first_stage = self._first_stage_retrieve_from_temp(
                    object_queries, operation_queries, image_path
                )
            else:
                # Direct retrieval from main KB
                first_stage = self.first_stage_retrieve(
                    object_queries, operation_queries
                )

        if not first_stage:
            logger.warning("First stage returned no results")
            return {}

        # Second stage: agent filtering
        selected_keys = self.second_stage_filter(user_question, first_stage)

        # Filter results
        final_results = {k: v for k, v in first_stage.items() if k in selected_keys}

        logger.debug(
            f"Two-stage retrieval: {len(first_stage)} -> {len(final_results)} entries"
        )

        return final_results

    def retrieve_single_stage(
        self, object_queries: List[str], operation_queries: List[str]
    ) -> Dict[str, str]:
        """
        Retrieve using only single stage (no agent filtering).

        This is useful for ablation studies and comparison.

        Args:
            object_queries: List of object anchors
            operation_queries: List of operation anchors

        Returns:
            Retrieval results without filtering
        """
        return self.first_stage_retrieve(object_queries, operation_queries)
