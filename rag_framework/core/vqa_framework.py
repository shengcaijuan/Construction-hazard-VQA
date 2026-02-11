"""
COHS VQA Framework - Main Framework Class

This module integrates all four core modules into a unified framework for
construction occupational health and safety visual question answering.

Framework Architecture:
1. External COHS Knowledge Base - Vector databases for safety guidelines
2. Visual Semantic Anchor Extraction - Extract hazard-relevant elements
3. Two-Stage Retrieval - Agent-based intelligent filtering
4. Chunked Knowledge Delivery - Handle long text blocks

This is the main entry point for using the RAG framework.
"""

import logging
from typing import Any, Dict, List, Optional
from ..models import LLMClient
from .visual_anchor_extractor import VisualAnchorExtractor
from .knowledge_base import COHSKnowledgeBase
from .two_stage_retriever import TwoStageRetriever
from .knowledge_chunker import KnowledgeChunker
from ..prompts import SystemPrompts
from ..utils import encode_image, renumber_sentences

logger = logging.getLogger(__name__)

class COHSVQAFramework:
    """
    COHS VQA Framework - Main Framework Class.

    Integrates all four core modules to provide a complete RAG-based solution
    for construction safety hazard VQA.
    """

    def __init__(
        self,
        object_db_path: str,
        operation_db_path: str,
        llm_model: str = "gpt-4o",
        temperature: float = 0.0,
        use_temp_db: bool = True,
        embeddings_model: str = "text-embedding-3-large",
    ):
        """
        Initialize the COHS VQA Framework.

        Args:
            object_db_path: Path to FAISS index for object safety guidelines
            operation_db_path: Path to FAISS index for operation safety guidelines
            llm_model: LLM model to use (e.g., "gpt-4o", "qwen-vl-max")
            temperature: Temperature parameter for LLM generation
            use_temp_db: Whether to use temporary vector database (default: True)
            embeddings_model: Embedding model for temp DB (must match KB)
        """
        # Initialize LLM client
        self.llm_client = LLMClient(model=llm_model, temperature=temperature)
        self.model_name = llm_model

        # Initialize system prompts
        self.system_prompts = SystemPrompts()

        # Initialize core modules
        self.anchor_extractor = VisualAnchorExtractor(self.llm_client)
        self.knowledge_base = COHSKnowledgeBase(
            object_db_path=object_db_path,
            operation_db_path=operation_db_path,
            embedding_model=embeddings_model,
        )
        self.two_stage_retriever = TwoStageRetriever(
            llm_client=self.llm_client,
            knowledge_base=self.knowledge_base,
            use_temp_db=use_temp_db,
            embeddings_model=embeddings_model,
        )
        self.knowledge_chunker = KnowledgeChunker(self.llm_client)

        # Store settings
        self.use_temp_db = use_temp_db

        logger.info(
            f"COHSVQAFramework initialized with model: {llm_model}, "
            f"temp_db: {use_temp_db}"
        )

    def answer_question(
        self,
        image_path: str,
        user_question: str,
        enable_two_stage_retrieval: bool = True,
        enable_chunking: bool = True,
    ) -> Dict[str, Any]:
        """
        Answer a user question using the complete RAG framework.

        This is the main entry point that implements the full pipeline:
        1. Extract visual semantic anchors from the image
        2. Retrieve relevant knowledge (single or two-stage)
        3. Process knowledge with optional chunking
        4. Generate comprehensive answer

        Args:
            image_path: Path to the construction image
            user_question: The user's question about safety hazards
            enable_two_stage_retrieval: Whether to use two-stage retrieval (default: True)
            enable_chunking: Whether to use knowledge chunking (default: True)

        Returns:
            Dictionary with:
            {
                "answer": "The generated answer text",
                "extracted_anchors": {
                    "objects": ["Scaffolding"],
                    "operations": ["Welding operation"]
                },
                "retrieved_knowledge": {
                    "Scaffolding": "Safety guidelines..."
                }
            }
        """
        # Encode image
        try:
            image_base64 = encode_image(image_path)
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return {
                "answer": f"Error: Unable to process image - {e}",
                "error": str(e),
            }

        # Step 1: Extract visual semantic anchors
        logger.info(f"Extracting anchors from {image_path}")
        anchors = self.anchor_extractor.extract_all_anchors(image_path)
        object_anchors = anchors["object_anchors"]
        operation_anchors = anchors["operation_anchors"]

        # Check for extraction errors
        if "Error" in object_anchors or "Error" in operation_anchors:
            logger.error("Anchor extraction failed")
            return {
                "answer": "Error: Failed to extract visual anchors from image",
                "extracted_anchors": anchors,
            }

        # Step 2: Retrieve relevant knowledge
        if enable_two_stage_retrieval:
            logger.info("Using two-stage retrieval with agent filtering")
            knowledge = self.two_stage_retriever.retrieve(
                user_question=user_question,
                object_queries=object_anchors,
                operation_queries=operation_anchors,
                image_path=image_path if self.use_temp_db else None,
            )
        else:
            logger.info("Using single-stage retrieval")
            knowledge = self.two_stage_retriever.retrieve_single_stage(
                object_queries=object_anchors, operation_queries=operation_anchors
            )

        if not knowledge:
            logger.warning("No knowledge retrieved")
            return {
                "answer": "No relevant safety guidelines found for the detected elements.",
                "extracted_anchors": anchors,
                "retrieved_knowledge": {},
            }

        # Step 3: Generate answer using retrieved knowledge
        logger.info(f"Generating answer with {len(knowledge)} knowledge entries")
        answer = self._answer_with_knowledge(
            image_base64=image_base64,
            user_question=user_question,
            knowledge=knowledge,
            object_anchors=object_anchors,
            operation_anchors=operation_anchors,
            enable_chunking=enable_chunking,
        )

        return {
            "answer": answer,
            "extracted_anchors": {
                "objects": object_anchors,
                "operations": operation_anchors,
            },
            "retrieved_knowledge": knowledge,
        }

    def _answer_with_knowledge(
        self,
        image_base64: str,
        user_question: str,
        knowledge: Dict[str, str],
        object_anchors: List[str],
        operation_anchors: List[str],
        enable_chunking: bool,
    ) -> str:
        """
        Generate answer using retrieved knowledge.

        Internal method that orchestrates the detection of different hazard types.

        Args:
            image_base64: Base64-encoded image
            user_question: User's question
            knowledge: Retrieved knowledge {anchor: text}
            object_anchors: Extracted object anchors
            operation_anchors: Extracted operation anchors
            enable_chunking: Whether to use knowledge chunking

        Returns:
            Generated answer text
        """
        # Determine question type for appropriate prompt selection
        question_class = self._classify_question(user_question)

        # Collect all hazard descriptions
        hazard_descriptions = []

        # Check if there's an operation query
        has_operation = bool(operation_anchors)

        # Process object-related knowledge
        need_object_detection = question_class in [
            "cohs_hazards",
            "unsafe_objects",
        ]
        if object_anchors and need_object_detection:
            object_hazards = self._detect_object_hazards(
                image_base64=image_base64,
                user_question=user_question,
                knowledge=knowledge,
                object_anchors=object_anchors,
                has_operation_query=has_operation,
                enable_chunking=enable_chunking,
            )
            if object_hazards:
                hazard_descriptions.append(object_hazards)

        # Process operation-related knowledge
        need_operation_detection = question_class != "unsafe_objects"
        if operation_anchors and need_operation_detection:
            operation_hazards = self._detect_operation_hazards(
                image_base64=image_base64,
                user_question=user_question,
                knowledge=knowledge,
                operation_anchors=operation_anchors,
                question_class=question_class,
                enable_chunking=enable_chunking,
            )
            if operation_hazards:
                hazard_descriptions.append(operation_hazards)

            # Always include general hazards when operation is present
            general_hazards = self._detect_general_hazards(
                image_base64=image_base64, user_question=user_question
            )
            if general_hazards:
                hazard_descriptions.append(general_hazards)

        # Combine and renumber
        if hazard_descriptions:
            combined = "\n".join(hazard_descriptions)
            return renumber_sentences(combined)
        else:
            return "Construction site is safe."

    def _detect_object_hazards(
        self,
        image_base64: str,
        user_question: str,
        knowledge: Dict[str, str],
        object_anchors: List[str],
        has_operation_query: bool,
        enable_chunking: bool,
    ) -> str:
        """Detect object-related hazards using retrieved knowledge."""
        hazard_list = []

        for anchor in object_anchors:
            if anchor not in knowledge:
                continue

            knowledge_text = knowledge[anchor]

            # Process with chunker
            answer, _ = self.knowledge_chunker.process_knowledge(
                anchor=anchor,
                knowledge_text=knowledge_text,
                anchor_type="object",
                image_base64=image_base64,
                user_question=user_question,
                system_prompt_template=lambda text, num: self.system_prompts.get_object_hazard_detection_prompt(
                    text, num
                ),
                has_operation_query=has_operation_query,
                enable_chunking=enable_chunking,
            )

            if answer and answer not in ["None.", "None"]:
                hazard_list.append(answer)

        return "\n".join(hazard_list)

    def _detect_operation_hazards(
        self,
        image_base64: str,
        user_question: str,
        knowledge: Dict[str, str],
        operation_anchors: List[str],
        question_class: str,
        enable_chunking: bool,
    ) -> str:
        """Detect operation-related hazards using retrieved knowledge."""
        if not operation_anchors:
            return ""

        anchor = operation_anchors[0]  # At most one operation
        if anchor not in knowledge:
            return ""

        knowledge_text = knowledge[anchor]

        # Process with chunker
        answer, _ = self.knowledge_chunker.process_knowledge(
            anchor=anchor,
            knowledge_text=knowledge_text,
            anchor_type="operation",
            image_base64=image_base64,
            user_question=user_question,
            system_prompt_template=lambda text, num: self.system_prompts.get_operation_hazard_detection_prompt(
                text, question_class
            ),
            has_operation_query=True,
            enable_chunking=enable_chunking,
        )

        return answer if answer and answer not in ["None.", "None"] else ""

    def _detect_general_hazards(
        self, image_base64: str, user_question: str
    ) -> str:
        """Detect general hazards (PPE, safety helmets, etc.)."""
        system_prompt = self.system_prompts.get_general_hazard_detection_prompt()

        response = self.llm_client.invoke_with_vision(
            image_base64=image_base64,
            system_prompt=system_prompt,
            user_prompt=user_question,
            max_tokens=400,
        )

        return response if response and response not in ["None.", "None"] else ""

    def answer_without_rag(self, image_path: str, user_question: str) -> str:
        """
        Answer a question without using RAG (baseline method).

        This is for comparison and ablation studies.

        Args:
            image_path: Path to the construction image
            user_question: The user's question

        Returns:
            Generated answer without knowledge enhancement
        """
        try:
            image_base64 = encode_image(image_path)
        except Exception as e:
            return f"Error: Unable to process image - {e}"

        system_prompt = self.system_prompts.get_baseline_detection_prompt()

        response = self.llm_client.invoke_with_vision(
            image_base64=image_base64,
            system_prompt=system_prompt,
            user_prompt=user_question,
            max_tokens=500,
        )

        return response if response else "No hazards detected."

    @staticmethod
    def _classify_question(question: str) -> str:
        """
        Classify the question type for appropriate prompt selection.

        Args:
            question: The user's question

        Returns:
            Question class: "cohs_hazards", "unsafe_behaviors", "ppe_lacking", or "unsafe_objects"
        """
        question_lower = question.lower()

        if "unsafe behavior" in question_lower:
            return "unsafe_behaviors"
        elif "ppe" in question_lower or "personal protective equipment" in question_lower:
            return "ppe_lacking"
        elif "unsafe object" in question_lower:
            return "unsafe_objects"
        else:
            return "cohs_hazards"

    def get_framework_info(self) -> Dict[str, Any]:
        """
        Get information about the framework configuration.

        Returns:
            Dictionary with framework information
        """
        info = {
            "model": self.model_name,
            "use_temp_db": self.use_temp_db,
            "knowledge_base_stats": self.knowledge_base.get_stats(),
            "system_prompts_available": True,
            "two_stage_retriever_available": True,
            "knowledge_chunker_available": True,
        }

        # Add temp DB cache info if enabled
        if self.use_temp_db and self.two_stage_retriever.temp_db_builder:
            info["temp_db_cache_size"] = (
                self.two_stage_retriever.temp_db_builder.get_cache_size()
            )

        return info

    def clear_temp_db_cache(self, image_path: Optional[str] = None) -> None:
        """
        Clear temporary database cache.

        Args:
            image_path: Specific image path to clear.
                       If None, clears all cached databases.
        """
        if self.use_temp_db and self.two_stage_retriever.temp_db_builder:
            self.two_stage_retriever.temp_db_builder.clear(image_path)
            logger.info(f"Cleared temp DB cache for: {image_path or 'all'}")
        else:
            logger.warning("Temp DB is not enabled, cannot clear cache")
