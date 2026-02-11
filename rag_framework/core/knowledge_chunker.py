"""
Knowledge Chunker for COHS VQA RAG Framework

This module implements chunked knowledge delivery to handle information overload.

For long text blocks (10-20 rules), the text is split into chunks and processed
in multiple rounds, with results aggregated at the end.

This addresses the "information overload" challenge mentioned in the paper.
"""

import logging
from typing import Callable, List, Tuple
from ..models import LLMClient
from ..prompts import SystemPrompts

logger = logging.getLogger(__name__)

class KnowledgeChunker:
    """
    Knowledge Chunker for handling long text blocks.

    Implements chunked knowledge delivery to prevent model overwhelm
    and reduce cognitive bias from complex scenes.

    This corresponds to "Chunked Knowledge Delivery" module of the paper's framework.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the knowledge chunker.

        Args:
            llm_client: LLM client for processing chunks
        """
        self.llm_client = llm_client
        self.system_prompts = SystemPrompts()

    def should_chunk(self, anchor: str, anchor_type: str) -> bool:
        """
        Determine if an anchor requires text chunking.

        Certain anchors have long safety guidelines that need to be split
        into chunks for effective processing.

        Args:
            anchor: The anchor name
            anchor_type: Type of anchor ("object" or "operation")

        Returns:
            True if chunking is required, False otherwise
        """
        return self.system_prompts.should_chunk(anchor, anchor_type)

    @staticmethod
    def split_text(text: str, delimiter: str = "|") -> List[str]:
        """
        Split text into chunks using a delimiter.

        Args:
            text: The text to split
            delimiter: Delimiter string (default: "|")

        Returns:
            List of text chunks
        """
        chunks = text.split(delimiter)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    @staticmethod
    def determine_num_descriptions(
            has_operation_query: bool,
            is_chunked: bool
    ) -> int:
        """
        Determine the number of hazard descriptions per output.

        Args:
            has_operation_query: Whether there is an operation query
            is_chunked: Whether processing is chunked

        Returns:
            Maximum number of hazard descriptions per output
        """
        if is_chunked:
            return 1  # Chunked: output 1 at a time
        else:
            # Non-chunked: 3 if no operation, 2 if operation exists
            return 3 if not has_operation_query else 2

    def process_with_chunking(
        self,
        knowledge_text: str,
        image_base64: str,
        user_question: str,
        system_prompt_template: Callable[[str, int], str],
    ) -> Tuple[str, int]:
        """
        Process knowledge text with chunking.

        Splits long text into chunks, processes each chunk separately,
        and aggregates results.

        Args:
            knowledge_text: The full knowledge text
            image_base64: Base64-encoded image
            user_question: User's question
            system_prompt_template: Function to generate system prompt (text, num) -> prompt

        Returns:
            Tuple of (aggregated_answer, hazard_count)
        """
        chunks = self.split_text(knowledge_text)
        results = []
        hazard_count = 0

        for chunk in chunks:
            # Determine number of descriptions for this chunk
            num_descriptions = 1  # Chunked processing always uses 1

            # Generate system prompt
            system_prompt = system_prompt_template(chunk, num_descriptions)

            # Process chunk
            response = self.llm_client.invoke_with_vision(
                image_base64=image_base64,
                system_prompt=system_prompt,
                user_prompt=user_question,
                max_tokens=400,
            )

            # Check if response indicates no hazards
            if response and response not in ["None.", "None"]:
                results.append(response)
                hazard_count += 1

        # Aggregate results
        aggregated_answer = "\n".join(results) if results else "None."
        return aggregated_answer, hazard_count

    def process_without_chunking(
        self,
        knowledge_text: str,
        image_base64: str,
        user_question: str,
        system_prompt_template: Callable[[str, int], str],
        has_operation_query: bool,
    ) -> Tuple[str, int]:
        """
        Process knowledge text without chunking.

        Processes the entire text in a single pass.

        Args:
            knowledge_text: The full knowledge text
            image_base64: Base64-encoded image
            user_question: User's question
            system_prompt_template: Function to generate system prompt (text, num) -> prompt
            has_operation_query: Whether there is an operation query

        Returns:
            Tuple of (answer, hazard_count)
        """
        # Determine number of descriptions
        num_descriptions = self.determine_num_descriptions(
            has_operation_query, is_chunked=False
        )

        # Generate system prompt
        system_prompt = system_prompt_template(knowledge_text, num_descriptions)

        # Process
        response = self.llm_client.invoke_with_vision(
            image_base64=image_base64,
            system_prompt=system_prompt,
            user_prompt=user_question,
            max_tokens=400,
        )

        # Check if response indicates no hazards
        if response and response in ["None.", "None"]:
            return "None.", 0
        else:
            return response, 1

    def process_knowledge(
        self,
        anchor: str,
        knowledge_text: str,
        anchor_type: str,
        image_base64: str,
        user_question: str,
        system_prompt_template: Callable[[str, int], str],
        has_operation_query: bool = False,
        enable_chunking: bool = True,
    ) -> Tuple[str, int]:
        """
        Process knowledge text with or without chunking.

        This is the main entry point that decides whether to use chunking
        based on the anchor type and configuration.

        Args:
            anchor: The anchor name
            knowledge_text: The knowledge text to process
            anchor_type: Type of anchor ("object" or "operation")
            image_base64: Base64-encoded image
            user_question: User's question
            system_prompt_template: Function to generate system prompt
            has_operation_query: Whether there is an operation query
            enable_chunking: Whether chunking is enabled

        Returns:
            Tuple of (answer, hazard_count)
        """
        needs_chunking = self.should_chunk(anchor, anchor_type)

        if enable_chunking and needs_chunking:
            logger.debug(f"Processing {anchor} with chunking")
            return self.process_with_chunking(
                knowledge_text=knowledge_text,
                image_base64=image_base64,
                user_question=user_question,
                system_prompt_template=system_prompt_template,
            )
        else:
            logger.debug(f"Processing {anchor} without chunking")
            return self.process_without_chunking(
                knowledge_text=knowledge_text,
                image_base64=image_base64,
                user_question=user_question,
                system_prompt_template=system_prompt_template,
                has_operation_query=has_operation_query,
            )
