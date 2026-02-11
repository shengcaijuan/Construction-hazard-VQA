"""
Visual Semantic Anchor Extractor for COHS VQA RAG Framework

This module implements extraction of visual semantic anchors from construction images.
Anchors are text representations of hazard-relevant elements that serve as the bridge
between images and knowledge retrieval.

Two types of anchors:
- Object Anchors: Related to objects' unsafe states (e.g., Scaffolding, Edge)
- Operation Anchors: Related to workers' unsafe behaviors (e.g., Welding operation)
"""

import json
import logging
from typing import List, Dict

from ..models.llm_client import LLMClient
from ..prompts.system_prompts import SystemPrompts
from ..utils.image_utils import encode_image

logger = logging.getLogger(__name__)


class VisualAnchorExtractor:
    """
    Visual Semantic Anchor Extractor.

    Extracts hazard-relevant objects and operations from construction images
    to serve as retrieval queries for the knowledge base.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the anchor extractor.

        Args:
            llm_client: LLM client for multimodal inference
        """
        self.llm_client = llm_client
        self.system_prompts = SystemPrompts()

    def extract_object_anchors(
        self, image_path: str, max_retries: int = 3
    ) -> List[str]:
        """
        Extract object anchors from a construction image.

        Object anchors are hazard-prone objects such as scaffolding, edges,
        electrical distribution boxes, etc.

        Args:
            image_path: Path to the image file
            max_retries: Maximum number of retries for LLM calls

        Returns:
            List of extracted object anchors (e.g., ["Scaffolding", "Edge"])
            Returns ["Error"] if all retries fail
        """
        # Encode image
        try:
            image_base64 = encode_image(image_path)
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ["Error"]

        # Get system prompt
        system_prompt = self.system_prompts.get_object_anchor_extraction_prompt()

        # Retry loop
        for attempt in range(max_retries):
            response = self.llm_client.invoke_with_vision(
                image_base64=image_base64,
                system_prompt=system_prompt,
                user_prompt="",
                max_tokens=300,
            )
            try:
                # Parse JSON response
                object_query = json.loads(response)
                return object_query

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Object anchor extraction - Attempt {attempt + 1}: "
                    f"JSON parsing error: {e}. Response: {response}"
                )
            except Exception as e:
                logger.warning(
                    f"Object anchor extraction - Attempt {attempt + 1}: Error: {e}"
                )

        logger.error(f"Object anchor extraction: All {max_retries} attempts failed")
        return ["Error"]

    def extract_operation_anchors(
        self, image_path: str, max_retries: int = 3
    ) -> List[str]:
        """
        Extract operation anchor from a construction image.

        Operation anchors are construction operations related to unsafe behaviors,
        such as welding, scaffolding work, etc.

        Args:
            image_path: Path to the image file
            max_retries: Maximum number of retries for LLM calls

        Returns:
            List with at most one operation anchor (e.g., ["Welding operation"])
            Returns empty list [] if no operation is detected
            Returns ["Error"] if all retries fail
        """
        # Encode image
        try:
            image_base64 = encode_image(image_path)
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ["Error"]

        # Get system prompt
        system_prompt = self.system_prompts.get_operation_anchor_extraction_prompt()

        operation_query_str = ""
        for attempt in range(max_retries):
            try:
                response = self.llm_client.invoke_with_vision(
                    image_base64=image_base64,
                    system_prompt=system_prompt,
                    user_prompt="",
                    max_tokens=500,
                )

                operation_query_str = response
                operation_query = json.loads(response)
                return operation_query

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Operation anchor extraction - Attempt {attempt + 1}: "
                    f"JSON parsing error: {e}. Response: {operation_query_str}"
                )
            except Exception as e:
                logger.warning(
                    f"Operation anchor extraction - Attempt {attempt + 1}: Error: {e}"
                )

        logger.error(f"Operation anchor extraction: All {max_retries} attempts failed")
        return ["Error"]

    def extract_all_anchors(
        self, image_path: str, max_retries: int = 3
    ) -> Dict[str, List[str]]:
        """
        Extract all types of anchors from a construction image.

        This is a convenience method that extracts both object and operation anchors.

        Args:
            image_path: Path to the image file
            max_retries: Maximum number of retries for LLM calls

        Returns:
            Dictionary with extracted anchors:
            {
                "object_anchors": ["Scaffolding", "Edge"],
                "operation_anchors": ["Welding operation"] or []
            }
        """
        object_anchors = self.extract_object_anchors(image_path, max_retries)
        operation_anchors = self.extract_operation_anchors(image_path, max_retries)

        return {
            "object_anchors": object_anchors,
            "operation_anchors": operation_anchors,
        }

    @staticmethod
    def validate_anchors(anchors: Dict[str, List[str]]) -> bool:
        """
        Validate that extracted anchors are properly formatted.

        Args:
            anchors: Dictionary with 'object_anchors' and 'operation_anchors' keys

        Returns:
            True if anchors are valid, False otherwise
        """
        if not isinstance(anchors, dict):
            return False

        object_anchors = anchors.get("object_anchors", [])
        operation_anchors = anchors.get("operation_anchors", [])

        if not isinstance(object_anchors, list) or not isinstance(
            operation_anchors, list
        ):
            return False

        # Check for error state
        if "Error" in object_anchors or "Error" in operation_anchors:
            return False

        # Validate operation anchors (at most one)
        if len(operation_anchors) > 1:
            return False

        return True

    @staticmethod
    def has_operation(anchors: Dict[str, List[str]]) -> bool:
        """
        Check if any operation anchors were extracted.

        Args:
            anchors: Dictionary with 'operation_anchors' key

        Returns:
            True if operation anchors exist and are not empty
        """
        operation_anchors = anchors.get("operation_anchors", [])
        return bool(operation_anchors) and "Error" not in operation_anchors

    @staticmethod
    def has_objects(anchors: Dict[str, List[str]]) -> bool:
        """
        Check if any object anchors were extracted.

        Args:
            anchors: Dictionary with 'object_anchors' key

        Returns:
            True if object anchors exist and are not empty
        """
        object_anchors = anchors.get("object_anchors", [])
        return bool(object_anchors) and "Error" not in object_anchors
