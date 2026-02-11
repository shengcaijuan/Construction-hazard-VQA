"""
COHS Knowledge Base Management for RAG Framework

This module manages the FAISS vector databases storing construction safety guidelines.

Two separate knowledge bases:
- Object Safety Guidelines: 87 rules for 15 hazard-prone objects
- Operation Safety Guidelines: 144 rules for 20 construction operations

This corresponds to Module 1 of the paper's framework:
"External COHS Knowledge Base"
"""

import os
import logging
from typing import Dict, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class COHSKnowledgeBase:
    """
    COHS Knowledge Base Manager.

    Manages FAISS vector databases for object and operation safety guidelines.
    Retrieves relevant safety rules based on visual semantic anchors.
    """

    def __init__(
        self,
        object_db_path: str,
        operation_db_path: str,
        embedding_model: str = "text-embedding-3-large",
    ):
        """
        Initialize the knowledge base.

        Args:
            object_db_path: Path to FAISS index for object safety guidelines
            operation_db_path: Path to FAISS index for operation safety guidelines
            embedding_model: Name of the embedding model (must match the one used for indexing)
        """
        self.object_db_path = object_db_path
        self.operation_db_path = operation_db_path
        self.embedding_model = embedding_model

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Load FAISS databases
        self.object_db = self._load_faiss(object_db_path)
        self.operation_db = self._load_faiss(operation_db_path)

        # Create retrievers
        self.object_retriever = (
            self.object_db.as_retriever() if self.object_db else None
        )
        self.operation_retriever = (
            self.operation_db.as_retriever() if self.operation_db else None
        )

    def _load_faiss(self, file_path: str) -> Optional[FAISS]:
        """
        Load FAISS index from file.

        Args:
            file_path: Path to the FAISS index directory

        Returns:
            FAISS vector store if successful, None otherwise
        """
        if os.path.exists(file_path):
            logger.info(f"Loading FAISS index from {file_path}")
            try:
                return FAISS.load_local(
                    file_path, self.embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {file_path}: {e}")
                return None
        else:
            logger.warning(f"FAISS index not found at {file_path}")
            return None

    def retrieve_object_knowledge(self, query: str) -> Optional[str]:
        """
        Retrieve object safety guidelines.

        Args:
            query: Retrieval query (typically an object anchor like "Scaffolding")

        Returns:
            Retrieved safety guidelines text, or None if retrieval fails
        """
        if self.object_retriever is None:
            logger.warning("Object retriever not available")
            return None

        try:
            results = self.object_retriever.invoke(query)
            if results and len(results) > 0:
                return str(results[0].page_content)
        except Exception as e:
            logger.error(f"Object knowledge retrieval failed: {e}")

        return None

    def retrieve_operation_knowledge(
        self, query: str
    ) -> Optional[str]:
        """
        Retrieve operation safety guidelines.

        Args:
            query: Retrieval query (typically an operation anchor like "Welding operation")

        Returns:
            Retrieved safety guidelines text, or None if retrieval fails
        """
        if self.operation_retriever is None:
            logger.warning("Operation retriever not available")
            return None

        try:
            results = self.operation_retriever.invoke(query)
            if results and len(results) > 0:
                return str(results[0].page_content)
        except Exception as e:
            logger.error(f"Operation knowledge retrieval failed: {e}")

        return None

    def retrieve_all_knowledge(
        self, object_queries: List[str], operation_queries: List[str]
    ) -> Dict[str, str]:
        """
        Batch retrieve all relevant knowledge.

        Retrieves safety guidelines for all provided object and operation anchors.

        Args:
            object_queries: List of object anchors
            operation_queries: List of operation anchors (typically at most one)

        Returns:
            Dictionary mapping anchor names to retrieved guidelines:
            {
                "Scaffolding": "Safety guidelines for scaffolding...",
                "Welding operation": "Safety guidelines for welding..."
            }
        """
        results = {}

        # Retrieve object knowledge
        for query in object_queries:
            knowledge = self.retrieve_object_knowledge(query)
            if knowledge:
                results[query] = knowledge

        # Retrieve operation knowledge
        for query in operation_queries:
            knowledge = self.retrieve_operation_knowledge(query)
            if knowledge:
                results[query] = knowledge

        return results

    def is_available(self) -> bool:
        """
        Check if both knowledge bases are available.

        Returns:
            True if both object and operation databases are loaded
        """
        return self.object_db is not None and self.operation_db is not None

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with knowledge base statistics
        """
        return {
            "object_db_loaded": self.object_db is not None,
            "operation_db_loaded": self.operation_db is not None,
            "object_db_path": self.object_db_path,
            "operation_db_path": self.operation_db_path,
            "embedding_model": self.embedding_model,
        }
