"""
Temporary Vector Database Builder for COHS VQA RAG Framework

This module implements a lightweight temporary vector database that stores
retrieved knowledge chunks for a specific image, enabling efficient secondary
retrieval for multiple questions about the same image.

Key features:
- In-memory caching
- Image-specific knowledge subset
- Unified temp DB combining object + operation chunks
- Automatic cache management with VQA instance lifecycle
- Consistent vector space using same embedding model
"""

import logging
from typing import Dict, List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from .knowledge_base import COHSKnowledgeBase

logger = logging.getLogger(__name__)


class TempVectorDBBuilder:
    """
    Temporary Vector Database Builder for COHS VQA.

    Builds a lightweight, in-memory vector database containing only the
    knowledge relevant to a specific image. This enables efficient
    secondary retrieval without repeating the initial retrieval process.

    Features:
    - Pure memory caching
    - Bound to VQA instance lifecycle
    - Uses image file path as cache key
    - Automatic cleanup on instance destruction
    - Unified temp DB

    Usage:
        builder = TempVectorDBBuilder(knowledge_base=kb)
        temp_db = builder.get_or_build(
            image_path="./images/scaffold1.jpg",
            object_queries=["Scaffolding"],
            operation_queries=["Work on scaffolding"]
        )
    """

    def __init__(
        self,
        knowledge_base: COHSKnowledgeBase,
        embeddings_model: str = "text-embedding-3-large",
    ):
        """
        Initialize the temporary vector database builder.

        Args:
            knowledge_base: Main COHS knowledge base for first-stage retrieval
            embeddings_model: Embedding model name (must match the one used for main KB)
        """
        self.knowledge_base = knowledge_base
        self.embeddings_model = embeddings_model
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)

        # In-memory cache: image_path -> unified_temp_db (single FAISS DB)
        # Stores retrieved chunks from BOTH object and operation queries
        self._cache: Dict[str, Optional[FAISS]] = {}

        logger.info(
            f"TempVectorDBBuilder initialized with embeddings: {embeddings_model}"
        )

    def get_or_build(
        self,
        image_path: str,
        object_queries: List[str],
        operation_queries: List[str],
    ) -> Optional[FAISS]:
        """
        Get cached temporary database or build a new unified one.

        If the image has been processed before, returns the cached database.
        Otherwise, performs first-stage retrieval and builds a new unified
        temporary database containing BOTH object and operation chunks.

        Args:
            image_path: Image file path (used as cache key)
            object_queries: List of object anchor queries
            operation_queries: List of operation anchor queries

        Returns:
            Unified FAISS temp database containing both object and operation chunks
            Returns None if no documents were retrieved
        """
        # Normalize path for consistent caching
        normalized_path = self._normalize_path(image_path)

        # Check cache
        if normalized_path in self._cache:
            logger.debug(f"Using cached temp DB for {normalized_path}")
            return self._cache[normalized_path]

        # Build new unified temporary database
        logger.info(f"Building unified temp DB for {normalized_path}")
        temp_db = self._build_unified_temp_db(object_queries, operation_queries)

        # Cache the result
        if temp_db is not None:
            self._cache[normalized_path] = temp_db

        return temp_db

    def _build_unified_temp_db(
        self, object_queries: List[str], operation_queries: List[str]
    ) -> Optional[FAISS]:
        """
        Build a unified temporary FAISS database from first-stage retrieval results.

        Args:
            object_queries: List of object anchor queries
            operation_queries: List of operation anchor queries

        Returns:
            Unified FAISS temp database containing both object and operation chunks
            Returns None if no documents were retrieved
        """
        all_documents = []
        object_documents = []
        operation_documents = []

        # Retrieve object documents
        if object_queries:
            object_documents = self._retrieve_object_documents(object_queries)
            all_documents.extend(object_documents)
            logger.debug(f"Retrieved {len(object_documents)} object documents")

        # Retrieve operation documents
        if operation_queries:
            operation_documents = self._retrieve_operation_documents(operation_queries)
            all_documents.extend(operation_documents)
            logger.debug(f"Retrieved {len(operation_documents)} operation documents")

        # Build unified temp database from all documents
        if not all_documents:
            logger.warning("No documents retrieved for temp DB construction")
            return None

        temp_db = self._build_temp_db(all_documents)
        logger.info(
            f"Built unified temp DB with {len(all_documents)} total docs "
            f"(object: {len(object_documents) if object_queries else 0}, "
            f"operation: {len(operation_documents) if operation_queries else 0})"
        )

        return temp_db

    def _retrieve_object_documents(self, queries: List[str]) -> List[Document]:
        """
        Retrieve object-related documents from main knowledge base.

        Args:
            queries: List of object anchor queries

        Returns:
            List of retrieved documents
        """
        documents = []

        for query in queries:
            knowledge = self.knowledge_base.retrieve_object_knowledge(query)
            if knowledge:
                documents.append(Document(
                    page_content=knowledge,
                    metadata={"source": query, "type": "object"}
                ))
            else:
                logger.warning(f"No object knowledge retrieved for query: {query}")

        return documents

    def _retrieve_operation_documents(self, queries: List[str]) -> List[Document]:
        """
        Retrieve operation-related documents from main knowledge base.

        Args:
            queries: List of operation anchor queries

        Returns:
            List of retrieved documents
        """
        documents = []

        for query in queries:
            knowledge = self.knowledge_base.retrieve_operation_knowledge(query)
            if knowledge:
                documents.append(Document(
                    page_content=knowledge,
                    metadata={"source": query, "type": "operation"}
                ))
            else:
                logger.warning(f"No operation knowledge retrieved for query: {query}")

        return documents

    def _build_temp_db(self, documents: List[Document]) -> FAISS:
        """
        Build a temporary FAISS database from documents.

        Creates an in-memory FAISS vector store using the same embedding
        model as the main knowledge base to maintain vector space consistency.

        Args:
            documents: List of documents to index

        Returns:
            FAISS vector store (in-memory)
        """
        if not documents:
            raise ValueError("Cannot build temp DB from empty document list")

        # Use from_documents with in-memory storage
        temp_db = FAISS.from_documents(documents, self.embeddings)

        return temp_db

    def retrieve_from_temp(
        self,
        query: str,
        image_path: str,
        top_k: int = 1,
    ) -> List[Document]:
        """
        Retrieve documents from the unified temporary database.

        Args:
            query: Retrieval query
            image_path: Image file path (cache key)
            top_k: Number of top results to retrieve

        Returns:
            List of retrieved documents (may include both object and operation chunks)
        """
        normalized_path = self._normalize_path(image_path)

        if normalized_path not in self._cache:
            logger.warning(f"No cached temp DB for {normalized_path}")
            return []

        temp_db = self._cache[normalized_path]

        if temp_db is None:
            logger.warning(f"Temp DB is None for {normalized_path}")
            return []

        try:
            results = temp_db.similarity_search(query, k=top_k)
            return results
        except Exception as e:
            logger.error(f"Temp DB retrieval failed: {e}")
            return []

    def clear(self, image_path: Optional[str] = None) -> None:
        """
        Clear cached temporary databases.

        Args:
            image_path: Specific image path to clear.
                       If None, clears all cached databases.
        """
        if image_path is None:
            self._cache.clear()
            logger.info("Cleared all temp DB caches")
        else:
            normalized_path = self._normalize_path(image_path)
            if normalized_path in self._cache:
                del self._cache[normalized_path]
                logger.info(f"Cleared temp DB cache for {normalized_path}")

    def has_cache(self, image_path: str) -> bool:
        """
        Check if a temporary database exists for the given image.

        Args:
            image_path: Image file path to check

        Returns:
            True if cached database exists
        """
        normalized_path = self._normalize_path(image_path)
        return normalized_path in self._cache

    def get_cache_size(self) -> int:
        """
        Get the number of cached temporary databases.

        Returns:
            Number of cached images
        """
        return len(self._cache)

    @staticmethod
    def _normalize_path(path: str) -> str:
        """
        Normalize file path for consistent caching.

        Args:
            path: Original file path

        Returns:
            Normalized path
        """
        # Convert to absolute path and normalize separators
        import os

        return os.path.normpath(os.path.abspath(path))

    def __del__(self):
        """Cleanup when instance is destroyed."""
        if hasattr(self, "_cache"):
            cache_size = len(self._cache)
            if cache_size > 0:
                logger.debug(f"TempVectorDBBuilder destroyed with {cache_size} cached DBs")
