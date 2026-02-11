"""
RAG Framework for Construction Occupational Health and Safety (COHS) Visual Question Answering

This module implements a plug-and-play visual knowledge enhancement framework based on RAG
to address the challenge of answering general open-ended questions about construction safety hazards.

Core Modules:
- VisualAnchorExtractor: Extract hazard-relevant objects and operations from images
- COHCKnowledgeBase: Manage FAISS vector databases for safety guidelines
- TwoStageRetriever: Implement agent-based dual-stage retrieval
- TempVectorDBBuilder: Build temporary vector databases for efficient multi-question scenarios
- KnowledgeChunker: Handle long text chunking for information overload
- COHSVQAFramework: Main framework integrating all modules
"""

__version__ = "1.1.0"

from core.vqa_framework import COHSVQAFramework
from core.temp_vector_db_builder import TempVectorDBBuilder

__all__ = [
    "COHSVQAFramework",
    "TempVectorDBBuilder",
]
