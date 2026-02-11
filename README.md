# COHS VQA RAG Framework

A Retrieval-Augmented Generation (RAG) framework for Construction Occupational Health and Safety (COHS) Visual Question Answering, based on Multimodal Large Language Models (MLLMs).

## Overview

This framework implements a pluggable visual knowledge enhancement system that transforms general-purpose MLLMs into domain-specific expert assistants for construction safety hazard oversight. It addresses three critical challenges in construction OHS VQA:

- **Cross-modal semantic misalignment** - Natural language questions don't match visual content
- **Knowledge demand variability** - Different questions require different amounts of retrieved knowledge
- **Information overload** - Too much information causes cognitive bias in model responses

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     COHS VQA RAG Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐                                              │
│  │ VisualAnchor   │ Extract hazard-relevant objects & operations  │
│  │ Extractor      │ → object_anchors, operation_anchors          │
│  └────────────────┘                                              │
│                           ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              TwoStageRetriever                           │     │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐ │     │
│  │  │  Temp Vector DB │    │   Agent-based Filtering     │ │     │
│  │  │  (Unified Cache)│    │   (Secondary Retrieval)     │ │     │
│  │  └─────────────────┘    └─────────────────────────────┘ │     │
│  └─────────────────────────────────────────────────────────┘     │
│                           ↓                                       │
│  ┌────────────────┐                                              │
│  │ Knowledge      │ Handle long text with chunking strategy      │
│  │ Chunker        │ → Processed knowledge chunks                 │
│  └────────────────┘                                              │
│                           ↓                                       │
│  ┌────────────────┐                                              │
│  │ VQA Framework  │ Generate final answer with MLLM              │
│  │                │ → Context-aware safety hazard analysis       │
│  └────────────────┘                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. VisualAnchorExtractor

**Purpose**: Extracts hazard-relevant semantic anchors from construction images.

**Functionality**:
- Identifies 15 types of critical objects prone to unsafe states (e.g., Scaffolding, Excavation, Electric equipment)
- Identifies 20 types of operational scenarios associated with unsafe behaviors (e.g., Work at height, Welding operation)
- Uses MLLMs to match visual features with predefined semantic keywords
- Returns structured anchors as retrieval queries for the knowledge base

**Key Methods**:
- `extract_anchors(image_path: str) -> Dict[str, List[str]]` - Extract object and operation anchors from an image

**Corresponds to**: Module 2 (Visual Semantic Anchor Extraction) in the paper

---

### 2. COHSKnowledgeBase

**Purpose**: Manages external construction OHS knowledge bases with FAISS vector databases.

**Functionality**:
- Maintains two separate vector databases for object and operation safety guidelines
- Uses "text-embedding-3-large" embedding model for vectorization
- Organizes knowledge into 16 object chunks and 22 operation chunks
- Each chunk focuses on specific objects or operations with visual relevance
- Provides efficient similarity search using Euclidean (L2) distance

**Knowledge Structure**:
- **Object Guidelines**: 87 rules targeting 15 critical construction objects
- **Operation Guidelines**: 144 rules covering 20 operational scenarios

**Key Methods**:
- `retrieve_object_knowledge(query: str) -> str` - Retrieve object-related safety guidelines
- `retrieve_operation_knowledge(query: str) -> str` - Retrieve operation-related safety guidelines
- `retrieve_all_knowledge(object_queries, operation_queries) -> Dict` - Retrieve all relevant knowledge

**Corresponds to**: Module 1 (External Knowledge Base) in the paper

---

### 3. TempVectorDBBuilder

**Purpose**: Builds and caches unified temporary vector databases for efficient multi-question scenarios.

**Design Principle**:
According to the paper: *"re-embeds the initially retrieved text chunks into **a** lightweight temporary knowledge base"*

The temporary database combines BOTH object and operation chunks into a **single unified** FAISS database, not separate databases.

**Functionality**:
- Performs first-stage retrieval from main knowledge base
- Re-embeds retrieved chunks into a unified in-memory FAISS database
- Caches databases by image path for subsequent questions
- Eliminates redundant retrieval operations for the same image
- Maintains vector space consistency using the same embedding model

**Key Methods**:
- `get_or_build(image_path, object_queries, operation_queries) -> FAISS` - Get or build unified temp DB
- `retrieve_from_temp(query, image_path, top_k) -> List[Document]` - Retrieve from cached temp DB
- `clear(image_path=None)` - Clear cached databases
- `has_cache(image_path) -> bool` - Check if cache exists

**Cache Structure**:
```python
_cache: Dict[str, Optional[FAISS]]
# Example:
{
    "./images/scaffold1.jpg": unified_temp_db,  # Contains both object + operation chunks
    "./images/electric1.jpg": unified_temp_db
}
```

**Corresponds to**: Module 3 (Temporary Knowledge Base) in the paper

---

### 4. TwoStageRetriever

**Purpose**: Implements intelligent dual-stage retrieval with agent-based decision-making.

**Functionality**:

**First Stage (Anchor-based Retrieval)**:
- Uses visual semantic anchors as retrieval queries
- Retrieves image-relevant safety guidelines from knowledge base
- Optionally uses cached temporary database for efficiency

**Second Stage (Agent-based Filtering)**:
- Leverages MLLM's reasoning capabilities as an agent
- Analyzes user question to determine which knowledge chunks are relevant
- Generates secondary retrieval queries based on question type
- Dynamically customizes knowledge volume for different questions

**Agent Decision Logic**:
- "What hazards exist?" → Select all retrieved chunks
- "How many types of PPE are missing?" → Select only PPE-related chunks
- "What unsafe behaviors?" → Select only operation-related chunks

**Key Methods**:
- `retrieve(user_question, object_queries, operation_queries, image_path) -> Dict` - Execute two-stage retrieval
- `first_stage_retrieve(object_queries, operation_queries) -> Dict` - Anchor-based retrieval
- `second_stage_filter(user_question, first_stage_results) -> List[str]` - Agent-based filtering

**Corresponds to**: Module 3 (Two-Stage Retrieval) in the paper

---

### 5. KnowledgeChunker

**Purpose**: Handles information overload through chunked knowledge delivery strategy.

**Problem Addressed**:
When numerous text chunks are input simultaneously, the model struggles to focus on violated rules amidst lengthy text, leading to erroneous outputs.

**Functionality**:
- Further divides long text chunks into smaller, processable pieces
- Implements step-by-step processing flow for MLLMs
- In each round, only one single text chunk is injected
- MLLM responds based on current knowledge chunk
- Cumulative results from all rounds form the final answer

**Key Methods**:
- `chunk_knowledge(knowledge_dict: Dict[str, str]) -> List[Dict]` - Split knowledge into chunks
- `process_with_chunking(question: str, knowledge_chunks, llm_client) -> str` - Process with chunking strategy

**Corresponds to**: Module 4 (Chunked Knowledge Delivery) in the paper

---

### 6. COHSVQAFramework

**Purpose**: Main framework that integrates all modules into a cohesive system.

**Functionality**:
- Orchestrates the complete VQA pipeline
- Manages module interactions and data flow
- Provides unified interface for hazard detection queries
- Supports configuration customization (model selection, feature toggles)
- Enables ablation studies through module control

**Pipeline Flow**:
1. Extract visual semantic anchors from input image
2. Perform two-stage retrieval (with optional temp DB caching)
3. Apply knowledge chunking for long content
4. Generate final answer using MLLM

**Configurable Features**:
- `enable_two_stage_retrieval` - Toggle agent-based filtering
- `enable_knowledge_chunking` - Toggle chunked knowledge delivery
- `use_temp_db` - Toggle temporary vector database caching

**Key Methods**:
- `answer_question(image_path, user_question) -> Dict` - Main VQA interface
- `answer_without_rag(image_path, user_question) -> str` - Baseline without RAG
- `get_framework_info() -> Dict` - Get framework configuration and status
- `clear_temp_db_cache(image_path=None)` - Clear temporary database cache

**Result Structure**:
```python
{
    "answer": "Generated hazard description",
    "extracted_anchors": {
        "objects": ["Scaffolding", "Edge"],
        "operations": ["Work on scaffolding"]
    },
    "retrieved_knowledge": {
        "Scaffolding": "Safety guidelines...",
        "Work on scaffolding": "Safety guidelines..."
    }
}
```

---

## Module Interactions

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Module Interaction Flow                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  COHSVQAFramework (Main Orchestrator)                                │
│       │                                                              │
│       ├── VisualAnchorExtractor → Extract anchors from image         │
│       │                                                              │
│       ├── COHSKnowledgeBase ────────┐                               │
│       │        ↑                     │                               │
│       │        │                     │                               │
│       │   TempVectorDBBuilder ◄─────┘ (First-stage retrieval)       │
│       │        │                     │                               │
│       │        │ (Cached unified DB) │                               │
│       │        ↓                     │                               │
│       ├── TwoStageRetriever ─────────┘ (Second-stage filtering)      │
│       │                                                              │
│       ├── KnowledgeChunker ──────────→ Handle long text             │
│       │                                                              │
│       └── LLM Answer Generation ──────→ Generate final answer        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Research Foundation

This framework is based on the research paper:

**"Retrieval-Augmented Multimodal Large Language Models for Visual Question Answering of Construction Occupational Health and Safety Hazards"**

The framework implements four key innovations from the paper:

1. **Construction OHS Knowledge Base** - Domain-specific external knowledge with visual relevance
2. **Visual Semantic Anchor Extraction** - Cross-modal semantic alignment
3. **Dual-Stage Retrieval with Agent Decision** - Dynamic knowledge customization
4. **Chunked Knowledge Delivery** - Information overload mitigation

---

## Version

**Current Version**: 1.1.0

---

## Citation

If you use this framework in your research, please cite the original paper.