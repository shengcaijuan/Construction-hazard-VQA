# COHS VQA RAG Framework

**Construction Occupational Health and Safety Visual Question Answering Framework**

A plug-and-play visual knowledge enhancement framework based on RAG (Retrieval-Augmented Generation) for answering general open-ended questions about construction safety hazards.

## Overview

This framework implements the method described in the research paper:
> "A Plug-and-Play Visual Knowledge Enhancement Framework Based on RAG for Open-Ended Visual Question Answering in Construction Hazard Detection"

### Four Core Modules

1. **External COHS Knowledge Base** - FAISS vector databases storing safety guidelines
2. **Visual Semantic Anchor Extraction** - Extract hazard-relevant elements from images
3. **Two-Stage Retrieval** - Agent-based intelligent knowledge filtering
4. **Chunked Knowledge Delivery** - Handle long text blocks to reduce overload

## Installation

### Requirements

```bash
pip install openai langchain-openai langchain-community pandas colorama
```

### Environment Variables

Set the following environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_API_BASE="your-custom-base-url"  # Optional
export DASHSCOPE_API_KEY="your-alibaba-api-key"  # For Qwen models
export ZHIPU_API_KEY="your-zhipu-api-key"  # For GLM models
```

## Quick Start

```python
from rag_framework.core import COHSVQAFramework

# Initialize the framework
framework = COHSVQAFramework(
    object_db_path="knowledge_base/object_safety_guidelines_FAISS",
    operation_db_path="knowledge_base/operation_safety_guidelines_FAISS",
    llm_model="gpt-4o"
)

# Answer a question
result = framework.answer_question(
    image_path="test_image.jpg",
    user_question="What occupational health and safety hazards exist?"
)

print(result["answer"])
```

## Project Structure

```
rag_framework/
├── core/                          # Core modules
│   ├── visual_anchor_extractor.py # Module 2: Visual Semantic Anchor Extraction
│   ├── knowledge_base.py          # Module 1: External COHS Knowledge Base
│   ├── two_stage_retriever.py     # Module 3: Two-Stage Retrieval
│   ├── knowledge_chunker.py       # Module 4: Chunked Knowledge Delivery
│   └── vqa_framework.py           # Main framework class
├── models/
│   └── llm_client.py              # Unified LLM client
├── prompts/
│   └── system_prompts.py          # System prompts management
├── utils/
│   └── image_utils.py             # Image processing utilities
└── example_usage.py               # Usage examples
```

## Supported Models

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Alibaba Qwen**: `qwen-vl-max`, `qwen2.5-vl-72b-instruct`
- **Zhipu GLM**: `glm-4v-flash`, `glm-4.5v`
- **Google Gemini**: `gemini-2.0-flash-exp`

## Usage Examples

### Full Framework (Recommended)

```python
result = framework.answer_question(
    image_path="test.jpg",
    user_question="What unsafe behaviors exist?",
    enable_two_stage_retrieval=True,  # Use agent filtering
    enable_chunking=True              # Use knowledge chunking
)
```

### Single-Stage Retrieval

```python
result = framework.answer_question(
    image_path="test.jpg",
    user_question="What unsafe behaviors exist?",
    enable_two_stage_retrieval=False,  # Skip agent filtering
    enable_chunking=True
)
```

### Baseline (No RAG)

```python
baseline_answer = framework.answer_without_rag(
    image_path="test.jpg",
    user_question="What hazards exist?"
)
```

### Question Types

The framework supports four types of questions:

1. **COHS Hazards**: General occupational health and safety hazards
2. **Unsafe Behaviors**: Worker unsafe behaviors
3. **PPE Lacking**: Missing personal protective equipment
4. **Unsafe Objects**: Hazardous object states

## Framework Workflow

```
Input: Image + User Question

Step 1: Visual Semantic Anchor Extraction
  ├─ Extract object anchors (e.g., "Scaffolding", "Edge")
  └─ Extract operation anchors (e.g., "Welding operation")

Step 2: Two-Stage Retrieval
  ├─ First Stage: Retrieve image-relevant safety guidelines
  └─ Second Stage: Agent filters for question-relevant content

Step 3: Chunked Knowledge Processing
  ├─ For long texts: Split into chunks and process iteratively
  └─ For short texts: Single-pass processing

Step 4: Answer Generation
  └─ Aggregate and format comprehensive answer

Output: Structured safety hazard description
```

## Ablation Study

The framework supports ablation studies to validate each module's contribution:

| Test | Configuration | Purpose |
|------|--------------|---------|
| Test1 | Baseline (No RAG) | Reference |
| Test3 | Single-stage retrieval | Validate anchor extraction |
| Test4 | Two-stage retrieval | Validate agent filtering |
| Test5 | Complete framework | Full system |

## Output Format

```python
{
    "answer": "1. Workers don't wear safety helmets.\n2. Scaffolding lacks bottom sweeping rods...",
    "extracted_anchors": {
        "objects": ["Scaffolding", "Edge"],
        "operations": ["Welding operation"]
    },
    "retrieved_knowledge": {
        "Scaffolding": "Safety guidelines for scaffolding...",
        "Welding operation": "Safety guidelines for welding..."
    }
}
```

## Paper Method Mapping

| Paper Module | Code Class |
|--------------|------------|
| External COHS Knowledge Base | `COHCKnowledgeBase` |
| Visual Semantic Anchor Extraction | `VisualAnchorExtractor` |
| Two-Stage Retrieval | `TwoStageRetriever` |
| Chunked Knowledge Delivery | `KnowledgeChunker` |
| Complete Framework | `COHSVQAFramework` |

## License

This code is for research and educational purposes.

## Citation

If you use this framework in your research, please cite our paper.
