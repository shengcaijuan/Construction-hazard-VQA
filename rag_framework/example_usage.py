"""
Example Usage of COHS VQA RAG Framework

This script demonstrates how to use the framework for construction safety
hazard detection in images.
"""

from core import COHSVQAFramework


def main():
    """
    Example usage of the COHS VQA Framework.
    """

    # ============================================================================
    # 1. Initialize the Framework
    # ============================================================================
    framework = COHSVQAFramework(
        object_db_path="../knowledge_base/object_safety_guidelines_FAISS",
        operation_db_path="../knowledge_base/operation_safety_guidelines_FAISS",
        llm_model="gpt-4o",  # Options: gpt-4o, qwen-vl-max, glm-4v-flash, etc.
        temperature=0.0,
    )

    # ============================================================================
    # 2. Define Your Question
    # ============================================================================
    question = "What occupational health and safety hazards exist at the construction site in the image?"

    # ============================================================================
    # 3. Process an Image
    # ============================================================================
    image_path = "test_image.jpg"

    result = framework.answer_question(
        image_path=image_path,
        user_question=question,
    )

    # ============================================================================
    # 4. View Results
    # ============================================================================
    print("Extracted Anchors:")
    print(f"  Objects: {result['extracted_anchors']['objects']}")
    print(f"  Operations: {result['extracted_anchors']['operations']}")

    print(f"\nRetrieved Knowledge Entries: {len(result['retrieved_knowledge'])}")
    for anchor in result['retrieved_knowledge'].keys():
        print(f"  - {anchor}")

    print(f"\nGenerated Answer:")
    print(result["answer"])

    # ============================================================================
    # 5. Get Framework Information
    # ============================================================================
    print("\n" + "=" * 60)
    print("Framework Information")
    print("=" * 60)

    info = framework.get_framework_info()
    print(f"Model: {info['model']}")
    print(f"Two-Stage Retrieval: {info['use_two_stage_retrieval']}")
    print(f"Knowledge Chunking: {info['use_knowledge_chunking']}")


if __name__ == "__main__":
    main()