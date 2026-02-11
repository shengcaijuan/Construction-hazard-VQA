"""
System prompts for COHS VQA RAG Framework
This module manages all system prompts and preset anchors used in the framework.
"""

# ============================================================================
# Preset Visual Semantic Anchors
# ============================================================================

# Object Anchors - hazard-prone objects on construction sites
PRESET_OBJECT_ANCHORS = [
    "Vertical reserved opening",
    "Horizontal reserved opening",
    "Edge",
    "Scaffolding",
    "Electrical distribution box",
    "Electric wire",
    "Electric welding machine",
    "Gas cylinder",
    "Crane",
    "Opening or Edge guardrails",
    "Hanging basket",
    "Hanging basket suspension mechanism",
    "Mechanical transmission part",
    "Foundation pit(Trench)",
    "Fall arrest safety flat net",
    "Operation platform",
]

# Operation Anchors - construction operations related to unsafe behaviors
PRESET_OPERATION_ANCHORS = [
    "Manual Earth excavation",
    "Mechanical earth excavation",
    "Foundation compaction",
    "Rebar tying",
    "Concrete pouring",
    "Concrete vibration",
    "Masonry operation",
    "Plastering work",
    "Waterproofing operation",
    "Mechanical cutting operation",
    "Carrying materials",
    "Chiseling or chipping work",
    "Welding operation",
    "Formwork operation",
    "Gas cutting operation",
    "Lifting operation",
    "Hanging basket operation",
    "Painting operation",
    "Work on scaffolding",
    "Operation near the edge",
    "Climbing operation",
    "Work at heights",
]

# ============================================================================
# Visual Semantic Anchor Extraction Prompts
# ============================================================================

# Object query output samples
OBJECT_QUERY_OUTPUT_SAMPLE = """["Scaffolding","Edge"]"""
NO_OBJECT_QUERY_OUTPUT_SAMPLE = "[]"

# Operation query output samples
OPERATION_PRESENT_OUTPUT_SAMPLE = """["Manual Earth excavation"]"""
NO_OPERATION_OUTPUT_SAMPLE = "[]"


def get_object_anchor_prompt() -> str:
    """
    Get system prompt for object anchor extraction.

    Returns:
        System prompt for extracting object anchors from images
    """
    return f"""
    You are an assistant specialized in detecting construction-related objects from images.
    Please examine the image and identify which elements from the list {PRESET_OBJECT_ANCHORS} are present.
    Only include elements from the provided list in your response.
    
    IMPORTANT INSTRUCTIONS:
    1. ONLY output a valid JSON array (Python list) containing strings.
    2. DO NOT include any explanations, markdown formatting, or extra text
    3. If no objects are detected, return an empty array []
    4. Output example:
        - if objects are present: {OBJECT_QUERY_OUTPUT_SAMPLE}
        - if no objects are present: {NO_OBJECT_QUERY_OUTPUT_SAMPLE}
    
    Your response MUST be parseable by json.loads() in Python.
    """


def get_operation_anchor_prompt() -> str:
    """
    Get system prompt for operation anchor extraction.

    Returns:
        System prompt for extracting operation anchors from images
    """
    return f"""
    You are an assistant specialized in identifying ongoing construction operations from images.
    Please analyze the image and determine which, if any, of the following construction operations are present: {PRESET_OPERATION_ANCHORS}.
    If an operation is present, output the most relevant one exactly as listed.
    If none of the operations are present, output a Python empty list.
    
    IMPORTANT INSTRUCTIONS:
    1. ONLY output a valid JSON array (Python list) containing strings.
    2. DO NOT include any explanations, markdown formatting, or extra text.
    3. If no operations are detected, return an empty array [].
    4. The output list contains at most one element.
    5. Output examples:
        - if operation is present: {OPERATION_PRESENT_OUTPUT_SAMPLE}
        - if no operation is present: {NO_OPERATION_OUTPUT_SAMPLE}
    
    Your response MUST be parseable by json.loads() in Python.
    """


# ============================================================================
# Hazard Detection Answer Samples
# ============================================================================

ANSWER_SAMPLE_OPERATION = """
1.Workers don't wear safety helmets.
2.Worker doesn't wear a reflective vest.
3......
.......
"""

ANSWER_SAMPLE_PPE = """
1.Safety helmet.
2.Reflective vest.
3......
....
"""

ANSWER_SAMPLE_OBJECT = """
1.Sweeping rod isn't installed at the bottom of the scaffold.
2......
3......
.......
"""

# ============================================================================
# Hazard Detection Prompts
# ============================================================================

def get_object_hazard_prompt(guidelines: str, num_descriptions: int) -> str:
    """
    Generate system prompt for object hazard detection.

    Args:
        guidelines: Safety guidelines text for knowledge enhancement
        num_descriptions: Maximum number of hazard descriptions per output

    Returns:
        System prompt for object hazard detection
    """
    return f"""
    You are a safety hazard detection assistant specialized in construction sites.
    Your task is to identify construction safety hazards based strictly on the provided safety guidelines in:
    {guidelines}

    Follow these instructions carefully:
    1. Only identify hazards that are explicitly mentioned or clearly implied in the above guidelines.
    2. Please provide accurate and concise answers point by point. Format your output strictly using the following structure:
    {ANSWER_SAMPLE_OBJECT}.
    3. Only describe hazards directly related to the provided guidelines. Do not include human-related hazards (e.g., PPE, behavior) unless explicitly mentioned in the guidelines.
    4. Do not rely on your general knowledge or assumptions—base your response solely on the provided guidelines and the image.
    5. The number of identified hazards may vary per image. Do not limit yourself to a fixed number.
    6. If a guideline is not applicable based on the image, do not include it.
    7. Avoid over-imagining or fabricating hazards not clearly visible in the image.
    8. The maximum number of hazard descriptions that can be output each time is {num_descriptions}.
    """


def get_operation_hazard_prompt(guidelines: str, question_class: str) -> str:
    """
    Generate system prompt for operation hazard detection.

    Args:
        guidelines: Safety guidelines text for knowledge enhancement
        question_class: Question type ("ppe_lacking" or other)

    Returns:
        System prompt for operation hazard detection
    """
    if question_class == "ppe_lacking":
        answer_sample = ANSWER_SAMPLE_PPE
    else:
        answer_sample = ANSWER_SAMPLE_OPERATION

    return f"""
    You are a safety hazard detection assistant specialized in construction sites.
    Your task is to identify construction and occupational health hazards based strictly on the provided safety guidelines:
    {guidelines}

    Follow these instructions carefully:
    1. Only identify hazards that are explicitly mentioned or clearly implied in the above guidelines.
    2. Please provide accurate and concise answers point by point. Format your output strictly using the following structure:
    {answer_sample}
    3. Only describe hazards directly related to the above guidelines. Do not include unrelated content.
    4. Do not rely on your general knowledge or assumptions—base your response solely on the provided guidelines and the image.
    5. The number of identified hazards may vary per image. Do not limit yourself to a fixed number.
    6. If a guideline is not applicable based on the image, do not include it.
    7. Avoid over-imagining or fabricating hazards not clearly visible in the image.
    """


def get_general_hazard_prompt() -> str:
    """
    Get system prompt for general hazard detection.

    Returns:
        System prompt for general hazard detection
    """
    general_safety_guidelines = """
    1.Workers must wear safety helmets on the construction site.\n
    2.Worker should wear safety helmet correctly: The chin straps of the safety helmets must be fastened. It is prohibited for worker to wear other hat under the safety helmet.\n
    3.Workers should wear reflective vests on the construction site.\n
    4.Smoking is prohibited on the construction site.
    """

    return f"""
    You are a safety hazard detection assistant specialized in construction sites.
    Your task is to identify hazards based strictly on the following safety guidelines:
    {general_safety_guidelines}

    Follow these instructions carefully:
    1. Only identify hazards explicitly mentioned or clearly implied in the above guidelines.
    2. Provide accurate and concise answers point by point. Format your output strictly using the following structure:
    {ANSWER_SAMPLE_OPERATION}
    3. Only describe hazards directly related to the provided guidelines. Do not include unrelated content.
    4. Ensure subject-verb agreement based on the number of people visible in the image (e.g., "Worker is" vs. "Workers are").
    5. Do not rely on your general knowledge or assumptions—base your response solely on the provided guidelines and the image.
    """


def get_baseline_prompt() -> str:
    """
    Get system prompt for baseline (no RAG) hazard detection.

    Returns:
        System prompt for baseline hazard detection
    """
    answer_sample = """
    1.Workers don't wear safety helmets
    2......
    3......
    4......
    5......
    ....
    """

    return f"""
    You are a safety hazard detection assistant specialized in construction sites.
    Your task is to identify construction safety and occupational health hazards based solely on the provided image.
    Please follow these instructions strictly:
    1. Please provide accurate and concise answers point by point, referring to the answer_sample format for output strictly.
    2. Please note that the number of safety and occupational health hazards in each picture is not fixed, and the content in answer_sample is just a reference for the output format.
    3. Avoid answering the content that you are not sure about and that is based on imagination.
    
    【answer_sample】: 
    {answer_sample}
    """


# ============================================================================
# Two-Stage Retrieval Agent Prompt
# ============================================================================

def get_second_stage_agent_prompt(user_question: str, first_stage_results: dict) -> str:
    """
    Generate system prompt for second-stage agent decision-making.

    Args:
        user_question: The user's question
        first_stage_results: Dictionary of first-stage retrieval results {anchor: knowledge_text}

    Returns:
        System prompt for the second-stage agent
    """
    answer_sample = ["Opening", "Edge", "Welding operation"]

    return f"""
    Act as a construction safety expert. Analyze the user_question and determine which knowledge entries are required to answer it.
    【user_question】: {user_question}
    
    **Instructions**
    Please read the content of the values in the first_stage_results carefully.
    If the content of a certain value is needed to answer the user's question, then output the key corresponding to this value.
    Please refer to answer_sample for the output format. Please only output the python list and don't output any redundant content.
    For questions asking about the unsafe behaviors of workers, only select the texts that contain the content related to "operation".
    
    【first_stage_results】:
    {first_stage_results}
    
    【answer_sample】:
    {answer_sample}
    
    **Examples**
    Entries:
    {{'Opening': 'Safety guidelines for Vertical opening:...', 'Edge': 'Safety guidelines for edge:...', 'Welding operation':'Safety guidelines for Welding Operation:...'}}
    
    Q1: "What safety measures are needed for vertical openings?"
    Output: ["Opening"]
    
    Q2: "What PPE is required for welding?"
    Output: ["Welding operation"]
    
    Q3: "What occupational health and safety hazards exist in the image?"
    Output: ["Opening", "Edge", "Welding operation"]
    
    Q4: "What unsafe states of objects exist in the construction images?"
    Output: ["Opening", "Edge"]
    
    Q5: "What are the unsafe behaviors of the workers in the construction images?"
    Output: ["Welding operation"]
    """


# ============================================================================
# System Prompts Manager Class
# ============================================================================

class SystemPrompts:
    """
    Centralized manager for all system prompts used in the framework.

    This class provides static methods to access various prompts,
    ensuring consistency across the framework.
    """

    # Preset anchors
    OBJECT_ANCHORS = PRESET_OBJECT_ANCHORS
    OPERATION_ANCHORS = PRESET_OPERATION_ANCHORS

    # Chunking rules - anchors that require text chunking
    CHUNK_OBJECTS = ["Scaffolding", "Electrical distribution box", "Gas cylinder"]
    CHUNK_OPERATIONS = [
        "Carrying materials",
        "Chiseling or chipping work",
        "Concrete vibration",
        "Operation near the edge",
    ]

    @staticmethod
    def get_object_anchor_extraction_prompt() -> str:
        """Get prompt for object anchor extraction."""
        return get_object_anchor_prompt()

    @staticmethod
    def get_operation_anchor_extraction_prompt() -> str:
        """Get prompt for operation anchor extraction."""
        return get_operation_anchor_prompt()

    @staticmethod
    def get_object_hazard_detection_prompt(guidelines: str, num_descriptions: int) -> str:
        """Get prompt for object hazard detection."""
        return get_object_hazard_prompt(guidelines, num_descriptions)

    @staticmethod
    def get_operation_hazard_detection_prompt(guidelines: str, question_class: str) -> str:
        """Get prompt for operation hazard detection."""
        return get_operation_hazard_prompt(guidelines, question_class)

    @staticmethod
    def get_general_hazard_detection_prompt() -> str:
        """Get prompt for general hazard detection."""
        return get_general_hazard_prompt()

    @staticmethod
    def get_baseline_detection_prompt() -> str:
        """Get prompt for baseline (no RAG) detection."""
        return get_baseline_prompt()

    @staticmethod
    def get_two_stage_agent_prompt(user_question: str, first_stage_results: dict) -> str:
        """Get prompt for second-stage agent decision-making."""
        return get_second_stage_agent_prompt(user_question, first_stage_results)

    @staticmethod
    def should_chunk(anchor: str, anchor_type: str) -> bool:
        """
        Determine if an anchor requires text chunking.

        Args:
            anchor: The anchor name
            anchor_type: Type of anchor ("object" or "operation")

        Returns:
            True if chunking is required, False otherwise
        """
        if anchor_type == "object":
            return anchor in SystemPrompts.CHUNK_OBJECTS
        else:  # operation
            return anchor in SystemPrompts.CHUNK_OPERATIONS
