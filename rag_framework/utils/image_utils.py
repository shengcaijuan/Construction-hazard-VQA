"""
Image processing utilities for COHS VQA RAG Framework
"""

import base64
from typing import Optional


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64-encoded image string

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If the image file cannot be read
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def renumber_sentences(original_string: str) -> str:
    """
    Renumber sentences in a string.

    This function takes a string where each line is treated as a sentence.
    It renumbers each sentence and removes empty lines.

    Args:
        original_string: Original string with multiple sentences, one per line

    Returns:
        Renumbered string with consecutive numbering

    Example:
        Input:  "1. First sentence\\n3. Third sentence\\n   Unnumbered line"
        Output: "1. First sentence\\n2. Third sentence\\n3. Unnumbered line"
    """
    lines = []
    for line in original_string.splitlines():
        stripped_line = line.strip()
        if stripped_line:
            # Check if line already starts with a number
            if not stripped_line.startswith(
                ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "0.", "10.")
            ):
                stripped_line = f"1. {stripped_line}"
            lines.append(stripped_line)

    new_lines = []
    count = 1
    for line in lines:
        # Split string, remove original number, add new consecutive number
        new_line = f"{count}. {line.split('.', 1)[1].strip()}"
        new_lines.append(new_line)
        count += 1

    return "\n".join(new_lines)


def validate_image_path(image_path: str) -> bool:
    """
    Validate that an image file exists and is readable.

    Args:
        image_path: Path to the image file

    Returns:
        True if the image file is valid, False otherwise
    """
    try:
        with open(image_path, "rb") as f:
            # Read a few bytes to verify the file is readable
            f.read(1024)
        return True
    except (FileNotFoundError, IOError, OSError):
        return False


def get_image_extension(image_path: str) -> Optional[str]:
    """
    Get the file extension of an image.

    Args:
        image_path: Path to the image file

    Returns:
        File extension including the dot (e.g., ".jpg", ".png"), or None if no extension
    """
    if "." in image_path:
        return image_path.rsplit(".", 1)[-1].lower()
    return None
