"""
Unified LLM Client for COHS VQA RAG Framework

This module provides a unified interface for interacting with various LLM providers
including OpenAI and Qwen.
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Supports:
    - OpenAI: gpt-4o, gpt-4o-mini
    - Qwen: qwen-vl-max, qwen2.5-vl-72b-instruct
    """

    # Model-specific configurations
    MODEL_CONFIGS = {
        "gpt-4o": {"api_key": os.getenv("API_KEY"), "base_url": os.getenv("BASE_URL")},
        "qwen2.5-vl-72b-instruct": {"api_key": os.getenv("API_KEY"), "base_url": os.getenv("BASE_URL")}
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model name (e.g., "gpt-4o", "qwen2.5-vl-72b-instruct")
            temperature: Temperature parameter for generation
            api_key: API key
            base_url: base URL
        """
        self.model = model
        self.temperature = temperature

        # Get model-specific config
        config = self.MODEL_CONFIGS.get(model, {})

        # Use provided credentials or defaults
        self.api_key = api_key or config.get("api_key")
        self.base_url = base_url or config.get("base_url")

        # Create OpenAI client (works with compatible APIs)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Invoke the LLM with text messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            response_format: Response format (e.g., {"type": "json_object"})

        Returns:
            LLM response content as string
        """
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature}

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def invoke_with_vision(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Invoke the multimodal LLM with image and text.

        Args:
            image_base64: Base64-encoded image
            system_prompt: System prompt
            user_prompt: User prompt
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response content as string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def set_temperature(self, temperature: float) -> None:
        """Update the temperature parameter."""
        self.temperature = temperature
