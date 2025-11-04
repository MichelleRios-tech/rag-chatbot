"""
LLM Provider Abstraction Layer

This module provides an abstraction layer for different LLM providers,
allowing the RAG system to work with both Anthropic's Claude API and
local models via LM Studio's OpenAI-compatible API.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import json
import anthropic
import openai
import google.generativeai as genai


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model: str):
        """
        Initialize the LLM provider.

        Args:
            api_key: API key for the provider
            model: Model name/identifier
        """
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[str, List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System prompt to guide the model
            messages: List of conversation messages
            tools: Optional list of tools the model can use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (response_text, tool_calls, full_messages_with_tool_results)
            - response_text: The final text response
            - tool_calls: List of tool calls made (empty if none)
            - full_messages_with_tool_results: Messages including tool results (None if no tools used)
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the provider is accessible and working.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        pass


class AnthropicProvider(LLMProvider):
    """Provider implementation for Anthropic's Claude API."""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)

    def _convert_tools_to_anthropic_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tools are already in Anthropic format from search_tools.py.
        Just return them as-is.
        """
        return tools

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[str, List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """Generate response using Anthropic's API."""

        # Prepare base parameters
        params = {
            "model": self.model,
            "system": system_prompt,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Add tools if provided
        if tools:
            params["tools"] = self._convert_tools_to_anthropic_format(tools)
            params["tool_choice"] = {"type": "auto"}

        # Make initial API call
        response = self.client.messages.create(**params)

        # Check if model wants to use tools
        if response.stop_reason == "tool_use":
            tool_calls = []

            # Extract tool use blocks
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })

            # Add assistant's response to messages
            updated_messages = messages.copy()
            updated_messages.append({"role": "assistant", "content": response.content})

            # Return tool calls - the AI generator will execute them and call again
            return "", tool_calls, updated_messages

        # Extract text response
        response_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text

        return response_text, [], None

    def test_connection(self) -> bool:
        """Test Anthropic API connection."""
        try:
            # Make a minimal API call to test connection
            self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            print(f"Anthropic connection test failed: {e}")
            return False

    def get_provider_name(self) -> str:
        return "Anthropic Claude"


class LMStudioProvider(LLMProvider):
    """Provider implementation for LM Studio's OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str, base_url: str = "http://localhost:1234/v1"):
        super().__init__(api_key, model)
        self.base_url = base_url
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _convert_tools_to_openai_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert tools from Anthropic format to OpenAI format.

        Anthropic format:
        {
            "name": "tool_name",
            "description": "...",
            "input_schema": {...}
        }

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "...",
                "parameters": {...}
            }
        }
        """
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })
        return openai_tools

    def _format_messages_with_system(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Combine system prompt with messages for OpenAI format.
        OpenAI expects system prompt as first message with role="system".
        Also converts tool results from Anthropic format to OpenAI format.
        """
        formatted_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            # Check if this is a user message with tool results (Anthropic format)
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Check if it contains tool results
                if any(isinstance(item, dict) and item.get("type") == "tool_result" for item in msg["content"]):
                    # Convert to OpenAI tool message format
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            formatted_messages.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item["content"]
                            })
                else:
                    formatted_messages.append(msg)
            else:
                formatted_messages.append(msg)

        return formatted_messages

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[str, List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """Generate response using LM Studio's OpenAI-compatible API."""

        # Format messages with system prompt
        formatted_messages = self._format_messages_with_system(system_prompt, messages)

        # Prepare base parameters
        params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Add tools if provided
        if tools:
            params["tools"] = self._convert_tools_to_openai_format(tools)
            params["tool_choice"] = "auto"

        # Make initial API call
        response = self.client.chat.completions.create(**params)

        # Check if model wants to use tools
        message = response.choices[0].message
        if message.tool_calls:
            tool_calls = []

            # Extract tool calls
            for tool_call in message.tool_calls:
                tool_calls.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": json.loads(tool_call.function.arguments)  # Parse JSON string to dict
                })

            # Add assistant's message with tool calls to messages
            updated_messages = formatted_messages.copy()
            updated_messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Return tool calls - the AI generator will execute them and call again
            return "", tool_calls, updated_messages

        # Extract text response
        response_text = message.content or ""

        return response_text, [], None

    def test_connection(self) -> bool:
        """Test LM Studio API connection."""
        try:
            # Make a minimal API call to test connection
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            print(f"LM Studio connection test failed: {e}")
            return False

    def get_provider_name(self) -> str:
        return f"LM Studio ({self.base_url})"


class GeminiProvider(LLMProvider):
    """Provider implementation for Google's Gemini API."""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name=model)

    def _convert_tools_to_gemini_format(self, tools: List[Dict[str, Any]]) -> List:
        """
        Convert tools from Anthropic format to Gemini FunctionDeclaration objects.

        Anthropic format:
        {
            "name": "tool_name",
            "description": "...",
            "input_schema": {...}
        }

        Gemini format: FunctionDeclaration objects
        """
        from google.ai.generativelanguage_v1beta.types import FunctionDeclaration, Schema, Type

        function_declarations = []
        for tool in tools:
            # Convert JSON Schema to Gemini Schema format
            input_schema = tool["input_schema"]

            # Map JSON Schema type to Gemini Type enum
            type_mapping = {
                "object": Type.OBJECT,
                "string": Type.STRING,
                "number": Type.NUMBER,
                "integer": Type.INTEGER,
                "boolean": Type.BOOLEAN,
                "array": Type.ARRAY
            }

            # Build Schema for parameters
            schema_dict = {}

            # Set the type
            if "type" in input_schema:
                schema_dict["type_"] = type_mapping.get(input_schema["type"], Type.STRING)

            # Add properties if present
            if "properties" in input_schema:
                properties = {}
                for prop_name, prop_def in input_schema["properties"].items():
                    prop_schema = {}

                    if "type" in prop_def:
                        prop_schema["type_"] = type_mapping.get(prop_def["type"], Type.STRING)

                    if "description" in prop_def:
                        prop_schema["description"] = prop_def["description"]

                    # Handle nested properties or arrays
                    if "items" in prop_def:
                        items_schema = {}
                        if "type" in prop_def["items"]:
                            items_schema["type_"] = type_mapping.get(prop_def["items"]["type"], Type.STRING)
                        prop_schema["items"] = Schema(**items_schema)

                    properties[prop_name] = Schema(**prop_schema)

                schema_dict["properties"] = properties

            # Add required fields if present
            if "required" in input_schema:
                schema_dict["required"] = input_schema["required"]

            # Add description if present
            if "description" in input_schema:
                schema_dict["description"] = input_schema["description"]

            # Create Schema
            parameters = Schema(**schema_dict)

            # Create FunctionDeclaration
            func_decl = FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=parameters
            )
            function_declarations.append(func_decl)

        return function_declarations

    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert messages from Anthropic format to Gemini format.
        Gemini uses 'user' and 'model' roles (not 'assistant').
        """
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Convert 'assistant' to 'model' for Gemini
            if role == "assistant":
                role = "model"

            # Handle complex content structures (e.g., tool results)
            if isinstance(content, list):
                # For tool results, we need to convert to Gemini's function response format
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item["text"]})
                        elif item.get("type") == "tool_result":
                            # Tool results will be handled separately in the main generate flow
                            pass
                    else:
                        parts.append({"text": str(item)})

                if parts:
                    gemini_messages.append({"role": role, "parts": parts})
            else:
                # Simple text message
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": content}]
                })

        return gemini_messages

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[str, List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """Generate response using Google's Gemini API."""

        # Prepare generation config
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        # Build the prompt by combining system prompt with the last user message
        # Gemini doesn't have a separate system prompt parameter, so we prepend it
        if messages and len(messages) > 0:
            # Get the last message (should be user message)
            last_message = messages[-1]["content"]
            combined_prompt = f"{system_prompt}\n\nUser: {last_message}"
        else:
            combined_prompt = system_prompt

        try:
            # Prepare tools if provided
            tools_param = None
            if tools:
                from google.ai.generativelanguage_v1beta.types import Tool

                gemini_function_declarations = self._convert_tools_to_gemini_format(tools)
                tools_param = [Tool(function_declarations=gemini_function_declarations)]

            # Make API call
            if tools_param:
                response = self.client.generate_content(
                    combined_prompt,
                    generation_config=generation_config,
                    tools=tools_param
                )
            else:
                response = self.client.generate_content(
                    combined_prompt,
                    generation_config=generation_config
                )

            # Check if model wants to use tools
            if response.candidates and response.candidates[0].content.parts:
                tool_calls = []
                text_parts = []

                for part in response.candidates[0].content.parts:
                    # Check for function call
                    if hasattr(part, 'function_call') and part.function_call:
                        # Extract tool call arguments
                        args_dict = {}
                        if part.function_call.args:
                            # Convert protobuf Struct to dict
                            for key, value in part.function_call.args.items():
                                # Handle different value types
                                if hasattr(value, 'string_value'):
                                    args_dict[key] = value.string_value
                                elif hasattr(value, 'number_value'):
                                    args_dict[key] = value.number_value
                                elif hasattr(value, 'bool_value'):
                                    args_dict[key] = value.bool_value
                                else:
                                    args_dict[key] = str(value)

                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",  # Gemini doesn't provide IDs
                            "name": part.function_call.name,
                            "input": args_dict
                        })

                    # Check for text
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                # If we have tool calls, return them
                if tool_calls:
                    # Build updated messages in Anthropic format for consistency
                    updated_messages = messages.copy()

                    # Add assistant's response with tool calls
                    response_content = []
                    if text_parts:
                        response_content.append({"type": "text", "text": " ".join(text_parts)})

                    for tc in tool_calls:
                        response_content.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["input"]
                        })

                    updated_messages.append({
                        "role": "assistant",
                        "content": response_content
                    })

                    return "", tool_calls, updated_messages

                # No tool calls, just return text
                if text_parts:
                    return " ".join(text_parts), [], None

            # Extract text response
            response_text = response.text if hasattr(response, 'text') else ""
            return response_text, [], None

        except Exception as e:
            print(f"Gemini API error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", [], None

    def test_connection(self) -> bool:
        """Test Gemini API connection."""
        try:
            # Make a minimal API call to test connection
            response = self.client.generate_content(
                "test",
                generation_config=genai.types.GenerationConfig(max_output_tokens=10)
            )
            return True
        except Exception as e:
            print(f"Gemini connection test failed: {e}")
            return False

    def get_provider_name(self) -> str:
        return "Google Gemini"


def create_provider(
    provider_type: str,
    api_key: str,
    model: str,
    base_url: Optional[str] = None
) -> LLMProvider:
    """
    Factory function to create an LLM provider instance.

    Args:
        provider_type: Type of provider ("anthropic", "gemini", or "lmstudio")
        api_key: API key for the provider
        model: Model name/identifier
        base_url: Base URL for LM Studio (optional)

    Returns:
        An instance of the requested provider

    Raises:
        ValueError: If provider_type is not recognized
    """
    if provider_type.lower() == "anthropic":
        return AnthropicProvider(api_key, model)
    elif provider_type.lower() == "gemini":
        return GeminiProvider(api_key, model)
    elif provider_type.lower() == "lmstudio":
        return LMStudioProvider(api_key, model, base_url or "http://localhost:1234/v1")
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
