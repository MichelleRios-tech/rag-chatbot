"""
Provider Registry for Model Discovery and Management

This module handles dynamic discovery of available LLM providers and their models.
It checks API key availability, fetches models from provider APIs, and manages
provider instantiation.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import google.generativeai as genai
import openai
from config import config
from llm_provider import create_provider, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an available model."""
    model_id: str
    display_name: str
    provider: str
    supports_tools: bool
    context_window: int
    is_default: bool = False


@dataclass
class ProviderInfo:
    """Information about an available provider."""
    name: str
    display_name: str
    models: List[ModelInfo]
    is_available: bool = True


class ProviderRegistry:
    """
    Registry for managing LLM providers and their available models.

    Handles:
    - Provider discovery based on API key availability
    - Dynamic model fetching from provider APIs
    - Model list caching (5-minute TTL)
    - Provider instantiation
    """

    def __init__(self, config_obj=None):
        """
        Initialize the provider registry.

        Args:
            config_obj: Configuration object (defaults to global config)
        """
        self.config = config_obj or config
        self._model_cache: Dict[str, List[ModelInfo]] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)

    def get_available_providers(self) -> Dict[str, bool]:
        """
        Check which providers have valid API keys configured.

        Returns:
            Dictionary mapping provider names to availability status
        """
        return {
            "anthropic": bool(self.config.ANTHROPIC_API_KEY.strip()),
            "gemini": bool(self.config.GEMINI_API_KEY.strip()),
            "lmstudio": True  # Always consider LM Studio available (local)
        }

    def _is_cache_valid(self, provider: str) -> bool:
        """Check if cached model list is still valid."""
        if provider not in self._cache_timestamp:
            return False

        age = datetime.now() - self._cache_timestamp[provider]
        return age < self._cache_ttl

    def _get_anthropic_models(self) -> List[ModelInfo]:
        """
        Get available Anthropic models.

        Note: Anthropic doesn't have a public models API, so we use a curated list.
        """
        if self._is_cache_valid("anthropic"):
            return self._model_cache["anthropic"]

        models = [
            ModelInfo(
                model_id="claude-sonnet-4-20250514",
                display_name="Claude Sonnet 4",
                provider="anthropic",
                supports_tools=True,
                context_window=200000,
                is_default=True
            ),
            ModelInfo(
                model_id="claude-opus-4-20250514",
                display_name="Claude Opus 4",
                provider="anthropic",
                supports_tools=True,
                context_window=200000,
                is_default=False
            ),
            ModelInfo(
                model_id="claude-3-5-sonnet-20241022",
                display_name="Claude 3.5 Sonnet",
                provider="anthropic",
                supports_tools=True,
                context_window=200000,
                is_default=False
            )
        ]

        self._model_cache["anthropic"] = models
        self._cache_timestamp["anthropic"] = datetime.now()
        return models

    def _get_gemini_models(self) -> List[ModelInfo]:
        """
        Get available Gemini models by querying the Gemini API.
        """
        if self._is_cache_valid("gemini"):
            return self._model_cache["gemini"]

        models = []

        try:
            # Configure Gemini with API key
            genai.configure(api_key=self.config.GEMINI_API_KEY)

            # Fetch available models
            for model in genai.list_models():
                # Only include models that support content generation
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace("models/", "")  # Remove 'models/' prefix

                    # Determine if this model supports tools
                    supports_tools = any(
                        'tools' in str(method).lower() or 'function' in str(method).lower()
                        for method in model.supported_generation_methods
                    )

                    models.append(ModelInfo(
                        model_id=model_name,
                        display_name=model.display_name or model_name,
                        provider="gemini",
                        supports_tools=supports_tools,
                        context_window=getattr(model, 'input_token_limit', 32000),
                        is_default=(model_name == self.config.GEMINI_MODEL)
                    ))

            # If we didn't get any models from API, use fallback
            if not models:
                logger.warning("No models returned from Gemini API, using fallback list")
                models = self._get_gemini_fallback_models()

        except Exception as e:
            logger.error(f"Failed to fetch Gemini models: {e}")
            # Use fallback models if API call fails
            models = self._get_gemini_fallback_models()

        self._model_cache["gemini"] = models
        self._cache_timestamp["gemini"] = datetime.now()
        return models

    def _get_gemini_fallback_models(self) -> List[ModelInfo]:
        """Fallback list of Gemini models if API fetch fails."""
        return [
            ModelInfo(
                model_id="gemini-2.0-flash-exp",
                display_name="Gemini 2.0 Flash (Experimental)",
                provider="gemini",
                supports_tools=True,
                context_window=32000,
                is_default=True
            ),
            ModelInfo(
                model_id="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                provider="gemini",
                supports_tools=True,
                context_window=128000,
                is_default=False
            ),
            ModelInfo(
                model_id="gemini-1.5-flash",
                display_name="Gemini 1.5 Flash",
                provider="gemini",
                supports_tools=True,
                context_window=32000,
                is_default=False
            )
        ]

    def _get_lmstudio_models(self) -> List[ModelInfo]:
        """
        Get available LM Studio models by querying the local API.
        """
        if self._is_cache_valid("lmstudio"):
            return self._model_cache["lmstudio"]

        models = []

        try:
            # Create OpenAI client pointing to LM Studio
            client = openai.OpenAI(
                api_key=self.config.LMSTUDIO_API_KEY,
                base_url=self.config.LMSTUDIO_BASE_URL
            )

            # Fetch available models
            response = client.models.list()

            for model in response.data:
                models.append(ModelInfo(
                    model_id=model.id,
                    display_name=model.id,  # LM Studio doesn't provide display names
                    provider="lmstudio",
                    supports_tools=False,  # Assume local models don't support tools by default
                    context_window=4096,  # Default context window
                    is_default=(model.id == self.config.LMSTUDIO_MODEL)
                ))

            # If no models found, use configured default
            if not models:
                logger.warning("No models found from LM Studio, using configured default")
                models = [ModelInfo(
                    model_id=self.config.LMSTUDIO_MODEL,
                    display_name=self.config.LMSTUDIO_MODEL,
                    provider="lmstudio",
                    supports_tools=False,
                    context_window=4096,
                    is_default=True
                )]

        except Exception as e:
            logger.warning(f"Failed to fetch LM Studio models (is LM Studio running?): {e}")
            # Return empty list if LM Studio is not running
            models = []

        self._model_cache["lmstudio"] = models
        self._cache_timestamp["lmstudio"] = datetime.now()
        return models

    def get_provider_models(self, provider_name: str) -> List[ModelInfo]:
        """
        Get available models for a specific provider.

        Args:
            provider_name: Name of the provider ("anthropic", "gemini", or "lmstudio")

        Returns:
            List of available models for the provider
        """
        if provider_name == "anthropic":
            return self._get_anthropic_models()
        elif provider_name == "gemini":
            return self._get_gemini_models()
        elif provider_name == "lmstudio":
            return self._get_lmstudio_models()
        else:
            logger.error(f"Unknown provider: {provider_name}")
            return []

    def get_all_available_models(self) -> List[ProviderInfo]:
        """
        Get all available models from all providers with valid API keys.

        Returns:
            List of ProviderInfo objects with available models
        """
        providers = []
        available_providers = self.get_available_providers()

        # Anthropic
        if available_providers["anthropic"]:
            models = self.get_provider_models("anthropic")
            if models:
                providers.append(ProviderInfo(
                    name="anthropic",
                    display_name="Anthropic Claude",
                    models=models,
                    is_available=True
                ))

        # Gemini
        if available_providers["gemini"]:
            models = self.get_provider_models("gemini")
            if models:
                providers.append(ProviderInfo(
                    name="gemini",
                    display_name="Google Gemini",
                    models=models,
                    is_available=True
                ))

        # LM Studio
        if available_providers["lmstudio"]:
            models = self.get_provider_models("lmstudio")
            if models:  # Only include if we found models (i.e., LM Studio is running)
                providers.append(ProviderInfo(
                    name="lmstudio",
                    display_name="LM Studio (Local)",
                    models=models,
                    is_available=True
                ))

        return providers

    def get_default_model_id(self) -> str:
        """
        Get the default model ID based on provider priority.

        Priority: Anthropic → Gemini → LM Studio

        Returns:
            Model ID of the default model
        """
        available = self.get_available_providers()

        # Priority order: Anthropic -> Gemini -> LM Studio
        if available["anthropic"]:
            return self.config.ANTHROPIC_MODEL
        elif available["gemini"]:
            return self.config.GEMINI_MODEL
        elif available["lmstudio"]:
            return self.config.LMSTUDIO_MODEL
        else:
            # Fallback to Anthropic model even if no key
            return self.config.ANTHROPIC_MODEL

    def create_provider_for_model(self, model_id: str) -> Optional[LLMProvider]:
        """
        Create a provider instance for the specified model.

        Args:
            model_id: ID of the model to create a provider for

        Returns:
            Provider instance or None if model not found
        """
        # Find which provider owns this model
        all_providers = self.get_all_available_models()

        for provider_info in all_providers:
            for model in provider_info.models:
                if model.model_id == model_id:
                    # Found the model, create appropriate provider
                    if provider_info.name == "anthropic":
                        return create_provider(
                            "anthropic",
                            self.config.ANTHROPIC_API_KEY,
                            model_id
                        )
                    elif provider_info.name == "gemini":
                        return create_provider(
                            "gemini",
                            self.config.GEMINI_API_KEY,
                            model_id
                        )
                    elif provider_info.name == "lmstudio":
                        return create_provider(
                            "lmstudio",
                            self.config.LMSTUDIO_API_KEY,
                            model_id,
                            self.config.LMSTUDIO_BASE_URL
                        )

        logger.error(f"Model not found: {model_id}")
        return None

    def find_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Find information about a specific model.

        Args:
            model_id: ID of the model to find

        Returns:
            ModelInfo object or None if not found
        """
        all_providers = self.get_all_available_models()

        for provider_info in all_providers:
            for model in provider_info.models:
                if model.model_id == model_id:
                    return model

        return None

    def clear_cache(self):
        """Clear the model cache to force refresh on next request."""
        self._model_cache.clear()
        self._cache_timestamp.clear()
