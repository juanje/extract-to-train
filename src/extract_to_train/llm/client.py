"""
Ollama client integration with LangChain for educational LLM interactions.

This module provides a clean interface to local Ollama models with educational
logging and error handling designed for learning environments.
"""

import logging
import time
from typing import Any

import httpx
from langchain_ollama import OllamaLLM
from pydantic import BaseModel

from ..models.config import ModelConfig, OllamaConfig

logger = logging.getLogger(__name__)


class ModelResponse(BaseModel):
    """
    Structured response from LLM model interactions.

    Educational Note: Structured responses help track model performance
    and enable quality analysis across different prompts and models.
    """

    content: str
    model_name: str
    temperature: float
    tokens_used: int | None = None
    response_time: float
    success: bool
    error_message: str | None = None


class OllamaClient:
    """
    Educational Ollama client with comprehensive logging and error handling.

    This client provides transparency into LLM interactions, making it ideal
    for learning about prompt engineering and model behavior.
    """

    def __init__(self, config: OllamaConfig, verbose: bool = False):
        """
        Initialize the Ollama client with educational features.

        Args:
            config: Ollama configuration settings
            verbose: Enable detailed logging for educational purposes
        """
        self.config = config
        self.verbose = verbose
        self._models_cache: dict[str, OllamaLLM] = {}

        if self.verbose:
            logger.info("🔧 Initializing Ollama client")
            logger.info(f"📡 Host: {config.host}")
            logger.info(f"⏱️  Timeout: {config.timeout}s")
            logger.info(f"🔄 Max retries: {config.max_retries}")

    def _get_model(self, model_config: ModelConfig) -> OllamaLLM:
        """
        Get or create a LangChain Ollama model instance.

        Args:
            model_config: Configuration for the model

        Returns:
            Configured OllamaLLM instance

        Educational Note: Model instances are cached to avoid recreation
        overhead and maintain consistency across calls.
        """
        cache_key = f"{model_config.name}_{model_config.temperature}"

        if cache_key not in self._models_cache:
            if self.verbose:
                logger.info(f"🆕 Creating new model instance: {model_config.name}")
                logger.info(f"🌡️  Temperature: {model_config.temperature}")
                logger.info(f"📏 Context window: {model_config.context_window}")

            model = OllamaLLM(
                model=model_config.name,
                base_url=self.config.host,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                num_ctx=model_config.context_window,
                num_predict=model_config.max_tokens,
                # Pass timeout through client_kwargs for HTTP client configuration
                client_kwargs={"timeout": self.config.timeout},
            )

            self._models_cache[cache_key] = model

        return self._models_cache[cache_key]

    async def generate_response(
        self, prompt: str, model_config: ModelConfig, max_retries: int | None = None
    ) -> ModelResponse:
        """
        Generate response from LLM with educational logging and error handling.

        Args:
            prompt: Input prompt for the model
            model_config: Model configuration to use
            max_retries: Override default retry count

        Returns:
            ModelResponse with generation results and metadata

        Educational Note: This method demonstrates proper error handling,
        retry logic, and performance monitoring for LLM interactions.
        """
        if max_retries is None:
            max_retries = self.config.max_retries

        if self.verbose:
            logger.info(f"🤖 Generating response with {model_config.name}")
            logger.info(f"📝 Prompt length: {len(prompt)} characters")
            logger.info(f"🎯 Temperature: {model_config.temperature}")

        model = self._get_model(model_config)
        start_time = time.time()
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if self.verbose and attempt > 0:
                    logger.info(f"🔄 Retry attempt {attempt}/{max_retries}")

                # Generate response
                response = await model.ainvoke(prompt)
                response_time = time.time() - start_time

                if self.verbose:
                    logger.info("✅ Response generated successfully")
                    logger.info(f"⏱️  Response time: {response_time:.2f}s")
                    logger.info(f"📊 Response length: {len(response)} characters")

                return ModelResponse(
                    content=response,
                    model_name=model_config.name,
                    temperature=model_config.temperature,
                    response_time=response_time,
                    success=True,
                )

            except Exception as e:
                last_error = e
                response_time = time.time() - start_time

                if self.verbose:
                    logger.warning(f"⚠️  Attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries:
                    delay = self.config.retry_delay * (
                        2**attempt
                    )  # Exponential backoff
                    if self.verbose:
                        logger.info(f"😴 Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ All {max_retries + 1} attempts failed")

        # All attempts failed
        return ModelResponse(
            content="",
            model_name=model_config.name,
            temperature=model_config.temperature,
            response_time=time.time() - start_time,
            success=False,
            error_message=str(last_error) if last_error else "Unknown error",
        )

    def check_model_availability(self, model_name: str) -> bool:
        """
        Check if a specific model is available in Ollama.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is available

        Educational Note: Always verify model availability before processing
        to provide clear error messages and prevent wasted time.
        """
        try:
            response = httpx.get(f"{self.config.host}/api/tags", timeout=5)

            if response.status_code == 200:
                models_data = response.json()
                available_models = [
                    model["name"] for model in models_data.get("models", [])
                ]

                if self.verbose:
                    logger.info(f"📋 Available models: {available_models}")

                return model_name in available_models

            return False

        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Error checking model availability: {e}")
            return False

    def get_available_models(self) -> list[str]:
        """
        Get list of all available models in Ollama.

        Returns:
            List of available model names

        Educational Note: Useful for configuration validation and
        providing users with available options.
        """
        try:
            response = httpx.get(f"{self.config.host}/api/tags", timeout=5)

            if response.status_code == 200:
                models_data = response.json()
                models = [model["name"] for model in models_data.get("models", [])]

                if self.verbose:
                    logger.info(f"📋 Found {len(models)} available models")
                    for model in models:
                        logger.info(f"  • {model}")

                return models

            return []

        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Error fetching model list: {e}")
            return []

    def validate_connectivity(self) -> tuple[bool, str | None]:
        """
        Validate basic Ollama connectivity without checking individual models.

        Returns:
            Tuple of (is_connected, error_message)

        Educational Note: Simplified connectivity check that trusts model
        configuration and only verifies Ollama is accessible.
        """
        try:
            response = httpx.get(f"{self.config.host}/api/tags", timeout=5)
            if response.status_code == 200:
                if self.verbose:
                    logger.info("✅ Ollama connectivity verified")
                return True, None
            else:
                return False, f"Ollama returned status {response.status_code}"
        except Exception as e:
            return False, f"Ollama connection failed: {str(e)}"

    def validate_configuration(
        self, models: dict[str, ModelConfig]
    ) -> tuple[bool, list[str]]:
        """
        Validate that all configured models are available.

        Args:
            models: Dictionary of model configurations to validate

        Returns:
            Tuple of (is_valid, list_of_errors)

        Educational Note: Configuration validation prevents runtime failures
        and provides clear guidance for setup issues.
        """
        errors = []

        if self.verbose:
            logger.info("🔍 Validating model configuration...")

        # Check Ollama connectivity and get models in a single call
        try:
            response = httpx.get(f"{self.config.host}/api/tags", timeout=5)
            if response.status_code != 200:
                errors.append(f"Cannot connect to Ollama at {self.config.host}")
                return False, errors

            # Get available models in a single call (no additional requests)
            models_data = response.json()
            available_models = [
                model["name"] for model in models_data.get("models", [])
            ]

            if self.verbose:
                logger.info(f"📋 Found {len(available_models)} available models")
                for model in available_models:
                    logger.info(f"  • {model}")

            # Check individual models against the single list (no additional API calls)
            for model_type, model_config in models.items():
                if model_config.name not in available_models:
                    errors.append(
                        f"Model '{model_config.name}' for {model_type} not found. "
                        f"Available models: {', '.join(available_models)}"
                    )
                else:
                    if self.verbose:
                        logger.info(f"✅ Model {model_type}: {model_config.name}")

        except Exception as e:
            errors.append(f"Ollama connection failed: {str(e)}")
            return False, errors

        is_valid = len(errors) == 0

        if self.verbose:
            if is_valid:
                logger.info("✅ All models validated successfully")
            else:
                logger.error(f"❌ Validation failed with {len(errors)} errors")
                for error in errors:
                    logger.error(f"  • {error}")

        return is_valid, errors

    async def test_model_response(self, model_config: ModelConfig) -> dict[str, Any]:
        """
        Test a model with a simple prompt to verify functionality.

        Args:
            model_config: Model configuration to test

        Returns:
            Dictionary with test results

        Educational Note: Testing models with simple prompts helps verify
        setup and provides insight into model behavior and performance.
        """
        test_prompt = "Please respond with exactly: 'Test successful. Model is working correctly.'"

        if self.verbose:
            logger.info(f"🧪 Testing model: {model_config.name}")

        time.time()
        response = await self.generate_response(
            test_prompt, model_config, max_retries=1
        )

        expected_response = "Test successful. Model is working correctly."
        is_correct = expected_response.lower() in response.content.lower()

        test_results = {
            "model_name": model_config.name,
            "success": response.success,
            "response_time": response.response_time,
            "correct_response": is_correct,
            "response_content": response.content[:100] + "..."
            if len(response.content) > 100
            else response.content,
            "error": response.error_message,
        }

        if self.verbose:
            if response.success:
                logger.info(f"✅ Test completed in {response.response_time:.2f}s")
                logger.info(f"🎯 Response accuracy: {'✅' if is_correct else '❌'}")
            else:
                logger.error(f"❌ Test failed: {response.error_message}")

        return test_results

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model to inspect

        Returns:
            Model information dictionary or None if not found

        Educational Note: Model information helps understand capabilities
        and limitations, which is crucial for optimal configuration.
        """
        try:
            response = httpx.post(
                f"{self.config.host}/api/show", json={"name": model_name}, timeout=10
            )

            if response.status_code == 200:
                model_info: dict[str, Any] = response.json()

                if self.verbose:
                    logger.info(f"ℹ️  Model info for {model_name}:")
                    logger.info(f"  📏 Size: {model_info.get('size', 'Unknown')}")
                    logger.info(
                        f"  🏗️  Family: {model_info.get('details', {}).get('family', 'Unknown')}"
                    )
                    logger.info(
                        f"  📊 Parameters: {model_info.get('details', {}).get('parameter_size', 'Unknown')}"
                    )

                return model_info

            return None

        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Error getting model info: {e}")
            return None
