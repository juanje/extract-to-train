"""
Configuration models for the extract-to-train application.

This module defines configuration structures with educational explanations
for optimal learning and transparency.
"""

from pathlib import Path

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ModelConfig(BaseModel):
    """
    Configuration for individual LLM models.

    Educational Note: These settings directly impact the quality and style
    of generated content. Understanding them is crucial for optimization.
    """

    name: str = Field(description="Ollama model name (e.g., 'llama3.1:8b')")
    temperature: float = Field(
        ge=0.0,
        le=2.0,
        description="Controls randomness: 0.0=deterministic, 1.0=creative",
    )
    context_window: int = Field(gt=0, description="Maximum context length in tokens")
    max_tokens: int | None = Field(
        default=None, description="Maximum tokens to generate (None for model default)"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter for diversity control",
    )
    explanation: str = Field(
        description="Educational explanation of why this configuration works"
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature_range(cls, v: float) -> float:
        """Ensure temperature is in reasonable range for our use cases."""
        if (v < 0.0 or v > 1.0) and v > 1.5:
            raise ValueError(
                "Temperature > 1.5 may produce incoherent outputs for our tasks"
            )
        return v


class ProcessingConfig(BaseModel):
    """
    Configuration for document processing and chunking.

    Educational Note: Proper chunking is crucial for maintaining context
    while staying within model limits.
    """

    chunk_size: int = Field(
        default=512, gt=100, description="Target size for text chunks in characters"
    )
    chunk_overlap: int = Field(
        default=100, ge=0, description="Overlap between chunks to preserve context"
    )
    max_pairs_per_chunk: int = Field(
        default=5, gt=0, description="Maximum Q&A pairs to generate per chunk"
    )
    min_chunk_size: int = Field(
        default=100, gt=0, description="Minimum chunk size to process"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info: ValidationInfo) -> int:
        """Ensure overlap doesn't exceed chunk size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v


class GenerationConfig(BaseModel):
    """
    Configuration for Q&A generation process.

    Educational Note: These settings control the diversity and quality
    of generated training examples.
    """

    question_types: list[str] = Field(
        default=["factual", "inferential", "comparative", "analytical"],
        description="Types of questions to generate for diversity",
    )
    difficulty_levels: list[str] = Field(
        default=["easy", "medium", "hard"],
        description="Difficulty levels to ensure varied complexity",
    )
    min_answer_length: int = Field(
        default=50, gt=0, description="Minimum answer length in characters"
    )
    max_answer_length: int = Field(
        default=500, gt=0, description="Maximum answer length in characters"
    )
    require_context_adherence: bool = Field(
        default=True, description="Ensure answers are based only on provided context"
    )

    @field_validator("max_answer_length")
    @classmethod
    def validate_answer_lengths(cls, v: int, info: ValidationInfo) -> int:
        """Ensure max length is greater than min length."""
        min_length = info.data.get("min_answer_length", 50)
        if v <= min_length:
            raise ValueError("Max answer length must be greater than min length")
        return v


class ValidationConfig(BaseModel):
    """
    Configuration for dataset validation and quality control.

    Educational Note: Rigorous validation ensures high-quality training data,
    which is essential for effective fine-tuning.
    """

    min_confidence_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for inclusion",
    )
    required_scores: dict[str, int] = Field(
        default={"accuracy": 7, "completeness": 6, "clarity": 6, "training_value": 7},
        description="Minimum scores for each validation criterion",
    )
    enable_auto_correction: bool = Field(
        default=False, description="Attempt to automatically correct minor issues"
    )
    max_retries: int = Field(
        default=2, ge=0, description="Maximum retries for failed validations"
    )


class OllamaConfig(BaseModel):
    """
    Configuration for Ollama service connection.

    Educational Note: These settings control how we connect to the local
    Ollama instance serving our models.
    """

    host: str = Field(
        default="http://localhost:11434", description="Ollama service URL"
    )
    timeout: int = Field(default=60, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(
        default=3, ge=0, description="Maximum number of retry attempts"
    )
    retry_delay: float = Field(
        default=1.0, ge=0.0, description="Delay between retries in seconds"
    )


class AppConfig(BaseModel):
    """
    Main application configuration.

    This brings together all configuration aspects and provides educational
    insights into optimal settings for different scenarios.
    """

    # Model configurations with educational explanations
    models: dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "extraction": ModelConfig(
                name="llama3.1:8B",
                temperature=0.3,
                context_window=4096,
                explanation=(
                    "Latest Llama model with improved instruction following and "
                    "balanced creativity for diverse question generation. "
                    "Temperature 0.3 provides good variety while maintaining consistency."
                ),
            ),
            "validation": ModelConfig(
                name="deepseek-r1:8B",
                temperature=0.1,
                context_window=4096,
                explanation=(
                    "Superior reasoning model designed for analysis and critique. "
                    "Low temperature (0.1) ensures consistent, objective evaluation. "
                    "Separate model provides unbiased validation independent of generation."
                ),
            ),
        }
    )

    # Processing settings
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    # Generation settings
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    # Validation settings
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    # Ollama connection settings
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)

    # Output settings
    output_dir: Path = Field(
        default=Path("output"), description="Directory for generated datasets"
    )
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    @classmethod
    def get_educational_config(cls) -> "AppConfig":
        """
        Get the recommended educational configuration.

        Returns:
            AppConfig instance optimized for learning and demonstration

        Educational Note: This configuration balances quality, speed, and
        educational value for optimal learning experience.
        """
        return cls(
            models={
                "extraction": ModelConfig(
                    name="llama3.1:8B",
                    temperature=0.3,
                    context_window=4096,
                    explanation=(
                        "🎯 EXTRACTION MODEL EXPLANATION:\n"
                        "• Llama 3.1:8B chosen for superior instruction following\n"
                        "• Temperature 0.3 balances creativity with consistency\n"
                        "• 4K context window handles typical document chunks\n"
                        "• This model excels at generating diverse, contextual questions"
                    ),
                ),
                "validation": ModelConfig(
                    name="deepseek-r1:8B",
                    temperature=0.1,
                    context_window=4096,
                    explanation=(
                        "🔍 VALIDATION MODEL EXPLANATION:\n"
                        "• DeepSeek-R1:8B specialized for reasoning and analysis\n"
                        "• Temperature 0.1 ensures consistent, objective evaluation\n"
                        "• Separate model prevents bias from generation process\n"
                        "• Superior at identifying quality issues and improvements"
                    ),
                ),
            },
            processing=ProcessingConfig(
                chunk_size=512,  # Optimal balance for context preservation and model efficiency
                chunk_overlap=100,  # ~20% overlap maintains continuity
                max_pairs_per_chunk=3,  # Conservative for quality over quantity
            ),
            generation=GenerationConfig(
                min_answer_length=75,  # Ensure substantial answers
                max_answer_length=400,  # Keep answers focused
                require_context_adherence=True,  # Prevent hallucinations
            ),
            validation=ValidationConfig(
                min_confidence_score=0.75,  # Higher bar for quality
                required_scores={
                    "accuracy": 8,  # High accuracy requirement
                    "completeness": 7,
                    "clarity": 7,
                    "training_value": 8,
                },
            ),
        )

    @classmethod
    def get_speed_optimized_config(cls) -> "AppConfig":
        """
        Get configuration optimized for speed over quality.

        Returns:
            AppConfig instance for faster processing

        Educational Note: Sometimes speed is prioritized for experimentation
        or when processing large documents.
        """
        config = cls.get_educational_config()

        # Use faster model for both tasks
        config.models["extraction"].name = "mistral:7B"
        config.models["validation"].name = "mistral:7B"

        # Larger chunks for faster processing
        config.processing.chunk_size = 1500
        config.processing.max_pairs_per_chunk = 5

        # Lower quality requirements
        config.validation.min_confidence_score = 0.6
        config.validation.required_scores = {
            "accuracy": 6,
            "completeness": 5,
            "clarity": 5,
            "training_value": 6,
        }

        return config

    def get_model_explanation(self, model_type: str) -> str:
        """
        Get educational explanation for a specific model configuration.

        Args:
            model_type: Type of model ('extraction' or 'validation')

        Returns:
            Detailed explanation of the model choice and settings
        """
        if model_type not in self.models:
            return f"Model type '{model_type}' not found in configuration"

        model = self.models[model_type]
        return (
            f"Model: {model.name}\n"
            f"Temperature: {model.temperature}\n"
            f"Context Window: {model.context_window:,} tokens\n"
            f"Top-p: {model.top_p}\n\n"
            f"Explanation:\n{model.explanation}"
        )

    def validate_ollama_connection(self) -> bool:
        """
        Validate that Ollama is accessible with current configuration.

        Returns:
            True if connection is successful

        Educational Note: Always verify your local setup before processing
        to avoid wasted time on configuration issues.
        """
        import httpx

        try:
            response = httpx.get(f"{self.ollama.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
