"""
Data models for dataset creation and validation.

This module defines the core data structures used throughout the extract-to-train
project, focusing on educational clarity and type safety.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DatasetFormat(str, Enum):
    """
    Supported dataset formats for fine-tuning.

    Each format is optimized for specific fine-tuning frameworks:
    - ALPACA: Universal format supported by Axolotl, Unsloth, HF+PEFT
    - SHAREGPT: Conversational format for Axolotl and Unsloth
    - OPENAI: API-compatible format for HuggingFace Transformers + PEFT
    """

    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    OPENAI = "openai"


class QuestionType(str, Enum):
    """
    Types of questions that can be generated for diverse training data.

    Educational Note: Different question types help the model learn various
    reasoning patterns and improve generalization.
    """

    FACTUAL = "factual"  # Direct facts from the document
    INFERENTIAL = "inferential"  # Requires reading between the lines
    COMPARATIVE = "comparative"  # Comparing concepts or ideas
    ANALYTICAL = "analytical"  # Deeper analysis and reasoning


class DifficultyLevel(str, Enum):
    """
    Difficulty levels for questions to create diverse training scenarios.

    Educational Note: Varied difficulty helps the model handle different
    complexity levels and improves robustness.
    """

    EASY = "easy"  # Straightforward, factual questions
    MEDIUM = "medium"  # Requires some reasoning or synthesis
    HARD = "hard"  # Complex analysis or multi-step reasoning


class QAPair(BaseModel):
    """
    A single question-answer pair with metadata for training.

    This is the core data structure representing one training example.
    All metadata helps with dataset analysis and quality control.
    """

    id: str = Field(description="Unique identifier for this Q&A pair")
    question: str = Field(description="The generated question")
    answer: str = Field(description="The corresponding answer")
    context: str = Field(description="Source context from the PDF")
    difficulty: DifficultyLevel = Field(description="Question difficulty level")
    question_type: QuestionType = Field(description="Type of question generated")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="AI confidence in the quality of this pair"
    )
    source_page: int | None = Field(
        default=None, description="Page number from source PDF"
    )

    @field_validator("question", "answer")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        """Ensure questions and answers have reasonable length."""
        if len(v.strip()) < 10:
            raise ValueError("Text must be at least 10 characters long")
        return v.strip()

    def to_alpaca(self, include_context: bool = True) -> dict[str, str]:
        """
        Convert to Alpaca format for training.

        Args:
            include_context: Whether to include source context in input field

        Returns:
            Dictionary in Alpaca format

        Educational Note: Alpaca format uses instruction/input/output structure
        which is ideal for instruction-following fine-tuning.
        """
        return {
            "instruction": self.question,
            "input": self.context if include_context else "",
            "output": self.answer,
        }

    def to_sharegpt(self) -> dict[str, list[dict[str, str]]]:
        """
        Convert to ShareGPT conversational format.

        Returns:
            Dictionary in ShareGPT format

        Educational Note: ShareGPT format simulates chat conversations,
        ideal for training conversational AI models.
        """
        return {
            "conversations": [
                {"from": "human", "value": self.question},
                {"from": "gpt", "value": self.answer},
            ]
        }

    def to_openai(
        self, system_prompt: str | None = None
    ) -> dict[str, list[dict[str, str]]]:
        """
        Convert to OpenAI API format.

        Args:
            system_prompt: Optional system message to include

        Returns:
            Dictionary in OpenAI format

        Educational Note: OpenAI format includes system/user/assistant roles
        and is compatible with many modern fine-tuning pipelines.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(
            [
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.answer},
            ]
        )

        return {"messages": messages}


class ValidationResult(BaseModel):
    """
    Results from validating a Q&A pair.

    Educational Note: Comprehensive validation helps ensure high-quality
    training data, which is crucial for effective fine-tuning.
    """

    is_valid: bool = Field(description="Overall validation result")
    accuracy_score: int = Field(ge=0, le=10, description="Factual accuracy (0-10)")
    completeness_score: int = Field(
        ge=0, le=10, description="Answer completeness (0-10)"
    )
    clarity_score: int = Field(
        ge=0, le=10, description="Clarity and readability (0-10)"
    )
    training_value_score: int = Field(
        ge=0, le=10, description="Value for training purposes (0-10)"
    )
    issues: list[str] = Field(
        default_factory=list, description="Specific issues identified"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score as average of all metrics."""
        scores = [
            self.accuracy_score,
            self.completeness_score,
            self.clarity_score,
            self.training_value_score,
        ]
        return sum(scores) / len(scores)


class DatasetStats(BaseModel):
    """
    Statistical analysis of the generated dataset.

    Educational Note: These statistics help understand dataset quality
    and diversity, which are key factors for successful fine-tuning.
    """

    total_pairs: int = Field(description="Total number of Q&A pairs")
    avg_question_length: int = Field(
        description="Average question length in characters"
    )
    avg_answer_length: int = Field(description="Average answer length in characters")
    difficulty_distribution: dict[str, int] = Field(
        description="Count of pairs by difficulty level"
    )
    question_type_distribution: dict[str, int] = Field(
        description="Count of pairs by question type"
    )
    pages_covered: int = Field(description="Number of PDF pages represented")
    estimated_tokens: int = Field(
        description="Estimated total tokens for cost calculation"
    )
    avg_confidence_score: float = Field(
        ge=0.0, le=1.0, description="Average confidence score across all pairs"
    )
    validation_pass_rate: float = Field(
        ge=0.0, le=1.0, description="Percentage of pairs that passed validation"
    )

    def display_summary(self) -> str:
        """
        Create a human-readable summary of dataset statistics.

        Returns:
            Formatted string with key statistics

        Educational Note: Clear statistics help users understand their
        dataset quality and make informed decisions about fine-tuning.
        """
        return f"""
📊 Dataset Statistics Summary:
├── Total Q&A pairs: {self.total_pairs:,}
├── Average question length: {self.avg_question_length} chars
├── Average answer length: {self.avg_answer_length} chars
├── Estimated tokens: ~{self.estimated_tokens:,} (for cost estimation)
├── Pages covered: {self.pages_covered}
├── Average confidence: {self.avg_confidence_score:.2f}
├── Validation pass rate: {self.validation_pass_rate:.1%}
├── Difficulty distribution: {dict(self.difficulty_distribution)}
└── Question type distribution: {dict(self.question_type_distribution)}

💡 Fine-tuning Notes:
• Total tokens help estimate training time and cost
• High validation pass rate ({self.validation_pass_rate:.1%}) indicates quality data
• Balanced distributions ensure diverse training scenarios
        """.strip()


class Dataset(BaseModel):
    """
    Complete dataset with Q&A pairs, validation results, and metadata.

    This is the main container for all generated training data and associated
    information needed for fine-tuning preparation.
    """

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about source document and generation process",
    )
    qa_pairs: list[QAPair] = Field(description="Generated Q&A pairs")
    stats: DatasetStats = Field(description="Dataset statistics and analysis")

    def export_format(
        self, format_type: DatasetFormat, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Export dataset in specified format for fine-tuning tools.

        Args:
            format_type: Target format (alpaca, sharegpt, openai)
            **kwargs: Format-specific options

        Returns:
            List of dictionaries ready for JSONL export

        Educational Note: Different formats are optimized for different
        fine-tuning frameworks and use cases.
        """
        if format_type == DatasetFormat.ALPACA:
            include_context = kwargs.get("include_context", True)
            return [
                pair.to_alpaca(include_context=include_context)
                for pair in self.qa_pairs
            ]

        elif format_type == DatasetFormat.SHAREGPT:
            return [pair.to_sharegpt() for pair in self.qa_pairs]

        elif format_type == DatasetFormat.OPENAI:
            system_prompt = kwargs.get("system_prompt")
            return [
                pair.to_openai(system_prompt=system_prompt) for pair in self.qa_pairs
            ]

        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def filter_by_quality(self, min_confidence: float = 0.7) -> "Dataset":
        """
        Create a new dataset with only high-quality pairs.

        Args:
            min_confidence: Minimum confidence score threshold

        Returns:
            New Dataset instance with filtered pairs

        Educational Note: Quality filtering ensures only the best examples
        are used for training, improving model performance.
        """
        filtered_pairs = [
            pair for pair in self.qa_pairs if pair.confidence_score >= min_confidence
        ]

        # Recalculate statistics for filtered dataset
        new_stats = self._calculate_stats(filtered_pairs)

        return Dataset(
            metadata={
                **self.metadata,
                "filtered": True,
                "min_confidence": min_confidence,
            },
            qa_pairs=filtered_pairs,
            stats=new_stats,
        )

    def _calculate_stats(self, pairs: list[QAPair]) -> DatasetStats:
        """Calculate statistics for a list of Q&A pairs."""
        if not pairs:
            return DatasetStats(
                total_pairs=0,
                avg_question_length=0,
                avg_answer_length=0,
                difficulty_distribution={},
                question_type_distribution={},
                pages_covered=0,
                estimated_tokens=0,
                avg_confidence_score=0.0,
                validation_pass_rate=0.0,
            )

        # Calculate distributions
        difficulty_dist: dict[str, int] = {}
        type_dist: dict[str, int] = {}
        pages = set()

        total_question_chars = 0
        total_answer_chars = 0
        total_confidence = 0.0

        for pair in pairs:
            # Length calculations
            total_question_chars += len(pair.question)
            total_answer_chars += len(pair.answer)
            total_confidence += pair.confidence_score

            # Distributions
            diff_key = str(pair.difficulty)
            type_key = str(pair.question_type)
            difficulty_dist[diff_key] = difficulty_dist.get(diff_key, 0) + 1
            type_dist[type_key] = type_dist.get(type_key, 0) + 1

            # Pages
            if pair.source_page is not None:
                pages.add(pair.source_page)

        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        total_chars = total_question_chars + total_answer_chars
        estimated_tokens = total_chars // 4

        return DatasetStats(
            total_pairs=len(pairs),
            avg_question_length=total_question_chars // len(pairs),
            avg_answer_length=total_answer_chars // len(pairs),
            difficulty_distribution=difficulty_dist,
            question_type_distribution=type_dist,
            pages_covered=len(pages),
            estimated_tokens=estimated_tokens,
            avg_confidence_score=total_confidence / len(pairs),
            validation_pass_rate=1.0,  # Will be updated during validation
        )
