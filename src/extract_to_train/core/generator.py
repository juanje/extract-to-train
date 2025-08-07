"""
Q&A generation module for creating diverse training datasets.

This module orchestrates the generation of question-answer pairs from extracted
PDF content using LLMs with educational transparency and quality controls.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..llm.client import OllamaClient
from ..llm.prompts import PromptTemplates
from ..models.config import AppConfig
from ..models.dataset import DifficultyLevel, QAPair, QuestionType
from .extractor import ExtractedChunk

logger = logging.getLogger(__name__)


class GenerationStats(BaseModel):
    """
    Statistics tracking for the generation process.

    Educational Note: Tracking generation statistics helps identify
    performance bottlenecks and quality patterns.
    """

    total_chunks_processed: int = 0
    total_pairs_generated: int = 0
    total_pairs_rejected: int = 0
    avg_generation_time: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0


class QAGenerator:
    """
    Educational Q&A generator with comprehensive quality controls.

    This generator creates diverse, high-quality question-answer pairs
    optimized for fine-tuning while providing educational insights
    into the generation process.
    """

    def __init__(
        self, ollama_client: OllamaClient, config: AppConfig, verbose: bool = False
    ):
        """
        Initialize the Q&A generator.

        Args:
            ollama_client: Client for LLM interactions
            config: Application configuration
            verbose: Enable detailed educational logging
        """
        self.client = ollama_client
        self.config = config
        self.verbose = verbose
        self.prompt_templates = PromptTemplates()

        if self.verbose:
            logger.info("🎯 Q&A Generator initialized")
            logger.info(f"🤖 Generation model: {config.models['extraction'].name}")
            logger.info(f"🌡️  Temperature: {config.models['extraction'].temperature}")
            logger.info(f"📊 Target types: {config.generation.question_types}")
            logger.info(f"📈 Difficulty levels: {config.generation.difficulty_levels}")

    async def generate_qa_pairs(
        self,
        chunks: list[ExtractedChunk],
        max_pairs_total: int | None = None,
        document_language: str | None = None,
        progress_file: str | None = None,
    ) -> tuple[list[QAPair], GenerationStats]:
        """
        Generate Q&A pairs from extracted document chunks.

        Args:
            chunks: List of document chunks to process
            max_pairs_total: Maximum total pairs to generate
            document_language: Language of the document (e.g., 'es', 'en', 'fr')
            progress_file: Optional file to save progressive Q&A pairs during generation

        Returns:
            Tuple of (generated_pairs, generation_statistics)

        Educational Note: This method demonstrates batch processing with
        quality controls and performance monitoring. Language-aware generation
        ensures Q&A pairs are created in the appropriate language. Progress
        saving allows recovery from interruptions in large document processing.
        """
        if self.verbose:
            logger.info(f"🚀 Starting Q&A generation from {len(chunks)} chunks")
            if max_pairs_total:
                logger.info(f"🎯 Target: {max_pairs_total} total pairs")
            if progress_file:
                logger.info(f"📝 Progress saving to: {progress_file}")

        generated_pairs: list[QAPair] = []
        stats = GenerationStats()
        total_generation_time = 0.0

        # Initialize progress file if specified
        if progress_file:
            # Clear the progress file at the start
            Path(progress_file).write_text("", encoding="utf-8")

        for i, chunk in enumerate(chunks, 1):
            if self.verbose:
                logger.info(
                    f"📝 Processing chunk {i}/{len(chunks)} (Page {chunk.page_number})"
                )
                logger.info(
                    f"📏 Chunk size: {chunk.char_count} chars, {chunk.word_count} words"
                )

            # Check if we've reached the maximum
            if max_pairs_total and len(generated_pairs) >= max_pairs_total:
                if self.verbose:
                    logger.info(f"🎯 Reached target of {max_pairs_total} pairs")
                break

            # Skip chunks that are too small
            if chunk.char_count < self.config.processing.min_chunk_size:
                if self.verbose:
                    logger.info(f"⏭️  Skipping small chunk ({chunk.char_count} chars)")
                continue

            try:
                # Calculate pairs to generate for this chunk
                remaining_pairs = (
                    max_pairs_total - len(generated_pairs)
                    if max_pairs_total
                    else self.config.processing.max_pairs_per_chunk
                )
                pairs_for_chunk = min(
                    self.config.processing.max_pairs_per_chunk, remaining_pairs
                )

                if pairs_for_chunk <= 0:
                    break

                # Generate pairs for this chunk
                chunk_pairs, generation_time = await self._generate_pairs_for_chunk(
                    chunk, pairs_for_chunk, document_language
                )

                # Filter and validate pairs
                valid_pairs = self._filter_generated_pairs(chunk_pairs, chunk)

                generated_pairs.extend(valid_pairs)
                stats.total_chunks_processed += 1
                stats.total_pairs_generated += len(valid_pairs)
                stats.total_pairs_rejected += len(chunk_pairs) - len(valid_pairs)
                total_generation_time += generation_time

                # Save valid pairs progressively if progress file is specified
                if progress_file and valid_pairs:
                    self._save_progress(valid_pairs, progress_file)

                if self.verbose:
                    logger.info(
                        f"✅ Generated {len(valid_pairs)} valid pairs from chunk"
                    )
                    logger.info(f"⏱️  Generation time: {generation_time:.2f}s")

            except Exception as e:
                stats.error_count += 1
                logger.error(f"❌ Error processing chunk {i}: {str(e)}")

                if self.verbose:
                    logger.error(f"🐛 Chunk content preview: {chunk.content[:100]}...")

        # Calculate final statistics
        stats.avg_generation_time = (
            total_generation_time / stats.total_chunks_processed
            if stats.total_chunks_processed > 0
            else 0.0
        )
        stats.success_rate = (
            stats.total_chunks_processed / len(chunks) if chunks else 0.0
        )

        if self.verbose:
            logger.info("🎉 Generation completed!")
            logger.info("📊 Final statistics:")
            logger.info(f"  • Total pairs generated: {stats.total_pairs_generated}")
            logger.info(
                f"  • Chunks processed: {stats.total_chunks_processed}/{len(chunks)}"
            )
            logger.info(f"  • Success rate: {stats.success_rate:.1%}")
            logger.info(f"  • Average time per chunk: {stats.avg_generation_time:.2f}s")
            logger.info(
                f"  • Rejection rate: {stats.total_pairs_rejected / (stats.total_pairs_generated + stats.total_pairs_rejected):.1%}"
            )

        return generated_pairs, stats

    async def _generate_pairs_for_chunk(
        self,
        chunk: ExtractedChunk,
        num_pairs: int,
        document_language: str | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Generate Q&A pairs for a single chunk.

        Args:
            chunk: Document chunk to process
            num_pairs: Number of pairs to generate

        Returns:
            Tuple of (raw_pairs, generation_time)

        Educational Note: This method shows how to handle LLM responses
        and parse structured output with error recovery.
        """
        # Prepare the prompt with language-specific instructions
        qa_prompt = self.prompt_templates.get_qa_generation_prompt()

        # Add language instruction if specified
        language_instruction = ""
        if document_language and document_language != "en":
            language_names = {
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ja": "Japanese",
                "ko": "Korean",
                "zh": "Chinese",
            }
            language_name = language_names.get(
                document_language, document_language.upper()
            )
            location_refs = {
                "es": "página, capítulo, sección",
                "fr": "page, chapitre, section",
                "de": "Seite, Kapitel, Abschnitt",
                "it": "pagina, capitolo, sezione",
                "pt": "página, capítulo, seção",
                "ru": "страница, глава, раздел",
                "ja": "ページ、章、セクション",
                "ko": "페이지, 장, 섹션",
                "zh": "页面、章节、部分",
            }
            location_ref = location_refs.get(
                document_language, "page, chapter, section"
            )
            language_instruction = f"\n\nIMPORTANT: Generate all questions and answers in {language_name} to match the document language. NEVER mention {location_ref} or any location references."

        formatted_prompt = (
            qa_prompt.format(content=chunk.content, num_questions=num_pairs)
            + language_instruction
        )

        if self.verbose:
            logger.info(f"🎨 Using prompt with {len(formatted_prompt)} characters")
            logger.info(f"🎯 Requesting {num_pairs} Q&A pairs")

        # Generate response
        response = await self.client.generate_response(
            formatted_prompt, self.config.models["extraction"]
        )

        if not response.success:
            raise RuntimeError(f"Generation failed: {response.error_message}")

        # Parse JSON response
        try:
            pairs_data = self._parse_json_response(response.content)

            if self.verbose:
                logger.info(f"📝 Parsed {len(pairs_data)} pairs from response")

            return pairs_data, response.response_time

        except Exception as e:
            logger.error(f"❌ Failed to parse response: {str(e)}")
            logger.error(f"🔍 Response preview: {response.content[:200]}...")
            raise

    def _parse_json_response(self, response_content: str) -> list[dict[str, Any]]:
        """
        Parse JSON response from LLM with error recovery.

        Args:
            response_content: Raw response from LLM

        Returns:
            List of parsed Q&A pair dictionaries

        Educational Note: LLM responses can be inconsistent, so robust
        parsing with error recovery is essential for production use.
        """
        # Clean the response content
        content = response_content.strip()

        # Try to find JSON array in the response
        start_markers = ["[", "```json\n[", "```\n["]
        end_markers = ["]", "]\n```", "]\n```"]

        json_content = None

        for start_marker, end_marker in zip(start_markers, end_markers, strict=False):
            start_idx = content.find(start_marker)
            if start_idx != -1:
                end_idx = content.find(end_marker, start_idx)
                if end_idx != -1:
                    json_content = content[
                        start_idx : end_idx + len(end_marker.split("\n")[0])
                    ]
                    break

        if json_content is None:
            # Fallback: try to parse the entire content
            json_content = content

        try:
            # Parse JSON
            pairs_data: Any = json.loads(json_content)

            # Ensure it's a list
            if isinstance(pairs_data, dict):
                pairs_data = [pairs_data]
            elif not isinstance(pairs_data, list):
                raise ValueError("Response is not a list or dictionary")

            return pairs_data  # type: ignore

        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed_content = self._fix_common_json_issues(json_content)

            try:
                fixed_pairs_data: Any = json.loads(fixed_content)
                if isinstance(fixed_pairs_data, dict):
                    fixed_pairs_data = [fixed_pairs_data]
                return fixed_pairs_data  # type: ignore
            except json.JSONDecodeError as json_err:
                raise ValueError(
                    f"Could not parse JSON response: {str(json_err)}"
                ) from json_err

    def _fix_common_json_issues(self, json_content: str) -> str:
        """
        Fix common JSON formatting issues in LLM responses.

        Args:
            json_content: Potentially malformed JSON

        Returns:
            Fixed JSON string
        """
        # Remove common formatting issues
        fixed = json_content

        # Remove markdown code blocks
        fixed = fixed.replace("```json", "").replace("```", "")

        # Fix missing commas between objects
        fixed = fixed.replace("}\n{", "},\n{")
        fixed = fixed.replace("} {", "}, {")

        # Ensure array format
        if not fixed.strip().startswith("["):
            fixed = "[" + fixed
        if not fixed.strip().endswith("]"):
            fixed = fixed + "]"

        return fixed

    def _filter_generated_pairs(
        self, raw_pairs: list[dict[str, Any]], source_chunk: ExtractedChunk
    ) -> list[QAPair]:
        """
        Filter and convert raw pairs to validated QAPair objects.

        Args:
            raw_pairs: Raw dictionary pairs from LLM
            source_chunk: Source chunk for context

        Returns:
            List of validated QAPair objects

        Educational Note: Validation ensures training data quality
        and filters out problematic examples that could hurt performance.
        """
        valid_pairs = []

        for i, pair_data in enumerate(raw_pairs):
            try:
                # Validate required fields
                if not all(key in pair_data for key in ["question", "answer"]):
                    if self.verbose:
                        logger.warning(f"⚠️  Pair {i + 1}: Missing required fields")
                    continue

                question = pair_data["question"].strip()
                answer = pair_data["answer"].strip()

                # Basic quality checks
                if (
                    len(question) < 10
                    or len(answer) < self.config.generation.min_answer_length
                ):
                    if self.verbose:
                        logger.warning(
                            f"⚠️  Pair {i + 1}: Too short (Q:{len(question)}, A:{len(answer)})"
                        )
                    continue

                if len(answer) > self.config.generation.max_answer_length:
                    if self.verbose:
                        logger.warning(
                            f"⚠️  Pair {i + 1}: Answer too long ({len(answer)} chars)"
                        )
                    # Truncate rather than reject
                    answer = (
                        answer[: self.config.generation.max_answer_length].rsplit(
                            " ", 1
                        )[0]
                        + "..."
                    )

                # Parse metadata with defaults
                question_type = pair_data.get("question_type", "factual")
                difficulty = pair_data.get("difficulty", "medium")
                confidence = float(pair_data.get("confidence", 0.8))

                # Validate enums
                try:
                    QuestionType(question_type)
                    DifficultyLevel(difficulty)
                except ValueError:
                    question_type = "factual"
                    difficulty = "medium"

                # Create QAPair object
                qa_pair = QAPair(
                    id=str(uuid.uuid4()),
                    question=question,
                    answer=answer,
                    context=source_chunk.content,
                    difficulty=DifficultyLevel(difficulty),
                    question_type=QuestionType(question_type),
                    confidence_score=max(0.0, min(1.0, confidence)),
                    source_page=source_chunk.page_number,
                )

                valid_pairs.append(qa_pair)

                if self.verbose:
                    logger.info(
                        f"✅ Pair {i + 1}: Valid ({question_type}, {difficulty})"
                    )

            except Exception as e:
                if self.verbose:
                    logger.warning(f"⚠️  Pair {i + 1}: Validation failed - {str(e)}")
                continue

        return valid_pairs

    def analyze_generation_quality(self, pairs: list[QAPair]) -> dict[str, Any]:
        """
        Analyze the quality and diversity of generated pairs.

        Args:
            pairs: List of generated QAPair objects

        Returns:
            Dictionary with quality analysis

        Educational Note: Quality analysis helps identify areas for
        improvement in prompt engineering and model selection.
        """
        if not pairs:
            return {"error": "No pairs to analyze"}

        # Analyze distributions
        type_dist: dict[str, int] = {}
        difficulty_dist: dict[str, int] = {}
        confidence_scores = []
        question_lengths = []
        answer_lengths = []

        for pair in pairs:
            type_dist[pair.question_type] = type_dist.get(pair.question_type, 0) + 1
            difficulty_dist[pair.difficulty] = (
                difficulty_dist.get(pair.difficulty, 0) + 1
            )
            confidence_scores.append(pair.confidence_score)
            question_lengths.append(len(pair.question))
            answer_lengths.append(len(pair.answer))

        analysis: dict[str, Any] = {
            "total_pairs": len(pairs),
            "type_distribution": type_dist,
            "difficulty_distribution": difficulty_dist,
            "avg_confidence": sum(confidence_scores) / len(confidence_scores),
            "confidence_range": [min(confidence_scores), max(confidence_scores)],
            "avg_question_length": sum(question_lengths) / len(question_lengths),
            "avg_answer_length": sum(answer_lengths) / len(answer_lengths),
            "question_length_range": [min(question_lengths), max(question_lengths)],
            "answer_length_range": [min(answer_lengths), max(answer_lengths)],
        }

        # Quality indicators
        analysis["diversity_score"] = len(type_dist) / len(QuestionType)
        analysis["balance_score"] = (
            min(type_dist.values()) / max(type_dist.values()) if type_dist else 0
        )
        analysis["quality_indicators"] = {
            "high_confidence_rate": sum(1 for c in confidence_scores if c >= 0.8)
            / len(confidence_scores),
            "appropriate_length_rate": sum(
                1
                for a_len in answer_lengths
                if self.config.generation.min_answer_length
                <= a_len
                <= self.config.generation.max_answer_length
            )
            / len(answer_lengths),
        }

        if self.verbose:
            logger.info("📊 Generation Quality Analysis:")
            logger.info(f"  • Diversity score: {analysis['diversity_score']:.2f}")
            logger.info(f"  • Balance score: {analysis['balance_score']:.2f}")
            logger.info(f"  • Average confidence: {analysis['avg_confidence']:.2f}")
            logger.info(
                f"  • High confidence rate: {analysis['quality_indicators']['high_confidence_rate']:.1%}"
            )

        return analysis

    def _save_progress(self, valid_pairs: list[QAPair], progress_file: str) -> None:
        """
        Save generated Q&A pairs progressively to a file.

        Args:
            valid_pairs: List of validated Q&A pairs to save
            progress_file: Path to the progress file

        Educational Note: Progressive saving allows recovery from interruptions
        in large document processing and provides real-time progress visibility.
        """
        try:
            from datetime import datetime

            with open(progress_file, "a", encoding="utf-8") as f:
                for pair in valid_pairs:
                    # Convert to Alpaca format for consistency
                    alpaca_entry = {
                        "instruction": pair.question,
                        "input": "",
                        "output": pair.answer,
                        "metadata": {
                            "id": pair.id,
                            "type": pair.question_type.value,
                            "difficulty": pair.difficulty.value,
                            "confidence_score": pair.confidence_score,
                            "source_context": pair.context[:100] + "..."
                            if len(pair.context) > 100
                            else pair.context,
                            "source_page": pair.source_page,
                            "timestamp": datetime.now().isoformat(),
                        },
                    }
                    f.write(json.dumps(alpaca_entry, ensure_ascii=False) + "\n")

            if self.verbose:
                logger.info(f"💾 Saved {len(valid_pairs)} pairs to progress file")

        except Exception as e:
            logger.error(f"❌ Failed to save progress: {str(e)}")
            # Don't raise - progress saving failure shouldn't stop the main process
