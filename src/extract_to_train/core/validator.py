"""
Dataset validation module for ensuring high-quality training data.

This module provides comprehensive validation of Q&A pairs using specialized
reasoning models to ensure dataset quality for effective fine-tuning.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel

from ..llm.client import OllamaClient
from ..llm.prompts import PromptTemplates
from ..models.config import AppConfig
from ..models.dataset import QAPair, ValidationResult

logger = logging.getLogger(__name__)


class ValidationStats(BaseModel):
    """
    Statistics from the validation process.

    Educational Note: Validation statistics help identify systematic
    issues and track improvements in dataset quality over time.
    """

    total_pairs_validated: int = 0
    pairs_passed: int = 0
    pairs_failed: int = 0
    avg_validation_time: float = 0.0
    avg_overall_score: float = 0.0
    avg_accuracy_score: float = 0.0
    avg_completeness_score: float = 0.0
    avg_clarity_score: float = 0.0
    avg_training_value_score: float = 0.0
    common_issues: dict[str, int] = {}
    validation_errors: int = 0


class DatasetValidator:
    """
    Educational dataset validator with comprehensive quality assessment.

    Uses specialized reasoning models to provide thorough evaluation
    of Q&A pairs with detailed feedback for improvement.
    """

    def __init__(
        self, ollama_client: OllamaClient, config: AppConfig, verbose: bool = False
    ):
        """
        Initialize the dataset validator.

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
            logger.info("🔍 Dataset Validator initialized")
            logger.info(f"🤖 Validation model: {config.models['validation'].name}")
            logger.info(f"🌡️  Temperature: {config.models['validation'].temperature}")
            logger.info(f"🎯 Min confidence: {config.validation.min_confidence_score}")
            logger.info(f"📊 Required scores: {config.validation.required_scores}")

    async def validate_dataset(
        self, qa_pairs: list[QAPair], enable_auto_correction: bool = False
    ) -> tuple[list[QAPair], list[QAPair], ValidationStats]:
        """
        Validate a complete dataset of Q&A pairs.

        Args:
            qa_pairs: List of Q&A pairs to validate
            enable_auto_correction: Whether to attempt automatic corrections

        Returns:
            Tuple of (valid_pairs, invalid_pairs, validation_statistics)

        Educational Note: Batch validation provides comprehensive quality
        control and enables systematic improvement of training datasets.
        """
        if self.verbose:
            logger.info(f"🚀 Starting validation of {len(qa_pairs)} Q&A pairs")
            logger.info(
                f"🛠️  Auto-correction: {'enabled' if enable_auto_correction else 'disabled'}"
            )

        valid_pairs = []
        invalid_pairs = []
        stats = ValidationStats()
        total_validation_time = 0.0

        # Track scores for statistics
        all_scores: dict[str, list[float]] = {
            "overall": [],
            "accuracy": [],
            "completeness": [],
            "clarity": [],
            "training_value": [],
        }

        for i, qa_pair in enumerate(qa_pairs, 1):
            if self.verbose:
                logger.info(
                    f"🔍 Validating pair {i}/{len(qa_pairs)} (ID: {qa_pair.id[:8]}...)"
                )

            try:
                # Validate the pair
                validation_result, validation_time = await self._validate_single_pair(
                    qa_pair
                )

                total_validation_time += validation_time
                stats.total_pairs_validated += 1

                # Collect scores for statistics
                all_scores["overall"].append(validation_result.overall_score)
                all_scores["accuracy"].append(validation_result.accuracy_score)
                all_scores["completeness"].append(validation_result.completeness_score)
                all_scores["clarity"].append(validation_result.clarity_score)
                all_scores["training_value"].append(
                    validation_result.training_value_score
                )

                # Track common issues
                for issue in validation_result.issues:
                    stats.common_issues[issue] = stats.common_issues.get(issue, 0) + 1

                # Determine if pair passes validation
                if self._passes_validation_criteria(validation_result):
                    valid_pairs.append(qa_pair)
                    stats.pairs_passed += 1

                    if self.verbose:
                        logger.info(
                            f"✅ Pair passed (Score: {validation_result.overall_score:.1f})"
                        )

                else:
                    # Try auto-correction if enabled
                    if enable_auto_correction and validation_result.suggestions:
                        corrected_pair = await self._attempt_correction(
                            qa_pair, validation_result
                        )

                        if corrected_pair:
                            # Re-validate corrected pair
                            corrected_result, _ = await self._validate_single_pair(
                                corrected_pair
                            )

                            if self._passes_validation_criteria(corrected_result):
                                valid_pairs.append(corrected_pair)
                                stats.pairs_passed += 1

                                if self.verbose:
                                    logger.info("🔧 Pair corrected and passed")
                                continue

                    invalid_pairs.append(qa_pair)
                    stats.pairs_failed += 1

                    if self.verbose:
                        logger.info(
                            f"❌ Pair failed (Score: {validation_result.overall_score:.1f})"
                        )
                        if validation_result.issues:
                            logger.info(
                                f"🐛 Issues: {', '.join(validation_result.issues[:2])}"
                            )

            except Exception as e:
                stats.validation_errors += 1
                logger.error(f"❌ Error validating pair {i}: {str(e)}")

                # Treat validation errors as failed pairs
                invalid_pairs.append(qa_pair)

        # Calculate final statistics
        if stats.total_pairs_validated > 0:
            stats.avg_validation_time = (
                total_validation_time / stats.total_pairs_validated
            )
            stats.avg_overall_score = sum(all_scores["overall"]) / len(
                all_scores["overall"]
            )
            stats.avg_accuracy_score = sum(all_scores["accuracy"]) / len(
                all_scores["accuracy"]
            )
            stats.avg_completeness_score = sum(all_scores["completeness"]) / len(
                all_scores["completeness"]
            )
            stats.avg_clarity_score = sum(all_scores["clarity"]) / len(
                all_scores["clarity"]
            )
            stats.avg_training_value_score = sum(all_scores["training_value"]) / len(
                all_scores["training_value"]
            )

        if self.verbose:
            self._log_validation_summary(stats)

        return valid_pairs, invalid_pairs, stats

    async def _validate_single_pair(
        self, qa_pair: QAPair
    ) -> tuple[ValidationResult, float]:
        """
        Validate a single Q&A pair using the reasoning model.

        Args:
            qa_pair: Q&A pair to validate

        Returns:
            Tuple of (validation_result, validation_time)

        Educational Note: Single pair validation demonstrates how to use
        specialized reasoning models for quality assessment.
        """
        # Prepare validation prompt
        validation_prompt = self.prompt_templates.get_validation_prompt()
        formatted_prompt = validation_prompt.format(
            question=qa_pair.question, answer=qa_pair.answer, context=qa_pair.context
        )

        if self.verbose:
            logger.info(
                f"🎨 Using validation prompt with {len(formatted_prompt)} characters"
            )

        # Get validation response
        response = await self.client.generate_response(
            formatted_prompt, self.config.models["validation"]
        )

        if not response.success:
            raise RuntimeError(f"Validation failed: {response.error_message}")

        # Parse validation result
        try:
            validation_data = self._parse_validation_response(response.content)

            validation_result = ValidationResult(
                is_valid=validation_data.get("is_valid", False),
                accuracy_score=int(validation_data.get("accuracy_score", 0)),
                completeness_score=int(validation_data.get("completeness_score", 0)),
                clarity_score=int(validation_data.get("clarity_score", 0)),
                training_value_score=int(
                    validation_data.get("training_value_score", 0)
                ),
                issues=validation_data.get("issues", []),
                suggestions=validation_data.get("suggestions", []),
            )

            if self.verbose:
                logger.info(
                    f"📊 Validation scores: A:{validation_result.accuracy_score} "
                    f"C:{validation_result.completeness_score} "
                    f"Cl:{validation_result.clarity_score} "
                    f"T:{validation_result.training_value_score}"
                )

            return validation_result, response.response_time

        except Exception as e:
            logger.error(f"❌ Failed to parse validation response: {str(e)}")
            logger.error(f"🔍 Response preview: {response.content[:200]}...")
            raise

    def _parse_validation_response(self, response_content: str) -> dict[str, Any]:
        """
        Parse validation response from the reasoning model.

        Args:
            response_content: Raw response from validation model

        Returns:
            Dictionary with validation results

        Educational Note: Parsing validation responses requires robust
        error handling due to the structured nature of the output.
        """
        content = response_content.strip()

        # Try to find JSON in the response
        json_content = None

        # Look for JSON markers
        start_markers = ["{", "```json\n{", "```\n{"]
        end_markers = ["}", "}\n```", "}\n```"]

        for start_marker, end_marker in zip(start_markers, end_markers, strict=False):
            start_idx = content.find(start_marker)
            if start_idx != -1:
                end_idx = content.rfind(end_marker)
                if end_idx > start_idx:
                    json_content = content[start_idx : end_idx + 1]
                    break

        if json_content is None:
            json_content = content

        try:
            validation_data: dict[str, Any] = json.loads(json_content)

            # Ensure required fields have defaults
            defaults = {
                "is_valid": False,
                "accuracy_score": 0,
                "completeness_score": 0,
                "clarity_score": 0,
                "training_value_score": 0,
                "issues": [],
                "suggestions": [],
            }

            for key, default_value in defaults.items():
                if key not in validation_data:
                    validation_data[key] = default_value

            # Validate score ranges
            for score_key in [
                "accuracy_score",
                "completeness_score",
                "clarity_score",
                "training_value_score",
            ]:
                score = validation_data.get(score_key, 0)
                validation_data[score_key] = max(0, min(10, int(score)))

            return validation_data

        except json.JSONDecodeError:
            # Try to extract scores using text parsing as fallback
            return self._parse_validation_fallback(content)

    def _parse_validation_fallback(self, content: str) -> dict[str, Any]:
        """
        Fallback parser for validation responses that aren't valid JSON.

        Args:
            content: Raw response content

        Returns:
            Dictionary with extracted validation data
        """
        import re

        # Default values
        result: dict[str, Any] = {
            "is_valid": False,
            "accuracy_score": 5,
            "completeness_score": 5,
            "clarity_score": 5,
            "training_value_score": 5,
            "issues": [],
            "suggestions": [],
        }

        # Try to extract scores using regex
        score_patterns = {
            "accuracy_score": r"accuracy[_\s]*score[:\s]*(\d+)",
            "completeness_score": r"completeness[_\s]*score[:\s]*(\d+)",
            "clarity_score": r"clarity[_\s]*score[:\s]*(\d+)",
            "training_value_score": r"training[_\s]*value[_\s]*score[:\s]*(\d+)",
        }

        for key, pattern in score_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                result[key] = max(0, min(10, int(match.group(1))))

        # Determine validity based on average score
        avg_score = (
            sum(
                [
                    result["accuracy_score"],
                    result["completeness_score"],
                    result["clarity_score"],
                    result["training_value_score"],
                ]
            )
            / 4
        )

        result["is_valid"] = avg_score >= 7.0

        # Try to extract issues and suggestions
        if "issues" in content.lower():
            result["issues"] = ["Parsing issues detected in validation response"]

        if "suggestions" in content.lower() or "improve" in content.lower():
            result["suggestions"] = ["Review and improve based on validation feedback"]

        logger.warning("🔧 Used fallback parsing for validation response")
        return result

    def _passes_validation_criteria(self, validation_result: ValidationResult) -> bool:
        """
        Determine if a validation result meets the acceptance criteria.

        Args:
            validation_result: Result from validation process

        Returns:
            True if the pair meets quality standards

        Educational Note: Clear validation criteria ensure consistent
        quality standards across the entire dataset.
        """
        # Check overall score threshold
        if validation_result.overall_score < (
            sum(self.config.validation.required_scores.values())
            / len(self.config.validation.required_scores)
        ):
            return False

        # Check individual score requirements
        score_mapping = {
            "accuracy": validation_result.accuracy_score,
            "completeness": validation_result.completeness_score,
            "clarity": validation_result.clarity_score,
            "training_value": validation_result.training_value_score,
        }

        for criterion, required_score in self.config.validation.required_scores.items():
            if score_mapping.get(criterion, 0) < required_score:
                return False

        # Check for critical issues
        critical_issues = ["hallucination", "factual error", "incoherent", "off-topic"]

        for issue in validation_result.issues:
            if any(critical in issue.lower() for critical in critical_issues):
                return False

        return True

    async def _attempt_correction(
        self, qa_pair: QAPair, validation_result: ValidationResult
    ) -> QAPair | None:
        """
        Attempt to automatically correct a failed Q&A pair.

        Args:
            qa_pair: Original Q&A pair that failed validation
            validation_result: Validation result with issues and suggestions

        Returns:
            Corrected QAPair or None if correction failed

        Educational Note: Automatic correction can salvage borderline
        examples instead of discarding them entirely.
        """
        if not validation_result.suggestions:
            return None

        try:
            # Prepare correction prompt
            correction_prompt = self.prompt_templates.get_correction_prompt()
            formatted_prompt = correction_prompt.format(
                question=qa_pair.question,
                answer=qa_pair.answer,
                context=qa_pair.context,
                issues="; ".join(validation_result.issues),
                suggestions="; ".join(validation_result.suggestions),
            )

            if self.verbose:
                logger.info(f"🔧 Attempting correction for pair {qa_pair.id[:8]}...")

            # Get correction response
            response = await self.client.generate_response(
                formatted_prompt,
                self.config.models["validation"],  # Use same model for consistency
            )

            if not response.success:
                logger.warning(
                    f"⚠️  Correction generation failed: {response.error_message}"
                )
                return None

            # Parse correction result
            correction_data = json.loads(response.content)

            # Create corrected pair
            corrected_pair = QAPair(
                id=qa_pair.id,  # Keep same ID
                question=correction_data.get("corrected_question", qa_pair.question),
                answer=correction_data.get("corrected_answer", qa_pair.answer),
                context=qa_pair.context,
                difficulty=qa_pair.difficulty,
                question_type=qa_pair.question_type,
                confidence_score=float(correction_data.get("confidence", 0.7)),
                source_page=qa_pair.source_page,
            )

            if self.verbose:
                logger.info(
                    f"🔧 Correction completed with confidence {corrected_pair.confidence_score:.2f}"
                )

            return corrected_pair

        except Exception as e:
            logger.warning(f"⚠️  Correction failed: {str(e)}")
            return None

    def _log_validation_summary(self, stats: ValidationStats) -> None:
        """
        Log a comprehensive validation summary.

        Args:
            stats: Validation statistics to summarize
        """
        pass_rate = (
            stats.pairs_passed / stats.total_pairs_validated
            if stats.total_pairs_validated > 0
            else 0
        )

        logger.info("🎉 Validation completed!")
        logger.info("📊 Validation Summary:")
        logger.info(f"  • Total pairs validated: {stats.total_pairs_validated}")
        logger.info(f"  • Passed: {stats.pairs_passed} ({pass_rate:.1%})")
        logger.info(f"  • Failed: {stats.pairs_failed}")
        logger.info(f"  • Errors: {stats.validation_errors}")
        logger.info(f"  • Average validation time: {stats.avg_validation_time:.2f}s")
        logger.info("📈 Quality Scores (0-10):")
        logger.info(f"  • Overall: {stats.avg_overall_score:.1f}")
        logger.info(f"  • Accuracy: {stats.avg_accuracy_score:.1f}")
        logger.info(f"  • Completeness: {stats.avg_completeness_score:.1f}")
        logger.info(f"  • Clarity: {stats.avg_clarity_score:.1f}")
        logger.info(f"  • Training Value: {stats.avg_training_value_score:.1f}")

        if stats.common_issues:
            logger.info("🐛 Most Common Issues:")
            sorted_issues = sorted(
                stats.common_issues.items(), key=lambda x: x[1], reverse=True
            )
            for issue, count in sorted_issues[:5]:
                logger.info(f"  • {issue}: {count} times")

    def get_validation_report(
        self,
        valid_pairs: list[QAPair],
        invalid_pairs: list[QAPair],
        stats: ValidationStats,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive validation report.

        Args:
            valid_pairs: List of pairs that passed validation
            invalid_pairs: List of pairs that failed validation
            stats: Validation statistics

        Returns:
            Dictionary with complete validation report

        Educational Note: Detailed reports help understand dataset
        quality and guide improvements in the generation process.
        """
        total_pairs = len(valid_pairs) + len(invalid_pairs)

        return {
            "summary": {
                "total_pairs": total_pairs,
                "valid_pairs": len(valid_pairs),
                "invalid_pairs": len(invalid_pairs),
                "pass_rate": len(valid_pairs) / total_pairs if total_pairs > 0 else 0,
                "validation_errors": stats.validation_errors,
            },
            "quality_metrics": {
                "avg_overall_score": stats.avg_overall_score,
                "avg_accuracy_score": stats.avg_accuracy_score,
                "avg_completeness_score": stats.avg_completeness_score,
                "avg_clarity_score": stats.avg_clarity_score,
                "avg_training_value_score": stats.avg_training_value_score,
            },
            "performance": {
                "avg_validation_time": stats.avg_validation_time,
                "total_validation_time": stats.avg_validation_time
                * stats.total_pairs_validated,
            },
            "common_issues": dict(
                sorted(stats.common_issues.items(), key=lambda x: x[1], reverse=True)
            ),
            "recommendations": self._generate_recommendations(
                stats, valid_pairs, invalid_pairs
            ),
        }

    def _generate_recommendations(
        self,
        stats: ValidationStats,
        valid_pairs: list[QAPair],
        invalid_pairs: list[QAPair],
    ) -> list[str]:
        """
        Generate improvement recommendations based on validation results.

        Args:
            stats: Validation statistics
            valid_pairs: Valid Q&A pairs
            invalid_pairs: Invalid Q&A pairs

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        # Pass rate recommendations
        pass_rate = (
            stats.pairs_passed / stats.total_pairs_validated
            if stats.total_pairs_validated > 0
            else 0
        )

        if pass_rate < 0.7:
            recommendations.append(
                "Low pass rate (<70%): Consider adjusting generation prompts or model temperature"
            )
        elif pass_rate > 0.95:
            recommendations.append(
                "Very high pass rate (>95%): Consider raising validation standards for better quality"
            )

        # Score-specific recommendations
        if stats.avg_accuracy_score < 7.0:
            recommendations.append(
                "Low accuracy scores: Improve context adherence in generation prompts"
            )

        if stats.avg_completeness_score < 7.0:
            recommendations.append(
                "Low completeness scores: Encourage more comprehensive answers in prompts"
            )

        if stats.avg_clarity_score < 7.0:
            recommendations.append(
                "Low clarity scores: Focus on improving answer structure and readability"
            )

        if stats.avg_training_value_score < 7.0:
            recommendations.append(
                "Low training value: Generate more diverse question types and difficulty levels"
            )

        # Common issues recommendations
        if stats.common_issues:
            top_issue = max(stats.common_issues.items(), key=lambda x: x[1])
            if top_issue[1] > stats.total_pairs_validated * 0.2:  # >20% of pairs
                recommendations.append(
                    f"Address common issue: '{top_issue[0]}' affects {top_issue[1]} pairs"
                )

        if not recommendations:
            recommendations.append(
                "Validation results look good! Consider minor optimizations for even better quality."
            )

        return recommendations
