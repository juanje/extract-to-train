"""
Main processing pipeline for educational dataset creation.

This module orchestrates the complete workflow from PDF extraction
to validated Q&A dataset generation with comprehensive educational logging.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..llm.client import OllamaClient
from ..models.config import AppConfig
from ..models.dataset import Dataset, DatasetFormat, DatasetStats
from .extractor import DocumentExtractor, ExtractionResult
from .generator import GenerationStats, QAGenerator
from .validator import DatasetValidator, ValidationStats

logger = logging.getLogger(__name__)


class PipelineStats(BaseModel):
    """
    Comprehensive statistics from the entire pipeline execution.

    Educational Note: Pipeline statistics provide complete visibility
    into the dataset creation process for learning and optimization.
    """

    total_execution_time: float = 0.0
    extraction_time: float = 0.0
    generation_time: float = 0.0
    validation_time: float = 0.0

    # Source document stats
    source_filename: str = ""
    source_pages: int = 0
    chunks_created: int = 0

    # Generation stats
    pairs_generated: int = 0
    pairs_validated: int = 0
    pairs_passed: int = 0
    pairs_failed: int = 0

    # Quality metrics
    final_pass_rate: float = 0.0
    avg_confidence_score: float = 0.0
    avg_quality_score: float = 0.0

    # Performance metrics
    chunks_per_second: float = 0.0
    pairs_per_minute: float = 0.0


class ExtractToTrainPipeline:
    """
    Educational pipeline for complete PDF-to-dataset transformation.

    This pipeline provides a transparent, educational approach to creating
    high-quality training datasets from PDF documents.
    """

    def __init__(self, config: AppConfig, verbose: bool = False):
        """
        Initialize the complete pipeline.

        Args:
            config: Application configuration
            verbose: Enable detailed educational logging
        """
        self.config = config
        self.verbose = verbose

        # Initialize components
        self.ollama_client = OllamaClient(config.ollama, verbose=verbose)
        self.extractor = DocumentExtractor(verbose=verbose)
        self.generator = QAGenerator(self.ollama_client, config, verbose=verbose)
        self.validator = DatasetValidator(self.ollama_client, config, verbose=verbose)

        if self.verbose:
            logger.info("🚀 Extract-to-Train Pipeline initialized")
            logger.info(
                f"🔧 Configuration: {config.models['extraction'].name} + {config.models['validation'].name}"
            )
            logger.info(
                f"📊 Target: {config.processing.max_pairs_per_chunk} pairs per chunk"
            )

    def set_document_language(self, language: str | None) -> None:
        """
        Set the language for document processing.

        Args:
            language: Document language code (e.g., 'es', 'en', 'fr')

        Educational Note: Specifying document language helps optimize
        Q&A generation for non-English content.
        """
        self._document_language = language
        if self.verbose and language:
            logger.info(f"🌐 Document language set to: {language}")

    async def process_pdf(
        self,
        pdf_path: Path,
        output_path: Path | None = None,
        max_pairs: int | None = None,
        max_chunks: int | None = None,
        skip_validation: bool = False,
        enable_auto_correction: bool = False,
        progress_file: str | None = None,
    ) -> tuple[Dataset, PipelineStats]:
        """
        Process a PDF document into a validated training dataset.

        Args:
            pdf_path: Path to the PDF document
            output_path: Optional output path for the dataset
            max_pairs: Maximum number of Q&A pairs to generate
            max_chunks: Maximum number of chunks/pages to process (for testing large docs)
            skip_validation: Skip the validation step for faster processing
            enable_auto_correction: Enable automatic correction of failed pairs
            progress_file: File to save progressive Q&A pairs during processing

        Returns:
            Tuple of (final_dataset, pipeline_statistics)

        Educational Note: This method demonstrates the complete workflow
        for transforming documents into training-ready datasets. Progress saving
        allows recovery from interruptions in large document processing.
        """
        if self.verbose:
            logger.info(f"📖 Starting pipeline for: {pdf_path.name}")
            logger.info(f"🎯 Max pairs: {max_pairs or 'unlimited'}")
            logger.info(f"📄 Max chunks: {max_chunks or 'unlimited'}")
            logger.info(
                f"✅ Validation: {'disabled' if skip_validation else 'enabled'}"
            )
            logger.info(
                f"🔧 Auto-correction: {'enabled' if enable_auto_correction else 'disabled'}"
            )
            if progress_file:
                logger.info(f"📝 Progress file: {progress_file}")

        pipeline_start_time = time.time()
        stats = PipelineStats(source_filename=pdf_path.name)

        try:
            # Step 1: Validate environment (quick connectivity check only)
            await self._validate_environment(full_validation=False)

            # Step 2: Extract content from document
            if self.verbose:
                logger.info("📄 Step 1: Extracting content from document...")

            extraction_start = time.time()
            extraction_result = self._extract_document_content(pdf_path)
            stats.extraction_time = time.time() - extraction_start

            if not extraction_result.success:
                raise RuntimeError(
                    f"Document extraction failed: {extraction_result.error_message}"
                )

            stats.source_pages = extraction_result.metadata.total_pages
            stats.chunks_created = len(extraction_result.chunks)

            if self.verbose:
                logger.info(
                    f"✅ Extracted {len(extraction_result.chunks)} chunks from {stats.source_pages} pages"
                )

            # Limit chunks if specified (for testing large documents)
            chunks_to_process = extraction_result.chunks
            if max_chunks and max_chunks < len(extraction_result.chunks):
                chunks_to_process = extraction_result.chunks[:max_chunks]
                if self.verbose:
                    logger.info(
                        f"📄 Limited to first {max_chunks} chunks for processing"
                    )

            # Step 3: Generate Q&A pairs
            if self.verbose:
                logger.info("🤖 Step 2: Generating Q&A pairs...")

            generation_start = time.time()
            qa_pairs, generation_stats = await self.generator.generate_qa_pairs(
                chunks_to_process,
                max_pairs_total=max_pairs,
                document_language=extraction_result.metadata.language,
                progress_file=progress_file,
            )
            stats.generation_time = time.time() - generation_start
            stats.pairs_generated = len(qa_pairs)

            if self.verbose:
                logger.info(f"✅ Generated {len(qa_pairs)} Q&A pairs")

            # Step 4: Validate dataset (optional)
            valid_pairs = qa_pairs
            validation_stats = ValidationStats()

            if not skip_validation:
                if self.verbose:
                    logger.info("🔍 Step 3: Validating dataset quality...")

                validation_start = time.time()
                (
                    valid_pairs,
                    invalid_pairs,
                    validation_stats,
                ) = await self.validator.validate_dataset(
                    qa_pairs, enable_auto_correction=enable_auto_correction
                )
                stats.validation_time = time.time() - validation_start

                stats.pairs_validated = len(qa_pairs)
                stats.pairs_passed = len(valid_pairs)
                stats.pairs_failed = len(invalid_pairs)

                if self.verbose:
                    logger.info(
                        f"✅ Validation complete: {len(valid_pairs)}/{len(qa_pairs)} pairs passed"
                    )

            # Step 5: Create final dataset
            if self.verbose:
                logger.info("📊 Step 4: Creating final dataset...")

            final_dataset = self._create_final_dataset(
                valid_pairs, extraction_result, generation_stats, validation_stats
            )

            # Step 6: Calculate final statistics
            stats.total_execution_time = time.time() - pipeline_start_time
            stats.final_pass_rate = (
                stats.pairs_passed / stats.pairs_generated
                if stats.pairs_generated > 0
                else 1.0
            )
            stats.avg_confidence_score = (
                sum(p.confidence_score for p in valid_pairs) / len(valid_pairs)
                if valid_pairs
                else 0.0
            )

            # Performance metrics
            if stats.extraction_time > 0:
                stats.chunks_per_second = stats.chunks_created / stats.extraction_time
            if stats.generation_time > 0:
                stats.pairs_per_minute = (
                    stats.pairs_generated / stats.generation_time
                ) * 60

            # Step 7: Save dataset if output path specified
            if output_path:
                self._save_dataset(final_dataset, output_path)

            if self.verbose:
                self._log_pipeline_summary(stats, final_dataset)

            return final_dataset, stats

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise

    async def _validate_environment(self, full_validation: bool = False) -> None:
        """
        Validate that the environment is properly configured.

        Args:
            full_validation: If True, checks all models. If False, only checks connectivity.

        Educational Note: Environment validation prevents runtime failures
        and provides clear guidance for setup issues.
        """
        if self.verbose:
            logger.info("🔍 Validating environment...")

        if full_validation:
            # Full validation: Check Ollama connectivity and all models
            is_valid, errors = self.ollama_client.validate_configuration(
                self.config.models
            )

            if not is_valid:
                error_msg = "Environment validation failed:\n" + "\n".join(
                    f"  • {error}" for error in errors
                )
                raise RuntimeError(error_msg)
        else:
            # Quick validation: Only check Ollama connectivity
            is_connected, error_msg_result = self.ollama_client.validate_connectivity()
            error_msg = error_msg_result or "Unknown error"

            if not is_connected:
                raise RuntimeError(f"Ollama connection failed: {error_msg or 'Unknown error'}")

        if self.verbose:
            logger.info("✅ Environment validation passed")

    def _extract_document_content(self, document_path: Path) -> ExtractionResult:
        """
        Extract content from document (PDF or Markdown).

        Args:
            document_path: Path to document file

        Returns:
            ExtractionResult with content and metadata
        """
        # Validate document first
        is_valid, error_msg = self.extractor.validate_document(document_path)
        if not is_valid:
            raise ValueError(f"Document validation failed: {error_msg}")

        # Extract content
        extraction_result = self.extractor.extract_from_document(
            document_path,
            chunk_size=self.config.processing.chunk_size,
            chunk_overlap=self.config.processing.chunk_overlap,
            language=getattr(self, "_document_language", None),
        )

        return extraction_result

    def _create_final_dataset(
        self,
        qa_pairs: list[Any],
        extraction_result: ExtractionResult,
        generation_stats: GenerationStats,
        validation_stats: ValidationStats,
    ) -> Dataset:
        """
        Create the final dataset with complete metadata.

        Args:
            qa_pairs: Final list of Q&A pairs
            extraction_result: Results from PDF extraction
            generation_stats: Statistics from generation process
            validation_stats: Statistics from validation process

        Returns:
            Complete Dataset object
        """
        # Calculate dataset statistics
        if qa_pairs:
            dataset_stats = DatasetStats(
                total_pairs=len(qa_pairs),
                avg_question_length=sum(len(p.question) for p in qa_pairs)
                // len(qa_pairs),
                avg_answer_length=sum(len(p.answer) for p in qa_pairs) // len(qa_pairs),
                difficulty_distribution={},
                question_type_distribution={},
                pages_covered=len({p.source_page for p in qa_pairs if p.source_page}),
                estimated_tokens=sum(len(p.question) + len(p.answer) for p in qa_pairs)
                // 4,  # Rough estimate
                avg_confidence_score=sum(p.confidence_score for p in qa_pairs)
                / len(qa_pairs),
                validation_pass_rate=validation_stats.pairs_passed
                / validation_stats.total_pairs_validated
                if validation_stats.total_pairs_validated > 0
                else 1.0,
            )

            # Calculate distributions
            for pair in qa_pairs:
                dataset_stats.difficulty_distribution[pair.difficulty] = (
                    dataset_stats.difficulty_distribution.get(pair.difficulty, 0) + 1
                )
                dataset_stats.question_type_distribution[pair.question_type] = (
                    dataset_stats.question_type_distribution.get(pair.question_type, 0)
                    + 1
                )
        else:
            dataset_stats = DatasetStats(
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

        # Create comprehensive metadata
        metadata = {
            "source_document": extraction_result.metadata.filename,
            "extraction_method": "docling",
            "generation_model": self.config.models["extraction"].name,
            "validation_model": self.config.models["validation"].name,
            "processing_config": {
                "chunk_size": self.config.processing.chunk_size,
                "chunk_overlap": self.config.processing.chunk_overlap,
                "max_pairs_per_chunk": self.config.processing.max_pairs_per_chunk,
            },
            "generation_stats": {
                "total_chunks_processed": generation_stats.total_chunks_processed,
                "success_rate": generation_stats.success_rate,
                "avg_generation_time": generation_stats.avg_generation_time,
            },
            "validation_stats": {
                "total_validated": validation_stats.total_pairs_validated,
                "pass_rate": validation_stats.pairs_passed
                / validation_stats.total_pairs_validated
                if validation_stats.total_pairs_validated > 0
                else 1.0,
                "avg_quality_score": validation_stats.avg_overall_score,
            },
            "document_metadata": extraction_result.metadata.dict(),
        }

        return Dataset(metadata=metadata, qa_pairs=qa_pairs, stats=dataset_stats)

    def _save_dataset(self, dataset: Dataset, output_path: Path) -> None:
        """
        Save dataset to file.

        Args:
            dataset: Dataset to save
            output_path: Output file path
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON for now (will be enhanced with format options in CLI)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset.dict(), f, indent=2, ensure_ascii=False)

        if self.verbose:
            logger.info(f"💾 Dataset saved to: {output_path}")
            logger.info(f"📊 File size: {output_path.stat().st_size / 1024:.1f} KB")

    def export_dataset(
        self,
        dataset: Dataset,
        output_path: Path,
        format_type: DatasetFormat,
        **format_options: Any,
    ) -> None:
        """
        Export dataset in specified format.

        Args:
            dataset: Dataset to export
            output_path: Output file path
            format_type: Target format (alpaca, sharegpt, openai)
            **format_options: Format-specific options

        Educational Note: Different formats serve different fine-tuning
        frameworks and use cases.
        """
        if self.verbose:
            logger.info(f"📤 Exporting dataset in {format_type} format...")

        # Convert to target format
        export_data = dataset.export_format(format_type, **format_options)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSONL (one JSON object per line)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in export_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self.verbose:
            logger.info(f"💾 Exported {len(export_data)} entries to: {output_path}")
            logger.info(f"📊 Format: {format_type}")
            logger.info(f"💿 File size: {output_path.stat().st_size / 1024:.1f} KB")

    def _log_pipeline_summary(self, stats: PipelineStats, dataset: Dataset) -> None:
        """
        Log a comprehensive pipeline execution summary.

        Args:
            stats: Pipeline execution statistics
            dataset: Final dataset
        """
        logger.info("🎉 Pipeline execution completed successfully!")
        logger.info("=" * 60)
        logger.info("📊 PIPELINE SUMMARY")
        logger.info("=" * 60)

        # Source document info
        logger.info("📄 Source Document:")
        logger.info(f"  • File: {stats.source_filename}")
        logger.info(f"  • Pages: {stats.source_pages}")
        logger.info(f"  • Chunks created: {stats.chunks_created}")

        # Processing performance
        logger.info("⏱️  Processing Performance:")
        logger.info(f"  • Total time: {stats.total_execution_time:.1f}s")
        logger.info(
            f"  • Extraction: {stats.extraction_time:.1f}s ({stats.extraction_time / stats.total_execution_time:.1%})"
        )
        logger.info(
            f"  • Generation: {stats.generation_time:.1f}s ({stats.generation_time / stats.total_execution_time:.1%})"
        )
        logger.info(
            f"  • Validation: {stats.validation_time:.1f}s ({stats.validation_time / stats.total_execution_time:.1%})"
        )
        logger.info(
            f"  • Processing speed: {stats.chunks_per_second:.1f} chunks/sec, {stats.pairs_per_minute:.1f} pairs/min"
        )

        # Dataset quality
        logger.info("📈 Dataset Quality:")
        logger.info(f"  • Generated pairs: {stats.pairs_generated}")
        logger.info(f"  • Final pairs: {len(dataset.qa_pairs)}")
        logger.info(f"  • Pass rate: {stats.final_pass_rate:.1%}")
        logger.info(f"  • Avg confidence: {stats.avg_confidence_score:.2f}")
        logger.info(f"  • Estimated tokens: ~{dataset.stats.estimated_tokens:,}")

        # Distribution info
        logger.info("🎯 Question Distribution:")
        for q_type, count in dataset.stats.question_type_distribution.items():
            percentage = count / dataset.stats.total_pairs * 100
            logger.info(f"  • {q_type.capitalize()}: {count} ({percentage:.1f}%)")

        logger.info("📊 Difficulty Distribution:")
        for difficulty, count in dataset.stats.difficulty_distribution.items():
            percentage = count / dataset.stats.total_pairs * 100
            logger.info(f"  • {difficulty.capitalize()}: {count} ({percentage:.1f}%)")

        # Educational insights
        logger.info("=" * 60)
        logger.info("💡 EDUCATIONAL INSIGHTS")
        logger.info("=" * 60)

        if stats.final_pass_rate >= 0.8:
            logger.info(
                "✅ High validation pass rate indicates good prompt engineering"
            )
        elif stats.final_pass_rate >= 0.6:
            logger.info("⚠️  Moderate pass rate - consider prompt optimization")
        else:
            logger.info(
                "❌ Low pass rate - review generation prompts and model settings"
            )

        if dataset.stats.estimated_tokens > 100000:
            logger.info("💰 Large dataset - consider token costs for fine-tuning")

        if len(set(dataset.stats.question_type_distribution.values())) == 1:
            logger.info("🎯 Perfect question type balance achieved")
        else:
            logger.info("📈 Consider adjusting generation for better type balance")

        logger.info("=" * 60)

    async def test_pipeline_components(self) -> dict[str, bool]:
        """
        Test all pipeline components to verify functionality.

        Returns:
            Dictionary with component test results

        Educational Note: Component testing helps identify setup issues
        before processing actual documents.
        """
        if self.verbose:
            logger.info("🧪 Testing pipeline components...")

        test_results = {}

        # Test Ollama connectivity
        try:
            available_models = self.ollama_client.get_available_models()
            test_results["ollama_connection"] = len(available_models) > 0

            if self.verbose:
                logger.info(f"✅ Ollama: {len(available_models)} models available")
        except Exception as e:
            test_results["ollama_connection"] = False
            logger.error(f"❌ Ollama connection failed: {e}")

        # Test model availability
        for model_type, model_config in self.config.models.items():
            try:
                available = self.ollama_client.check_model_availability(
                    model_config.name
                )
                test_results[f"model_{model_type}"] = available

                if self.verbose:
                    status = "✅" if available else "❌"
                    logger.info(f"{status} Model {model_type}: {model_config.name}")
            except Exception as e:
                test_results[f"model_{model_type}"] = False
                logger.error(f"❌ Model {model_type} check failed: {e}")

        # Test model responses
        if test_results.get("model_extraction", False):
            try:
                test_result = await self.ollama_client.test_model_response(
                    self.config.models["extraction"]
                )
                test_results["generation_model_response"] = test_result["success"]

                if self.verbose:
                    status = "✅" if test_result["success"] else "❌"
                    logger.info(f"{status} Generation model response test")
            except Exception as e:
                test_results["generation_model_response"] = False
                logger.error(f"❌ Generation model test failed: {e}")

        if test_results.get("model_validation", False):
            try:
                test_result = await self.ollama_client.test_model_response(
                    self.config.models["validation"]
                )
                test_results["validation_model_response"] = test_result["success"]

                if self.verbose:
                    status = "✅" if test_result["success"] else "❌"
                    logger.info(f"{status} Validation model response test")
            except Exception as e:
                test_results["validation_model_response"] = False
                logger.error(f"❌ Validation model test failed: {e}")

        all_passed = all(test_results.values())

        if self.verbose:
            logger.info(
                f"🧪 Component tests completed: {'✅ All passed' if all_passed else '❌ Some failed'}"
            )

        return test_results
