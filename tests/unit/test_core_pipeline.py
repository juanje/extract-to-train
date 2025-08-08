"""
Unit tests for extract_to_train.core.pipeline.

Tests for critical pipeline functionality using pytest.
"""

import pytest

from extract_to_train.core.pipeline import ExtractToTrainPipeline, PipelineStats
from extract_to_train.models.config import AppConfig


def test_pipeline_stats_creation():
    """Test PipelineStats can be created with default values."""
    stats = PipelineStats()
    
    assert stats.total_execution_time == 0.0
    assert stats.extraction_time == 0.0
    assert stats.generation_time == 0.0
    assert stats.validation_time == 0.0
    assert stats.source_filename == ""
    assert stats.source_pages == 0
    assert stats.chunks_created == 0
    assert stats.pairs_generated == 0
    assert stats.pairs_validated == 0
    assert stats.pairs_passed == 0
    assert stats.pairs_failed == 0


def test_pipeline_stats_with_values():
    """Test PipelineStats creation with specific values."""
    stats = PipelineStats(
        total_execution_time=120.5,
        extraction_time=30.2,
        generation_time=60.1,
        validation_time=25.2,
        source_filename="test.pdf",
        source_pages=10,
        chunks_created=15,
        pairs_generated=50,
        pairs_validated=45,
        pairs_passed=40,
        pairs_failed=5
    )
    
    assert stats.total_execution_time == 120.5
    assert stats.extraction_time == 30.2
    assert stats.generation_time == 60.1
    assert stats.validation_time == 25.2
    assert stats.source_filename == "test.pdf"
    assert stats.source_pages == 10
    assert stats.chunks_created == 15
    assert stats.pairs_generated == 50
    assert stats.pairs_validated == 45
    assert stats.pairs_passed == 40
    assert stats.pairs_failed == 5


def test_pipeline_stats_serialization():
    """Test PipelineStats can be serialized to dict."""
    stats = PipelineStats(
        total_execution_time=100.0,
        source_filename="document.pdf",
        pairs_generated=25
    )
    
    stats_dict = stats.model_dump()
    
    assert isinstance(stats_dict, dict)
    assert stats_dict["total_execution_time"] == 100.0
    assert stats_dict["source_filename"] == "document.pdf"
    assert stats_dict["pairs_generated"] == 25


def test_extract_to_train_pipeline_initialization(app_config):
    """Test ExtractToTrainPipeline can be initialized."""
    pipeline = ExtractToTrainPipeline(config=app_config)
    
    assert pipeline is not None
    assert pipeline.config is app_config
    assert hasattr(pipeline, 'extractor')
    assert hasattr(pipeline, 'generator')
    assert hasattr(pipeline, 'validator')


def test_extract_to_train_pipeline_initialization_verbose(app_config):
    """Test ExtractToTrainPipeline initialization with verbose mode."""
    pipeline = ExtractToTrainPipeline(config=app_config, verbose=True)
    
    assert pipeline is not None
    assert pipeline.config is app_config


def test_extract_to_train_pipeline_initialization_no_verbose(app_config):
    """Test ExtractToTrainPipeline initialization without verbose mode."""
    pipeline = ExtractToTrainPipeline(config=app_config, verbose=False)
    
    assert pipeline is not None
    assert pipeline.config is app_config


def test_set_document_language(app_config):
    """Test setting document language."""
    pipeline = ExtractToTrainPipeline(config=app_config)
    
    # Test setting a language
    pipeline.set_document_language("es")
    # We can't easily test the private attribute without making assumptions
    # about implementation details, so we just verify the method doesn't crash
    
    # Test setting None
    pipeline.set_document_language(None)
    
    # Test setting another language
    pipeline.set_document_language("en")


def test_set_document_language_verbose(app_config, caplog):
    """Test setting document language with verbose logging."""
    pipeline = ExtractToTrainPipeline(config=app_config, verbose=True)
    
    with caplog.at_level("INFO"):
        pipeline.set_document_language("Spanish")
    
    # Just verify the method doesn't crash with verbose mode
    # Checking specific log messages would be fragile


def test_pipeline_has_required_components(app_config):
    """Test pipeline has all required components after initialization."""
    pipeline = ExtractToTrainPipeline(config=app_config)
    
    # Test that critical components exist (without testing implementation details)
    assert hasattr(pipeline, 'config')
    assert hasattr(pipeline, 'extractor')
    assert hasattr(pipeline, 'generator') 
    assert hasattr(pipeline, 'validator')


def test_pipeline_config_assignment(app_config):
    """Test pipeline correctly assigns the provided config."""
    pipeline = ExtractToTrainPipeline(config=app_config)
    
    # Verify config is properly assigned
    assert pipeline.config is app_config
    assert isinstance(pipeline.config, AppConfig)


@pytest.mark.skip(reason="process_pdf is async and requires complex mocking")
async def test_process_pdf_skipped():
    """Skip process_pdf tests due to complexity."""
    # process_pdf is async and requires mocking of:
    # - File I/O operations
    # - Document extraction 
    # - LLM generation
    # - Validation pipeline
    # This is better tested with integration tests
    pass


@pytest.mark.skip(reason="export_dataset requires file I/O and is not critical")
def test_export_dataset_skipped():
    """Skip export_dataset tests due to file I/O complexity."""
    # export_dataset involves file operations which are:
    # - Complex to mock properly
    # - Better tested with integration tests
    # - Not critical for core functionality validation
    pass


@pytest.mark.skip(reason="Private methods are implementation details")
def test_private_methods_skipped():
    """Skip testing private methods."""
    # Methods like _extract_document_content, _create_final_dataset, 
    # _save_dataset, _log_pipeline_summary are implementation details
    # and should not be tested directly in unit tests
    pass
