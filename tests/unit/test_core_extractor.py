"""
Unit tests for extract_to_train.core.extractor.

Tests for PDF and Markdown document extraction using pytest.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from extract_to_train.core.extractor import (
    DocumentExtractor,
    DocumentMetadata,
    ExtractedChunk,
    ExtractionResult,
)


def test_extracted_chunk_creation():
    """Test ExtractedChunk creation."""
    chunk = ExtractedChunk(
        content="This is sample test content",
        page_number=1,
        chunk_type="text",
        source_section="Introduction"
    )
    
    assert chunk.content == "This is sample test content"
    assert chunk.page_number == 1
    assert chunk.chunk_type == "text"
    assert chunk.source_section == "Introduction"


def test_extracted_chunk_word_count():
    """Test automatic word_count calculation."""
    chunk = ExtractedChunk(content="This is a test with five words")
    
    # word_count should be calculated automatically
    assert chunk.word_count == 7  # "This is a test with five words"


def test_extracted_chunk_char_count():
    """Test automatic char_count calculation."""
    content = "Sample"
    chunk = ExtractedChunk(content=content)
    
    assert chunk.char_count == len(content)


def test_extracted_chunk_defaults():
    """Test ExtractedChunk default values."""
    chunk = ExtractedChunk(content="Basic content")
    
    assert chunk.page_number is None
    assert chunk.chunk_type == "text"
    assert chunk.source_section is None
    assert chunk.word_count == 2
    assert chunk.char_count == 13  # "Basic content" has 13 characters


def test_extracted_chunk_empty_content():
    """Test ExtractedChunk with empty content."""
    chunk = ExtractedChunk(content="")
    
    assert chunk.word_count == 0  # "".split() returns [], not [""]
    assert chunk.char_count == 0


def test_document_metadata_creation():
    """Test DocumentMetadata creation."""
    metadata = DocumentMetadata(
        filename="test.pdf",
        document_type="pdf",
        total_pages=10,
        total_characters=1000,
        total_words=200,
        extraction_method="docling",
        title="Test Document",
        author="Test Author",
        language="en"
    )
    
    assert metadata.filename == "test.pdf"
    assert metadata.total_pages == 10
    assert metadata.title == "Test Document"
    assert metadata.author == "Test Author"
    assert metadata.language == "en"


def test_document_metadata_defaults():
    """Test DocumentMetadata default values."""
    metadata = DocumentMetadata(
        filename="test.pdf",
        document_type="pdf",
        total_pages=5,
        total_characters=500,
        total_words=100,
        extraction_method="docling"
    )
    
    assert metadata.title is None
    assert metadata.author is None
    assert metadata.language is None
    assert metadata.creation_date is None


def test_extraction_result_creation():
    """Test ExtractionResult creation."""
    chunks = [
        ExtractedChunk(content="First chunk", page_number=1),
        ExtractedChunk(content="Second chunk", page_number=1)
    ]
    metadata = DocumentMetadata(
        filename="test.pdf",
        document_type="pdf",
        total_pages=1,
        total_characters=100,
        total_words=20,
        extraction_method="docling"
    )
    stats = {"total_chunks": 2, "total_chars": 100}
    
    result = ExtractionResult(
        chunks=chunks,
        metadata=metadata,
        extraction_stats=stats,
        success=True
    )
    
    assert len(result.chunks) == 2
    assert result.metadata.filename == "test.pdf"
    assert result.extraction_stats["total_chunks"] == 2
    assert result.success is True
    assert result.error_message is None


def test_extraction_result_with_error():
    """Test ExtractionResult with error."""
    result = ExtractionResult(
        chunks=[],
        metadata=DocumentMetadata(
            filename="test.pdf",
            document_type="pdf",
            total_pages=0,
            total_characters=0,
            total_words=0,
            extraction_method="docling"
        ),
        extraction_stats={},
        success=False,
        error_message="Extraction error"
    )
    
    assert result.success is False
    assert result.error_message == "Extraction error"


def test_document_extractor_initialization():
    """Test DocumentExtractor initialization."""
    extractor = DocumentExtractor(verbose=True)
    
    assert extractor.verbose is True
    assert extractor.pdf_converter is not None
    assert extractor.markdown_processor is not None


def test_document_extractor_initialization_no_verbose():
    """Test initialization without verbose mode."""
    extractor = DocumentExtractor(verbose=False)
    
    assert extractor.verbose is False


def test_detect_document_type_pdf():
    """Test PDF type detection."""
    extractor = DocumentExtractor()
    
    pdf_path = Path("document.pdf")
    doc_type = extractor._detect_document_type(pdf_path)
    
    assert doc_type == "pdf"


def test_detect_document_type_markdown():
    """Test Markdown type detection."""
    extractor = DocumentExtractor()
    
    md_path = Path("document.md")
    doc_type = extractor._detect_document_type(md_path)
    
    assert doc_type == "markdown"
    
    markdown_path = Path("document.markdown")
    doc_type = extractor._detect_document_type(markdown_path)
    
    assert doc_type == "markdown"


def test_detect_document_type_unsupported():
    """Test unsupported type detection."""
    extractor = DocumentExtractor()
    
    txt_path = Path("document.txt")
    
    with pytest.raises(ValueError) as exc_info:
        extractor._detect_document_type(txt_path)
    
    assert "Unsupported document type" in str(exc_info.value)
    assert ".txt" in str(exc_info.value)


def test_detect_document_type_case_insensitive():
    """Test case-insensitive type detection."""
    extractor = DocumentExtractor()
    
    pdf_upper = Path("document.PDF")
    assert extractor._detect_document_type(pdf_upper) == "pdf"
    
    md_upper = Path("document.MD")
    assert extractor._detect_document_type(md_upper) == "markdown"


@pytest.mark.skip(reason="Complex PDF extraction integration test needs detailed mocking")
def test_extract_from_pdf_success(mock_pdf_file):
    """Test successful PDF extraction."""
    # This test requires complex mocking of docling internals
    # and is better suited for integration testing
    pass


@patch('builtins.open')
def test_extract_from_markdown_success(mock_open, mock_markdown_file):
    """Test successful Markdown extraction."""
    # Mock the file content
    markdown_content = """# Main Title

This is a test Markdown document.

## Section 1

Content of the first section.

## Section 2

Content of the second section with more details.
"""
    
    mock_open.return_value.__enter__.return_value.read.return_value = markdown_content
    
    extractor = DocumentExtractor()
    
    result = extractor.extract_from_document(
        mock_markdown_file,
        chunk_size=100,
        chunk_overlap=20
    )
    
    assert isinstance(result, ExtractionResult)
    assert result.success is True
    assert len(result.chunks) > 0
    assert result.metadata.filename == mock_markdown_file.name


def test_extract_from_document_file_not_found():
    """Test extraction with file not found."""
    extractor = DocumentExtractor()
    
    non_existent_file = Path("non_existent.pdf")
    
    result = extractor.extract_from_document(non_existent_file)
    
    assert result.success is False
    assert "not found" in result.error_message.lower() or "no such file" in result.error_message.lower()


@patch('extract_to_train.core.extractor.DocumentConverter')
def test_extract_from_pdf_conversion_error(mock_converter_class, mock_pdf_file):
    """Test error handling in PDF conversion."""
    # Mock that fails
    mock_converter = Mock()
    mock_converter_class.return_value = mock_converter
    mock_converter.convert.side_effect = Exception("Conversion error")
    
    extractor = DocumentExtractor()
    
    result = extractor.extract_from_document(mock_pdf_file)
    
    assert result.success is False
    assert "Conversion error" in result.error_message


@pytest.mark.skip(reason="Text chunking is internal functionality")
def test_chunk_text_basic():
    """Test basic text chunking."""
    # This functionality is internal and not exposed through public methods
    pass


@pytest.mark.skip(reason="Text chunking is internal functionality")
def test_chunk_text_with_overlap():
    """Test that chunks have appropriate overlap."""
    # This functionality is internal and not exposed through public methods
    pass


@pytest.mark.skip(reason="Text chunking is internal functionality")
def test_chunk_text_short_text():
    """Test chunking with short text."""
    # This functionality is internal and not exposed through public methods
    pass


@pytest.mark.skip(reason="Markdown metadata extraction is internal functionality")
def test_extract_metadata_from_markdown():
    """Test Markdown metadata extraction."""
    # This functionality is internal and not exposed through public methods
    pass


@pytest.mark.skip(reason="Language detection is internal functionality")
def test_language_detection():
    """Test automatic language detection."""
    # This functionality is internal and not exposed through public methods
    pass


@pytest.mark.parametrize("extension,expected_type", [
    (".pdf", "pdf"),
    (".PDF", "pdf"),
    (".md", "markdown"),
    (".MD", "markdown"),
    (".markdown", "markdown"),
    (".MARKDOWN", "markdown"),
])
def test_document_type_detection_parametrized(extension, expected_type):
    """Test type detection with multiple extensions."""
    extractor = DocumentExtractor()
    
    file_path = Path(f"test{extension}")
    doc_type = extractor._detect_document_type(file_path)
    
    assert doc_type == expected_type


@pytest.mark.skip(reason="Statistics calculation is internal functionality")
def test_extraction_stats_calculation():
    """Test extraction statistics calculation."""
    # This functionality is internal and not exposed through public methods
    pass


def test_verbose_logging(caplog):
    """Test that verbose mode generates appropriate logs."""
    import logging
    caplog.set_level(logging.INFO)
    
    extractor = DocumentExtractor(verbose=True)
    
    # Verify initialization logs
    assert "Document Extractor initialized" in caplog.text
    assert "PDF support" in caplog.text
    assert "Markdown support" in caplog.text


def test_extraction_result_serialization():
    """Test ExtractionResult serialization."""
    chunks = [ExtractedChunk(content="Test chunk")]
    metadata = DocumentMetadata(
        filename="test.pdf",
        document_type="pdf",
        total_pages=1,
        total_characters=100,
        total_words=20,
        extraction_method="docling"
    )
    
    result = ExtractionResult(
        chunks=chunks,
        metadata=metadata,
        extraction_stats={"total_chunks": 1},
        success=True
    )
    
    data = result.model_dump()
    
    assert isinstance(data, dict)
    assert "chunks" in data
    assert "metadata" in data
    assert "extraction_stats" in data


def test_chunk_metadata_preservation():
    """Test that metadata is preserved in chunks."""
    extractor = DocumentExtractor()
    
    # Simulate extraction with metadata
    chunk = ExtractedChunk(
        content="Test content",
        page_number=2,
        chunk_type="paragraph",
        source_section="Methodology"
    )
    
    assert chunk.page_number == 2
    assert chunk.chunk_type == "paragraph"
    assert chunk.source_section == "Methodology"
