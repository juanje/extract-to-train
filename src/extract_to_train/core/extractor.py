"""
Document extraction module supporting PDF and Markdown for educational dataset creation.

This module handles PDF document processing with docling and Markdown processing
with native parsers, featuring educational logging and structure preservation
optimized for Q&A generation.
"""

import logging
import re
from pathlib import Path
from typing import Any

import markdown
from docling.document_converter import DocumentConverter
from pydantic import BaseModel, computed_field

logger = logging.getLogger(__name__)


class ExtractedChunk(BaseModel):
    """
    A chunk of extracted content with metadata.

    Educational Note: Chunking preserves context while staying within
    model limits. Metadata helps track source information for quality control.
    """

    content: str
    page_number: int | None = None
    chunk_type: str = "text"  # text, table, header, etc.
    source_section: str | None = None

    @computed_field  # Pydantic v2 way
    @property
    def word_count(self) -> int:
        """Calculate word count automatically."""
        return len(self.content.split())

    @computed_field  # Pydantic v2 way
    @property
    def char_count(self) -> int:
        """Calculate character count automatically."""
        return len(self.content)


class DocumentMetadata(BaseModel):
    """
    Metadata about the extracted document.

    Educational Note: Document metadata helps understand the source
    and provides context for generated training data.
    """

    filename: str
    document_type: str  # "pdf" or "markdown"
    total_pages: int  # For PDF: actual pages, For Markdown: logical sections
    total_characters: int
    total_words: int
    extraction_method: str  # "docling" for PDF, "native" for Markdown
    language: str | None = None
    title: str | None = None
    author: str | None = None
    creation_date: str | None = None


class ExtractionResult(BaseModel):
    """
    Complete result from document extraction process.

    Contains both the extracted content chunks and metadata about
    the extraction process for educational transparency.
    Supports both PDF and Markdown documents.
    """

    chunks: list[ExtractedChunk]
    metadata: DocumentMetadata
    extraction_stats: dict[str, int]
    success: bool
    error_message: str | None = None


class DocumentExtractor:
    """
    Educational document extractor supporting PDF and Markdown with comprehensive logging.

    This extractor supports multiple document formats:
    - PDF: Uses docling for robust extraction with structure preservation
    - Markdown: Uses native parsing for clean text extraction

    Focuses on preserving document structure and providing educational insights
    into the extraction process.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the document extractor with educational features.

        Args:
            verbose: Enable detailed logging for educational purposes
        """
        self.verbose = verbose

        # Initialize PDF processor (docling) with default configuration for maximum compatibility
        self.pdf_converter = DocumentConverter()

        # Initialize Markdown processor
        self.markdown_processor = markdown.Markdown(
            extensions=["meta", "toc", "tables", "fenced_code"],
            extension_configs={"toc": {"marker": "[TOC]"}},
        )

        if self.verbose:
            logger.info("📄 Document Extractor initialized")
            logger.info("🔧 PDF support: docling with default configuration")
            logger.info("📝 Markdown support: native parser with extensions")

    def _detect_document_type(self, file_path: Path) -> str:
        """
        Detect document type based on file extension.

        Args:
            file_path: Path to the document

        Returns:
            Document type: 'pdf' or 'markdown'

        Raises:
            ValueError: If document type is not supported
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return "pdf"
        elif suffix in [".md", ".markdown"]:
            return "markdown"
        else:
            raise ValueError(
                f"Unsupported document type: {suffix}. Supported: .pdf, .md, .markdown"
            )

    def extract_from_document(
        self,
        document_path: Path,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        language: str | None = None,
    ) -> ExtractionResult:
        """
        Extract content from document (PDF or Markdown) with intelligent chunking.

        Args:
            document_path: Path to the document file
            chunk_size: Target size for content chunks
            chunk_overlap: Overlap between chunks for context preservation
            language: Override document language (e.g., 'es', 'en', 'fr'). Auto-detected if None.

        Returns:
            ExtractionResult with extracted content and metadata

        Educational Note: This unified interface handles both PDF and Markdown
        documents with appropriate processing for each format. Language specification
        helps optimize Q&A generation for non-English content.
        """
        if self.verbose:
            logger.info(f"📖 Starting extraction from: {document_path.name}")
            logger.info(f"✂️  Chunk size: {chunk_size} characters")
            logger.info(f"🔗 Chunk overlap: {chunk_overlap} characters")

        try:
            # Detect document type
            doc_type = self._detect_document_type(document_path)

            if doc_type == "pdf":
                return self.extract_from_pdf(
                    document_path, chunk_size, chunk_overlap, language
                )
            elif doc_type == "markdown":
                return self.extract_from_markdown(
                    document_path, chunk_size, chunk_overlap, language
                )
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")

        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")

            return ExtractionResult(
                chunks=[],
                metadata=DocumentMetadata(
                    filename=document_path.name,
                    document_type="unknown",
                    total_pages=0,
                    total_characters=0,
                    total_words=0,
                    extraction_method="failed",
                ),
                extraction_stats={},
                success=False,
                error_message=str(e),
            )

    def extract_from_pdf(
        self,
        pdf_path: Path,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        language: str | None = None,
    ) -> ExtractionResult:
        """
        Extract content from PDF with intelligent chunking.

        Args:
            pdf_path: Path to the PDF file
            chunk_size: Target size for content chunks
            chunk_overlap: Overlap between chunks for context preservation

        Returns:
            ExtractionResult with extracted content and metadata

        Educational Note: This method demonstrates how to preserve document
        structure while creating appropriately sized chunks for LLM processing.
        """
        if self.verbose:
            logger.info(f"📖 Starting extraction from: {pdf_path.name}")
            logger.info(f"✂️  Chunk size: {chunk_size} characters")
            logger.info(f"🔗 Chunk overlap: {chunk_overlap} characters")

        try:
            # Validate file exists and is readable
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            if not pdf_path.suffix.lower() == ".pdf":
                raise ValueError(f"File must be a PDF: {pdf_path}")

            # Extract document using docling
            if self.verbose:
                logger.info("🔄 Processing document with docling...")

            result = self.pdf_converter.convert(pdf_path)
            document = result.document

            if self.verbose:
                logger.info("✅ Document processed successfully")
                logger.info(f"📑 Pages: {len(document.pages)}")
                logger.info(f"📝 Elements: {len(document.texts)}")

            # Extract metadata
            metadata = self._extract_metadata(document, pdf_path, language)

            # Extract and chunk content
            chunks = self._extract_and_chunk_content(
                document, chunk_size, chunk_overlap
            )

            # Calculate extraction statistics
            stats = self._calculate_stats(chunks)

            if self.verbose:
                logger.info("📊 Extraction completed:")
                logger.info(f"  • {len(chunks)} chunks created")
                logger.info(f"  • {stats['total_characters']} characters extracted")
                logger.info(f"  • {stats['total_words']} words extracted")
                logger.info(
                    f"  • Average chunk size: {stats['avg_chunk_size']} characters"
                )

            return ExtractionResult(
                chunks=chunks, metadata=metadata, extraction_stats=stats, success=True
            )

        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")

            return ExtractionResult(
                chunks=[],
                metadata=DocumentMetadata(
                    filename=pdf_path.name,
                    document_type="pdf",
                    total_pages=0,
                    total_characters=0,
                    total_words=0,
                    extraction_method="docling",
                ),
                extraction_stats={},
                success=False,
                error_message=str(e),
            )

    def extract_from_markdown(
        self,
        markdown_path: Path,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        language: str | None = None,
    ) -> ExtractionResult:
        """
        Extract content from Markdown document with intelligent chunking.

        Args:
            markdown_path: Path to the Markdown file
            chunk_size: Target size for content chunks
            chunk_overlap: Overlap between chunks for context preservation

        Returns:
            ExtractionResult with extracted content and metadata

        Educational Note: Markdown processing preserves structure while
        converting to clean text suitable for Q&A generation.
        """
        if self.verbose:
            logger.info(f"📖 Starting extraction from: {markdown_path.name}")
            logger.info(f"✂️  Chunk size: {chunk_size} characters")
            logger.info(f"🔗 Chunk overlap: {chunk_overlap} characters")

        try:
            # Validate file exists and is readable
            if not markdown_path.exists():
                raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

            if markdown_path.suffix.lower() not in [".md", ".markdown"]:
                raise ValueError(f"File must be a Markdown file: {markdown_path}")

            # Read markdown content
            if self.verbose:
                logger.info("🔄 Processing Markdown document...")

            with open(markdown_path, encoding="utf-8") as f:
                markdown_content = f.read()

            # Process markdown to extract metadata and content
            self.markdown_processor.convert(markdown_content)

            # Get metadata from markdown processor
            md_meta = getattr(self.markdown_processor, "Meta", {})

            # Reset markdown processor for next use
            self.markdown_processor.reset()

            if self.verbose:
                logger.info("✅ Markdown processed successfully")
                logger.info(f"📝 Content length: {len(markdown_content)} characters")

            # Extract plain text from markdown (remove markdown formatting)
            plain_text = self._markdown_to_plain_text(markdown_content)

            # Extract metadata
            metadata = self._extract_markdown_metadata(
                markdown_path, plain_text, md_meta, language
            )

            # Create chunks from content
            chunks = self._create_markdown_chunks(
                plain_text, markdown_content, chunk_size, chunk_overlap
            )

            # Calculate extraction statistics
            stats = self._calculate_stats(chunks)

            if self.verbose:
                logger.info("📊 Extraction completed:")
                logger.info(f"  • {len(chunks)} chunks created")
                logger.info(f"  • {stats['total_characters']} characters extracted")
                logger.info(f"  • {stats['total_words']} words extracted")
                logger.info(
                    f"  • Average chunk size: {stats['avg_chunk_size']} characters"
                )

            return ExtractionResult(
                chunks=chunks, metadata=metadata, extraction_stats=stats, success=True
            )

        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")

            return ExtractionResult(
                chunks=[],
                metadata=DocumentMetadata(
                    filename=markdown_path.name,
                    document_type="markdown",
                    total_pages=0,
                    total_characters=0,
                    total_words=0,
                    extraction_method="native",
                ),
                extraction_stats={},
                success=False,
                error_message=str(e),
            )

    def _markdown_to_plain_text(self, markdown_content: str) -> str:
        """
        Convert markdown content to plain text while preserving structure.

        Args:
            markdown_content: Raw markdown content

        Returns:
            Clean plain text suitable for chunking

        Educational Note: This method removes markdown formatting while
        preserving the logical structure and readability of the content.
        """
        # Remove markdown formatting patterns while preserving structure
        text = markdown_content

        # Remove code blocks but preserve content
        text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("```", ""), text)
        text = re.sub(r"`([^`]+)`", r"\1", text)  # Inline code

        # Remove headers markup but keep content
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove emphasis markup
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Italic
        text = re.sub(r"_([^_]+)_", r"\1", text)  # Underscore italic

        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # Remove horizontal rules
        text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)

        # Clean up lists (keep content, remove markers)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

        # Clean up blockquotes
        text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

        # Normalize whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Multiple newlines to double
        text = re.sub(r" +", " ", text)  # Multiple spaces to single

        return text.strip()

    def _extract_markdown_metadata(
        self,
        markdown_path: Path,
        plain_text: str,
        md_meta: dict[str, Any],
        language: str | None = None,
    ) -> DocumentMetadata:
        """
        Extract metadata from markdown document.

        Args:
            markdown_path: Path to markdown file
            plain_text: Processed plain text
            md_meta: Metadata extracted from markdown frontmatter

        Returns:
            DocumentMetadata with extracted information
        """
        total_chars = len(plain_text)
        total_words = len(plain_text.split())

        # Count logical sections (headers) as "pages"
        section_count = len(re.findall(r"^#{1,6}\s+", plain_text, re.MULTILINE)) or 1

        # Extract title from metadata or first header
        title = None
        if md_meta and "title" in md_meta:
            title = (
                md_meta["title"][0]
                if isinstance(md_meta["title"], list)
                else md_meta["title"]
            )
        else:
            # Try to find first header as title
            first_header = re.search(r"^#{1,6}\s+(.+)$", plain_text, re.MULTILINE)
            if first_header:
                title = first_header.group(1).strip()

        # Extract author from metadata
        author = None
        if md_meta and "author" in md_meta:
            author = (
                md_meta["author"][0]
                if isinstance(md_meta["author"], list)
                else md_meta["author"]
            )

        # Determine final language: user override > frontmatter > default
        final_language = language or (
            md_meta.get("language", ["en"])[0] if md_meta.get("language") else "en"
        )

        metadata = DocumentMetadata(
            filename=markdown_path.name,
            document_type="markdown",
            total_pages=section_count,
            total_characters=total_chars,
            total_words=total_words,
            extraction_method="native",
            title=title,
            author=author,
            language=final_language,
        )

        if self.verbose:
            logger.info("📋 Document metadata extracted:")
            logger.info(f"  • Title: {metadata.title or 'Not specified'}")
            logger.info(f"  • Author: {metadata.author or 'Not specified'}")
            logger.info(f"  • Sections: {metadata.total_pages}")
            logger.info(f"  • Language: {metadata.language}")

        return metadata

    def _create_markdown_chunks(
        self,
        plain_text: str,
        original_markdown: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[ExtractedChunk]:
        """
        Create intelligent chunks from markdown content.

        Args:
            plain_text: Processed plain text
            original_markdown: Original markdown content for context
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of ExtractedChunk objects

        Educational Note: Markdown chunking respects section boundaries
        to maintain logical coherence in generated Q&A pairs.
        """
        # Try to split by sections first (headers)
        sections = re.split(r"\n(?=#{1,6}\s+)", original_markdown)

        chunks = []
        current_section = 1

        for section in sections:
            if not section.strip():
                continue

            # Convert section to plain text
            section_text = self._markdown_to_plain_text(section).strip()

            if len(section_text) < 50:  # Skip very short sections
                continue

            if len(section_text) <= chunk_size:
                # Section fits in one chunk
                chunks.append(
                    ExtractedChunk(
                        content=section_text,
                        page_number=current_section,
                        chunk_type="section",
                        source_section=section.split("\n")[0][:50] + "..."
                        if len(section.split("\n")[0]) > 50
                        else section.split("\n")[0],
                        word_count=len(section_text.split()),
                        char_count=len(section_text),
                    )
                )
            else:
                # Section needs to be split into multiple chunks
                section_chunks = self._create_chunks_from_text(
                    section_text, current_section, chunk_size, chunk_overlap
                )
                # Update chunk metadata for sections
                for chunk in section_chunks:
                    chunk.chunk_type = "section_part"
                    chunk.source_section = (
                        section.split("\n")[0][:50] + "..."
                        if len(section.split("\n")[0]) > 50
                        else section.split("\n")[0]
                    )

                chunks.extend(section_chunks)

            current_section += 1

        if self.verbose:
            logger.info(
                f"✂️  Created {len(chunks)} chunks from {current_section - 1} sections"
            )

        return chunks

    def _extract_metadata(
        self, document: Any, pdf_path: Path, language: str | None = None
    ) -> DocumentMetadata:
        """
        Extract metadata from the document.

        Args:
            document: Docling document object
            pdf_path: Path to the original PDF

        Returns:
            DocumentMetadata with extracted information
        """
        # Calculate total content statistics
        total_text = " ".join([text.text for text in document.texts])
        total_chars = len(total_text)
        total_words = len(total_text.split())

        # Extract document properties if available
        doc_info = getattr(document, "metadata", {}) or {}

        # Determine final language: user override > document metadata > default
        final_language = language or doc_info.get("language", "en")

        metadata = DocumentMetadata(
            filename=pdf_path.name,
            document_type="pdf",
            total_pages=len(document.pages),
            total_characters=total_chars,
            total_words=total_words,
            extraction_method="docling",
            title=doc_info.get("title"),
            author=doc_info.get("author"),
            creation_date=doc_info.get("creation_date"),
            language=final_language,
        )

        if self.verbose:
            logger.info("📋 Document metadata extracted:")
            logger.info(f"  • Title: {metadata.title or 'Not specified'}")
            logger.info(f"  • Author: {metadata.author or 'Not specified'}")
            logger.info(f"  • Pages: {metadata.total_pages}")
            logger.info(f"  • Language: {metadata.language}")

        return metadata

    def _extract_and_chunk_content(
        self, document: Any, chunk_size: int, chunk_overlap: int
    ) -> list[ExtractedChunk]:
        """
        Extract content and create intelligent chunks.

        Args:
            document: Docling document object
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks

        Returns:
            List of ExtractedChunk objects

        Educational Note: Intelligent chunking preserves context and
        ensures each chunk has sufficient information for Q&A generation.
        """
        chunks = []

        # Group content by pages for better context preservation
        for page_num, _page in enumerate(document.pages, 1):
            if self.verbose:
                logger.info(f"📄 Processing page {page_num}/{len(document.pages)}")

            # Extract text elements from the page
            page_texts = [
                text
                for text in document.texts
                if hasattr(text, "prov")
                and text.prov
                and any(p.page_no == page_num for p in text.prov)
            ]

            if not page_texts:
                continue

            # Combine text elements with structure preservation
            page_content = self._combine_page_content(page_texts)

            if len(page_content.strip()) < 50:  # Skip very short pages
                continue

            # Create chunks from page content
            page_chunks = self._create_chunks_from_text(
                page_content, page_num, chunk_size, chunk_overlap
            )

            chunks.extend(page_chunks)

        if self.verbose:
            logger.info(
                f"✂️  Created {len(chunks)} chunks from {len(document.pages)} pages"
            )

        return chunks

    def _combine_page_content(self, page_texts: list[Any]) -> str:
        """
        Combine text elements while preserving structure.

        Args:
            page_texts: List of text elements from a page

        Returns:
            Combined text with preserved structure

        Educational Note: Structure preservation helps maintain the logical
        flow of information, which is crucial for meaningful Q&A generation.
        """
        # Sort by reading order if available
        import contextlib

        with contextlib.suppress(AttributeError, IndexError):
            # Attempt to sort by position (top to bottom, left to right)
            page_texts.sort(
                key=lambda x: (
                    getattr(x, "bbox", [0, 0, 0, 0])[1],  # y-coordinate (top)
                    getattr(x, "bbox", [0, 0, 0, 0])[0],  # x-coordinate (left)
                )
            )

        # Combine texts with appropriate spacing
        combined_text = ""
        for text_element in page_texts:
            text_content = text_element.text.strip()
            if text_content:
                # Add appropriate spacing based on content type
                if combined_text and not combined_text.endswith("\n"):
                    # Add spacing between different text elements
                    if text_content[0].isupper() and combined_text[-1] not in ".!?":
                        combined_text += "\n\n"  # Likely new section/paragraph
                    else:
                        combined_text += " "  # Continuation

                combined_text += text_content

        return combined_text

    def _create_chunks_from_text(
        self, text: str, page_num: int, chunk_size: int, chunk_overlap: int
    ) -> list[ExtractedChunk]:
        """
        Create overlapping chunks from text while preserving sentence boundaries.

        Args:
            text: Text content to chunk
            page_num: Source page number
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of ExtractedChunk objects

        Educational Note: Sentence boundary preservation ensures chunks
        contain complete thoughts, improving Q&A generation quality.
        """
        if len(text) <= chunk_size:
            # Text is small enough to be a single chunk
            return [
                ExtractedChunk(
                    content=text,
                    page_number=page_num,
                    chunk_type="text",
                    word_count=len(text.split()),
                    char_count=len(text),
                )
            ]

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + chunk_size

            if end >= len(text):
                # Last chunk
                chunk_text = text[start:].strip()
            else:
                # Try to end at sentence boundary
                chunk_text = text[start:end]

                # Look for sentence endings within the last 100 characters
                sentence_ends = []
                for i, char in enumerate(chunk_text[-100:]):
                    if char in ".!?":
                        # Check if this is likely a sentence end
                        pos = len(chunk_text) - 100 + i
                        if pos + 1 < len(chunk_text) and chunk_text[pos + 1].isspace():
                            sentence_ends.append(pos + 1)

                if sentence_ends:
                    # Use the last sentence ending
                    chunk_text = chunk_text[: sentence_ends[-1]].strip()
                else:
                    # Fallback to word boundary
                    words = chunk_text.split()
                    if len(words) > 1:
                        chunk_text = " ".join(words[:-1])
                    chunk_text = chunk_text.strip()

            if chunk_text and len(chunk_text) > 50:  # Only include substantial chunks
                chunks.append(
                    ExtractedChunk(
                        content=chunk_text,
                        page_number=page_num,
                        chunk_type="text",
                        word_count=len(chunk_text.split()),
                        char_count=len(chunk_text),
                    )
                )

            # Calculate next start position with overlap
            if end >= len(text):
                break

            # Move start position (accounting for overlap)
            start = max(start + chunk_size - chunk_overlap, start + 1)

        return chunks

    def _calculate_stats(self, chunks: list[ExtractedChunk]) -> dict[str, int]:
        """
        Calculate extraction statistics.

        Args:
            chunks: List of extracted chunks

        Returns:
            Dictionary with extraction statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "total_words": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        total_chars = sum(chunk.char_count for chunk in chunks)
        total_words = sum(chunk.word_count for chunk in chunks)
        chunk_sizes = [chunk.char_count for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chunk_size": total_chars // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
        }

    def validate_document(self, document_path: Path) -> tuple[bool, str | None]:
        """
        Validate document file before extraction.

        Args:
            document_path: Path to document file (PDF or Markdown)

        Returns:
            Tuple of (is_valid, error_message)

        Educational Note: Validation prevents common issues and provides
        clear error messages for debugging.
        """
        try:
            if not document_path.exists():
                return False, f"File does not exist: {document_path}"

            if not document_path.is_file():
                return False, f"Path is not a file: {document_path}"

            # Check file extension
            suffix = document_path.suffix.lower()
            if suffix not in [".pdf", ".md", ".markdown"]:
                return (
                    False,
                    f"Unsupported file type: {suffix}. Supported: .pdf, .md, .markdown",
                )

            # Check file size (reasonable limits)
            file_size = document_path.stat().st_size
            max_size = 100 * 1024 * 1024  # 100MB limit

            if file_size == 0:
                return False, "Document file is empty"

            if file_size > max_size:
                return (
                    False,
                    f"Document file too large: {file_size / (1024 * 1024):.1f}MB > 100MB",
                )

            # Basic document validation - actual content validation happens during extraction
            # This avoids configuration issues during validation phase

            # Additional validation for Markdown files
            if suffix in [".md", ".markdown"]:
                try:
                    with open(document_path, encoding="utf-8") as f:
                        content = f.read(
                            1000
                        )  # Read first 1000 chars to check encoding
                    if not content.strip():
                        return False, "Markdown file appears to be empty"
                except UnicodeDecodeError:
                    return False, "Markdown file is not valid UTF-8"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"
