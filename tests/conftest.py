"""
Configuración global de pytest y fixtures comunes para tests.

Este módulo proporciona fixtures reutilizables para todos los tests
del proyecto Extract-to-Train.
"""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pydantic import BaseModel

from extract_to_train.models.config import (
    AppConfig,
    GenerationConfig,
    ModelConfig,
    OllamaConfig,
    ProcessingConfig,
)
from extract_to_train.models.dataset import (
    Dataset,
    DatasetFormat,
    DatasetStats,
    DifficultyLevel,
    QAPair,
    QuestionType,
    ValidationResult,
)


@pytest.fixture
def sample_text() -> str:
    """Texto de muestra para tests."""
    return """
    Machine learning es una rama de la inteligencia artificial que permite a las computadoras
    aprender y mejorar automáticamente de la experiencia sin ser explícitamente programadas.
    
    Los algoritmos de machine learning construyen un modelo matemático basado en datos de
    entrenamiento para hacer predicciones o decisiones sin ser programados explícitamente
    para realizar la tarea.
    
    Hay tres tipos principales de machine learning:
    1. Aprendizaje supervisado
    2. Aprendizaje no supervisado  
    3. Aprendizaje por refuerzo
    """


@pytest.fixture
def sample_pdf_content() -> dict[str, Any]:
    """Mock extracted PDF content."""
    return {
        "text": "Este es el contenido de un PDF de prueba con información técnica.",
        "pages": [
            {
                "page_number": 1,
                "content": "Introducción al machine learning y sus aplicaciones.",
                "tables": [],
                "images": []
            },
            {
                "page_number": 2,
                "content": "Algoritmos de clasificación y regresión.",
                "tables": [{"data": "sample table data"}],
                "images": []
            }
        ],
        "metadata": {
            "title": "Documento de Prueba",
            "author": "Test Author",
            "pages": 2
        }
    }


@pytest.fixture
def sample_qa_pair() -> QAPair:
    """Par Q&A de muestra para tests."""
    return QAPair(
        id="test-001",
        question="¿Qué es machine learning?",
        answer="Machine learning es una rama de la inteligencia artificial que permite a las computadoras aprender automáticamente.",
        context="Machine learning es una rama de la inteligencia artificial...",
        difficulty=DifficultyLevel.MEDIUM,
        question_type=QuestionType.FACTUAL,
        confidence_score=0.85,
        source_page=1
    )


@pytest.fixture
def sample_qa_pairs(sample_qa_pair: QAPair) -> list[QAPair]:
    """Lista de pares Q&A para tests."""
    return [
        sample_qa_pair,
        QAPair(
            id="test-002",
            question="¿Cuáles son los tipos de machine learning?",
            answer="Los tres tipos principales son: supervisado, no supervisado y por refuerzo.",
            context="Hay tres tipos principales de machine learning...",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.COMPARATIVE,
            confidence_score=0.92,
            source_page=1
        ),
        QAPair(
            id="test-003",
            question="¿Cómo funcionan los algoritmos de clasificación?",
            answer="Los algoritmos de clasificación predicen categorías basándose en características de entrada.",
            context="Algoritmos de clasificación y regresión...",
            difficulty=DifficultyLevel.HARD,
            question_type=QuestionType.ANALYTICAL,
            confidence_score=0.78,
            source_page=2
        )
    ]


@pytest.fixture
def sample_validation_result() -> ValidationResult:
    """Sample validation result."""
    return ValidationResult(
        is_valid=True,
        accuracy_score=8,
        completeness_score=7,
        clarity_score=9,
        training_value_score=8,
        issues=["Respuesta podría ser más específica"],
        suggestions=["Añadir ejemplo concreto"]
    )


@pytest.fixture
def sample_dataset_stats() -> DatasetStats:
    """Sample dataset statistics."""
    return DatasetStats(
        total_pairs=3,
        avg_question_length=45,
        avg_answer_length=120,
        difficulty_distribution={"easy": 1, "medium": 1, "hard": 1},
        question_type_distribution={"factual": 1, "comparative": 1, "analytical": 1},
        pages_covered=2,
        estimated_tokens=500,
        avg_confidence_score=0.85,  # Campo requerido
        validation_pass_rate=0.9    # Campo requerido
    )


@pytest.fixture
def sample_dataset(
    sample_qa_pairs: list[QAPair],
    sample_validation_result: ValidationResult,
    sample_dataset_stats: DatasetStats
) -> Dataset:
    """Dataset completo de muestra."""
    return Dataset(
        metadata={
            "source_file": "test.pdf",
            "created_at": "2024-01-01T00:00:00Z",
            "format": "alpaca"
        },
        qa_pairs=sample_qa_pairs,
        validation_summary=sample_validation_result,
        stats=sample_dataset_stats
    )


@pytest.fixture
def ollama_config() -> OllamaConfig:
    """Ollama configuration for tests."""
    return OllamaConfig(
        host="http://localhost:11434",
        timeout=30,
        max_retries=3
    )


@pytest.fixture
def model_config() -> ModelConfig:
    """Model configuration for tests."""
    return ModelConfig(
        name="llama3.1:8b",
        temperature=0.3,
        context_window=4096,
        explanation="Configuración de prueba para tests unitarios"
    )


@pytest.fixture
def processing_config() -> ProcessingConfig:
    """Processing configuration for tests."""
    return ProcessingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        max_pairs_per_chunk=5
    )


@pytest.fixture
def generation_config() -> GenerationConfig:
    """Generation configuration for tests."""
    return GenerationConfig(
        question_types=["factual", "inferential", "comparative", "analytical"],
        difficulty_levels=["easy", "medium", "hard"],
        min_answer_length=50,
        max_answer_length=500,
        temperature_range=(0.1, 0.8)
    )


@pytest.fixture
def app_config(
    ollama_config: OllamaConfig,
    model_config: ModelConfig,
    processing_config: ProcessingConfig,
    generation_config: GenerationConfig
) -> AppConfig:
    """Complete application configuration for tests."""
    return AppConfig(
        ollama=ollama_config,
        models={
            "extraction": model_config,
            "validation": ModelConfig(
                name="deepseek-r1:8b",
                temperature=0.1,
                context_window=4096,
                explanation="Modelo de validación para tests"
            )
        },
        processing=processing_config,
        generation=generation_config
    )


@pytest.fixture
def mock_ollama_client():
    """Mock del cliente Ollama para tests."""
    client = Mock()
    
    # Mock métodos síncronos
    client.get_available_models.return_value = ["llama3.1:8b", "deepseek-r1:8b"]
    client.check_model_availability.return_value = True
    client.get_model_info.return_value = {
        "name": "llama3.1:8b",
        "size": "4.7GB",
        "digest": "abc123"
    }
    
    # Mock métodos asíncronos
    client.generate_response = AsyncMock()
    client.generate_response.return_value = "Respuesta generada de prueba"
    
    client.generate_structured_response = AsyncMock()
    client.generate_structured_response.return_value = {
        "question": "¿Qué es machine learning?",
        "answer": "Machine learning es una rama de la IA.",
        "difficulty": "medium",
        "question_type": "factual"
    }
    
    return client


@pytest.fixture
def mock_pdf_file(tmp_path: Path) -> Path:
    """Archivo PDF simulado para tests."""
    pdf_path = tmp_path / "test_document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n")
    return pdf_path


@pytest.fixture
def mock_markdown_file(tmp_path: Path) -> Path:
    """Archivo Markdown simulado para tests."""
    md_path = tmp_path / "test_document.md"
    md_path.write_text("""
# Título de Prueba

Este es un documento Markdown de prueba.

## Sección 1

Contenido de la primera sección.

## Sección 2

Contenido de la segunda sección con más detalles.
""")
    return md_path


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Directorio temporal para archivos de salida."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_jsonl_data() -> list[dict[str, Any]]:
    """Datos JSONL de muestra para tests."""
    return [
        {
            "instruction": "¿Qué es machine learning?",
            "input": "",
            "output": "Machine learning es una rama de la IA."
        },
        {
            "instruction": "Explica el concepto",
            "input": "aprendizaje supervisado",
            "output": "El aprendizaje supervisado usa datos etiquetados."
        }
    ]


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Respuesta simulada de LLM para tests."""
    return {
        "pairs": [
            {
                "question": "¿Qué es machine learning?",
                "answer": "Machine learning es una rama de la inteligencia artificial.",
                "difficulty": "medium",
                "question_type": "factual",
                "confidence": 0.85
            }
        ]
    }


@pytest.fixture
def mock_validation_response() -> dict[str, Any]:
    """Mock validation response for tests."""
    return {
        "is_valid": True,
        "accuracy_score": 8,
        "completeness_score": 7,
        "clarity_score": 9,
        "training_value_score": 8,
        "issues": ["Respuesta podría ser más específica"],
        "suggestions": ["Añadir ejemplo concreto"]
    }


@pytest.fixture(autouse=True)
def setup_logging():
    """Configurar logging para tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    yield
    # Cleanup después de cada test
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


@pytest.fixture
def mock_extraction_result() -> dict[str, Any]:
    """Mock extraction result."""
    return {
        "text": "Contenido extraído de documento",
        "chunks": [
            {
                "id": "chunk-1",
                "text": "Primer chunk de contenido",
                "page": 1,
                "start_char": 0,
                "end_char": 100
            },
            {
                "id": "chunk-2", 
                "text": "Segundo chunk de contenido",
                "page": 1,
                "start_char": 100,
                "end_char": 200
            }
        ],
        "metadata": {
            "pages": 2,
            "title": "Documento de Prueba"
        }
    }


# Useful decorators for tests
def async_test(coro):
    """Decorator for async tests."""
    import asyncio
    def wrapper():
        return asyncio.run(coro())
    return wrapper
