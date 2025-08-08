"""
Tests unitarios para extract_to_train.models.dataset.

Tests para validación y funcionalidad de modelos de dataset usando pytest.
"""

import pytest
from pydantic import ValidationError

from extract_to_train.models.dataset import (
    Dataset,
    DatasetFormat,
    DatasetStats,
    DifficultyLevel,
    QAPair,
    QuestionType,
    ValidationResult,
)


def test_dataset_format_values():
    """Test valid DatasetFormat values."""
    assert DatasetFormat.ALPACA == "alpaca"
    assert DatasetFormat.SHAREGPT == "sharegpt"
    assert DatasetFormat.OPENAI == "openai"


def test_dataset_format_from_string():
    """Test DatasetFormat creation from string."""
    assert DatasetFormat("alpaca") == DatasetFormat.ALPACA
    assert DatasetFormat("sharegpt") == DatasetFormat.SHAREGPT
    assert DatasetFormat("openai") == DatasetFormat.OPENAI


def test_dataset_format_invalid():
    """Test invalid DatasetFormat value."""
    with pytest.raises(ValueError):
        DatasetFormat("invalid_format")


def test_dataset_format_list_all():
    """Test obtener todos los formatos disponibles."""
    formats = list(DatasetFormat)
    assert len(formats) == 3
    assert DatasetFormat.ALPACA in formats
    assert DatasetFormat.SHAREGPT in formats
    assert DatasetFormat.OPENAI in formats


def test_question_type_values():
    """Test valid QuestionType values."""
    assert QuestionType.FACTUAL == "factual"
    assert QuestionType.INFERENTIAL == "inferential"
    assert QuestionType.COMPARATIVE == "comparative"
    assert QuestionType.ANALYTICAL == "analytical"


def test_question_type_from_string():
    """Test QuestionType creation from string."""
    assert QuestionType("factual") == QuestionType.FACTUAL
    assert QuestionType("inferential") == QuestionType.INFERENTIAL
    assert QuestionType("comparative") == QuestionType.COMPARATIVE
    assert QuestionType("analytical") == QuestionType.ANALYTICAL


def test_question_type_invalid():
    """Test invalid QuestionType value."""
    with pytest.raises(ValueError):
        QuestionType("invalid_type")


def test_difficulty_level_values():
    """Test valid DifficultyLevel values."""
    assert DifficultyLevel.EASY == "easy"
    assert DifficultyLevel.MEDIUM == "medium"
    assert DifficultyLevel.HARD == "hard"


def test_difficulty_level_from_string():
    """Test DifficultyLevel creation from string."""
    assert DifficultyLevel("easy") == DifficultyLevel.EASY
    assert DifficultyLevel("medium") == DifficultyLevel.MEDIUM
    assert DifficultyLevel("hard") == DifficultyLevel.HARD


def test_difficulty_level_invalid():
    """Test invalid DifficultyLevel value."""
    with pytest.raises(ValueError):
        DifficultyLevel("invalid_level")


def test_qa_pair_creation_valid(sample_qa_pair):
    """Test creación de QAPair válido."""
    assert sample_qa_pair.id == "test-001"
    assert sample_qa_pair.question == "¿Qué es machine learning?"
    assert sample_qa_pair.difficulty == DifficultyLevel.MEDIUM
    assert sample_qa_pair.question_type == QuestionType.FACTUAL
    assert sample_qa_pair.confidence_score == 0.85
    assert sample_qa_pair.source_page == 1


def test_qa_pair_minimal_required_fields():
    """Test QAPair con campos mínimos requeridos."""
    qa_pair = QAPair(
        id="minimal-001",
        question="¿Pregunta mínima?",
        answer="Respuesta mínima",
        context="Contexto mínimo",
        difficulty=DifficultyLevel.EASY,
        question_type=QuestionType.FACTUAL,
        confidence_score=0.5
    )
    
    assert qa_pair.id == "minimal-001"
    assert qa_pair.source_page is None  # Campo opcional


def test_qa_pair_invalid_confidence_score_low():
    """Test QAPair con confidence_score muy bajo."""
    with pytest.raises(ValidationError) as exc_info:
        QAPair(
            id="test-001",
            question="¿Pregunta?",
            answer="Respuesta",
            context="Contexto",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.FACTUAL,
            confidence_score=-0.1  # Inválido
        )
    
    assert "greater than or equal to 0" in str(exc_info.value)


def test_qa_pair_invalid_confidence_score_high():
    """Test QAPair con confidence_score muy alto."""
    with pytest.raises(ValidationError) as exc_info:
        QAPair(
            id="test-001",
            question="¿Pregunta?",
            answer="Respuesta",
            context="Contexto",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.FACTUAL,
            confidence_score=1.1  # Inválido
        )
    
    assert "less than or equal to 1" in str(exc_info.value)


def test_qa_pair_empty_required_fields():
    """Test QAPair con campos requeridos vacíos."""
    with pytest.raises(ValidationError):
        QAPair(
            id="",  # ID vacío
            question="¿Pregunta?",
            answer="Respuesta",
            context="Contexto",
            difficulty=DifficultyLevel.EASY,
            question_type=QuestionType.FACTUAL,
            confidence_score=0.5
        )


def test_qa_pair_serialization(sample_qa_pair):
    """Test serialización de QAPair."""
    data = sample_qa_pair.model_dump()
    
    assert isinstance(data, dict)
    assert data["id"] == "test-001"
    assert data["difficulty"] == "medium"
    assert data["question_type"] == "factual"
    assert data["confidence_score"] == 0.85


def test_qa_pair_deserialization():
    """Test deserialización de QAPair."""
    data = {
        "id": "test-002",
        "question": "¿Otra pregunta?",
        "answer": "Otra respuesta",
        "context": "Otro contexto",
        "difficulty": "hard",
        "question_type": "analytical",
        "confidence_score": 0.9,
        "source_page": 2
    }
    
    qa_pair = QAPair.model_validate(data)
    assert qa_pair.id == "test-002"
    assert qa_pair.difficulty == DifficultyLevel.HARD
    assert qa_pair.question_type == QuestionType.ANALYTICAL


def test_validation_result_creation_valid(sample_validation_result):
    """Test creación de ValidationResult válido."""
    assert sample_validation_result.is_valid is True
    assert sample_validation_result.accuracy_score == 8
    assert sample_validation_result.completeness_score == 7
    assert sample_validation_result.clarity_score == 9
    assert sample_validation_result.training_value_score == 8
    assert len(sample_validation_result.issues) == 1
    assert len(sample_validation_result.suggestions) == 1


def test_validation_result_invalid_scores():
    """Test ValidationResult con scores inválidos."""
    with pytest.raises(ValidationError) as exc_info:
        ValidationResult(
            is_valid=True,
            accuracy_score=11,  # > 10
            completeness_score=5,
            clarity_score=5,
            training_value_score=5,
            issues=[],
            suggestions=[]
        )
    
    assert "less than or equal to 10" in str(exc_info.value)


def test_validation_result_negative_scores():
    """Test ValidationResult con scores negativos."""
    with pytest.raises(ValidationError) as exc_info:
        ValidationResult(
            is_valid=True,
            accuracy_score=-1,  # < 0
            completeness_score=5,
            clarity_score=5,
            training_value_score=5,
            issues=[],
            suggestions=[]
        )
    
    assert "greater than or equal to 0" in str(exc_info.value)


def test_validation_result_empty_lists():
    """Test ValidationResult con listas vacías (válido)."""
    result = ValidationResult(
        is_valid=True,
        accuracy_score=8,
        completeness_score=7,
        clarity_score=9,
        training_value_score=8,
        issues=[],  # Válido estar vacío
        suggestions=[]  # Válido estar vacío
    )
    
    assert result.issues == []
    assert result.suggestions == []


def test_validation_result_failed_validation():
    """Test ValidationResult para validación fallida."""
    result = ValidationResult(
        is_valid=False,
        accuracy_score=3,
        completeness_score=2,
        clarity_score=4,
        training_value_score=2,
        issues=["Respuesta incorrecta", "Información faltante"],
        suggestions=["Revisar fuente", "Añadir más detalles"]
    )
    
    assert result.is_valid is False
    assert len(result.issues) == 2
    assert len(result.suggestions) == 2


def test_dataset_stats_creation_valid(sample_dataset_stats):
    """Test creación de DatasetStats válido."""
    assert sample_dataset_stats.total_pairs == 3
    assert sample_dataset_stats.avg_question_length == 45
    assert sample_dataset_stats.avg_answer_length == 120
    assert sample_dataset_stats.pages_covered == 2
    assert sample_dataset_stats.estimated_tokens == 500


def test_dataset_stats_distributions(sample_dataset_stats):
    """Test distribuciones en DatasetStats."""
    difficulty_dist = sample_dataset_stats.difficulty_distribution
    question_type_dist = sample_dataset_stats.question_type_distribution
    
    assert difficulty_dist["easy"] == 1
    assert difficulty_dist["medium"] == 1
    assert difficulty_dist["hard"] == 1
    
    assert question_type_dist["factual"] == 1
    assert question_type_dist["comparative"] == 1
    assert question_type_dist["analytical"] == 1


def test_dataset_stats_zero_values():
    """Test DatasetStats con valores en cero."""
    stats = DatasetStats(
        total_pairs=0,
        avg_question_length=0,
        avg_answer_length=0,
        difficulty_distribution={},
        question_type_distribution={},
        pages_covered=0,
        estimated_tokens=0,
        avg_confidence_score=0.0,  # Campo requerido
        validation_pass_rate=0.0   # Campo requerido
    )
    
    assert stats.total_pairs == 0
    assert stats.difficulty_distribution == {}


def test_dataset_stats_invalid_negative_values():
    """Test DatasetStats con valores negativos inválidos."""
    with pytest.raises(ValidationError):
        DatasetStats(
            total_pairs=-1,  # Inválido
            avg_question_length=45,
            avg_answer_length=120,
            difficulty_distribution={},
            question_type_distribution={},
            pages_covered=2,
            estimated_tokens=500,
            avg_confidence_score=0.85,
            validation_pass_rate=0.9
        )


def test_dataset_stats_calculate_from_qa_pairs(sample_qa_pairs):
    """Test cálculo de estadísticas desde lista de QAPairs."""
    # Este test verifica si existe un método helper para calcular stats
    # Si no existe, puede ser una funcionalidad útil para implementar
    
    # Simulamos el cálculo manual
    total_pairs = len(sample_qa_pairs)
    avg_question_len = sum(len(pair.question) for pair in sample_qa_pairs) // total_pairs
    avg_answer_len = sum(len(pair.answer) for pair in sample_qa_pairs) // total_pairs
    
    difficulty_dist = {}
    question_type_dist = {}
    pages = set()
    
    for pair in sample_qa_pairs:
        # Contar dificultades
        diff_str = str(pair.difficulty)
        difficulty_dist[diff_str] = difficulty_dist.get(diff_str, 0) + 1
        
        # Contar tipos de pregunta
        type_str = str(pair.question_type)
        question_type_dist[type_str] = question_type_dist.get(type_str, 0) + 1
        
        # Contar páginas
        if pair.source_page:
            pages.add(pair.source_page)
    
    stats = DatasetStats(
        total_pairs=total_pairs,
        avg_question_length=avg_question_len,
        avg_answer_length=avg_answer_len,
        difficulty_distribution=difficulty_dist,
        question_type_distribution=question_type_dist,
        pages_covered=len(pages),
        estimated_tokens=total_pairs * 100,  # Estimación simple
        avg_confidence_score=0.85,
        validation_pass_rate=0.9
    )
    
    assert stats.total_pairs == 3
    assert stats.pages_covered == 2  # páginas 1 y 2


def test_dataset_creation_valid(sample_dataset):
    """Test creación de Dataset válido."""
    assert isinstance(sample_dataset.metadata, dict)
    assert len(sample_dataset.qa_pairs) == 3
    # Dataset no tiene validation_summary - campo removido del modelo
    assert isinstance(sample_dataset.stats, DatasetStats)


def test_dataset_empty_qa_pairs():
    """Test Dataset con lista vacía de QA pairs."""
    dataset = Dataset(
        metadata={"source": "test"},
        qa_pairs=[],
        stats=DatasetStats(
            total_pairs=0,
            avg_question_length=0,
            avg_answer_length=0,
            difficulty_distribution={},
            question_type_distribution={},
            pages_covered=0,
            estimated_tokens=0,
            avg_confidence_score=0.0,
            validation_pass_rate=0.0
        )
    )
    
    assert len(dataset.qa_pairs) == 0
    assert dataset.stats.total_pairs == 0


def test_dataset_serialization(sample_dataset):
    """Test serialización completa de Dataset."""
    data = sample_dataset.model_dump()
    
    assert isinstance(data, dict)
    assert "metadata" in data
    assert "qa_pairs" in data
    # Dataset no incluye validation_summary en la serialización
    assert "stats" in data
    
    # Verificar que los QA pairs se serializan correctamente
    assert len(data["qa_pairs"]) == 3
    assert data["qa_pairs"][0]["id"] == "test-001"


def test_dataset_deserialization(sample_dataset):
    """Test deserialización completa de Dataset."""
    # Serializar y deserializar
    data = sample_dataset.model_dump()
    restored_dataset = Dataset.model_validate(data)
    
    assert len(restored_dataset.qa_pairs) == len(sample_dataset.qa_pairs)
    assert restored_dataset.qa_pairs[0].id == sample_dataset.qa_pairs[0].id
    assert restored_dataset.stats.total_pairs == sample_dataset.stats.total_pairs


def test_dataset_validation_consistency(sample_dataset):
    """Test consistencia entre stats y QA pairs."""
    # Las estadísticas deberían ser consistentes con los QA pairs
    assert sample_dataset.stats.total_pairs == len(sample_dataset.qa_pairs)
    
    # Verificar que las distribuciones suman al total
    difficulty_total = sum(sample_dataset.stats.difficulty_distribution.values())
    question_type_total = sum(sample_dataset.stats.question_type_distribution.values())
    
    assert difficulty_total == sample_dataset.stats.total_pairs
    assert question_type_total == sample_dataset.stats.total_pairs


@pytest.mark.parametrize("format_str,expected_format", [
    ("alpaca", DatasetFormat.ALPACA),
    ("sharegpt", DatasetFormat.SHAREGPT),
    ("openai", DatasetFormat.OPENAI),
])
def test_dataset_format_conversions(format_str, expected_format):
    """Test conversiones de string a DatasetFormat."""
    assert DatasetFormat(format_str) == expected_format


@pytest.mark.parametrize("confidence_score", [0.0, 0.5, 1.0])
def test_qa_pair_valid_confidence_scores(confidence_score):
    """Test QAPair con scores de confianza válidos en los límites."""
    qa_pair = QAPair(
        id="test-confidence",
        question="¿Pregunta de test?",
        answer="Respuesta de test",
        context="Contexto de test",
        difficulty=DifficultyLevel.MEDIUM,
        question_type=QuestionType.FACTUAL,
        confidence_score=confidence_score
    )
    
    assert qa_pair.confidence_score == confidence_score


@pytest.mark.parametrize("score", [0, 5, 10])
def test_validation_result_valid_scores(score):
    """Test ValidationResult con scores válidos en los límites."""
    result = ValidationResult(
        is_valid=True,
        accuracy_score=score,
        completeness_score=score,
        clarity_score=score,
        training_value_score=score,
        issues=[],
        suggestions=[]
    )
    
    assert result.accuracy_score == score
    assert result.completeness_score == score


def test_qa_pair_with_none_source_page():
    """Test QAPair con source_page como None."""
    qa_pair = QAPair(
        id="test-none-page",
        question="¿Pregunta sin página?",
        answer="Respuesta sin página específica",
        context="Contexto general",
        difficulty=DifficultyLevel.EASY,
        question_type=QuestionType.FACTUAL,
        confidence_score=0.7,
        source_page=None
    )
    
    assert qa_pair.source_page is None


def test_dataset_metadata_flexibility():
    """Test que Dataset acepta metadata flexible."""
    custom_metadata = {
        "source_file": "test.pdf",
        "creation_date": "2024-01-01",
        "author": "Test Author",
        "version": "1.0",
        "custom_field": "custom_value"
    }
    
    dataset = Dataset(
        metadata=custom_metadata,
        qa_pairs=[],
        stats=DatasetStats(
            total_pairs=0,
            avg_question_length=0,
            avg_answer_length=0,
            difficulty_distribution={},
            question_type_distribution={},
            pages_covered=0,
            estimated_tokens=0,
            avg_confidence_score=0.0,
            validation_pass_rate=0.0
        )
    )
    
    assert dataset.metadata["custom_field"] == "custom_value"
    assert dataset.metadata["author"] == "Test Author"