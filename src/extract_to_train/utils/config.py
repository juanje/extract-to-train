"""
Configuration utilities and helpers.

This module provides utilities for loading, validating, and managing
configuration settings with educational explanations.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ..models.config import AppConfig

# Load environment variables from .env file if it exists
load_dotenv()


def load_config_from_file(config_path: Path) -> AppConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        AppConfig instance

    Educational Note: File-based configuration allows users to save
    and share optimal settings for different use cases.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return AppConfig(**config_data)


def save_config_to_file(config: AppConfig, config_path: Path) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration to save
        config_path: Output file path
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.dict(), f, default_flow_style=False, indent=2)


def get_config_from_env() -> AppConfig:
    """
    Create configuration from environment variables.

    Returns:
        AppConfig with environment-based settings

    Educational Note: Environment variables provide a convenient way
    to configure settings in different deployment environments.
    """
    config = AppConfig.get_educational_config()

    # Ollama configuration
    if ollama_host := os.getenv("OLLAMA_HOST"):
        config.ollama.host = ollama_host

    # Model overrides
    if extract_model := os.getenv("EXTRACT_MODEL"):
        config.models["extraction"].name = extract_model

    if validate_model := os.getenv("VALIDATE_MODEL"):
        config.models["validation"].name = validate_model

    # Temperature overrides
    if extract_temp := os.getenv("EXTRACT_TEMPERATURE"):
        config.models["extraction"].temperature = float(extract_temp)

    if validate_temp := os.getenv("VALIDATE_TEMPERATURE"):
        config.models["validation"].temperature = float(validate_temp)

    # Processing overrides
    if chunk_size := os.getenv("CHUNK_SIZE"):
        config.processing.chunk_size = int(chunk_size)

    if chunk_overlap := os.getenv("CHUNK_OVERLAP"):
        config.processing.chunk_overlap = int(chunk_overlap)

    return config


def validate_config_compatibility(config: AppConfig) -> dict[str, str]:
    """
    Validate configuration for common compatibility issues.

    Args:
        config: Configuration to validate

    Returns:
        Dictionary of validation warnings/errors

    Educational Note: Configuration validation helps prevent common
    setup issues and provides guidance for optimal settings.
    """
    issues = {}

    # Check chunk overlap vs size
    if config.processing.chunk_overlap >= config.processing.chunk_size:
        issues["chunk_overlap"] = (
            f"Chunk overlap ({config.processing.chunk_overlap}) should be less than "
            f"chunk size ({config.processing.chunk_size})"
        )

    # Check temperature ranges
    if config.models["extraction"].temperature > 1.0:
        issues["extract_temperature"] = (
            f"High temperature ({config.models['extraction'].temperature}) may produce "
            "inconsistent results for dataset generation"
        )

    if config.models["validation"].temperature > 0.3:
        issues["validate_temperature"] = (
            f"High temperature ({config.models['validation'].temperature}) may reduce "
            "validation consistency"
        )

    # Check model compatibility
    if config.models["extraction"].name == config.models["validation"].name and (
        config.models["extraction"].temperature
        == config.models["validation"].temperature
    ):
        issues["model_separation"] = (
            "Using identical models and temperatures for generation and validation "
            "may reduce validation objectivity"
        )

    return issues


def get_recommended_configs() -> dict[str, AppConfig]:
    """
    Get pre-configured setups for different use cases.

    Returns:
        Dictionary of named configurations

    Educational Note: Pre-configured setups help users understand
    optimal settings for different scenarios and use cases.
    """
    configs = {}

    # Educational/learning configuration
    configs["educational"] = AppConfig.get_educational_config()

    # Speed-optimized configuration
    configs["speed"] = AppConfig.get_speed_optimized_config()

    # Quality-focused configuration
    quality_config = AppConfig.get_educational_config()
    quality_config.models["extraction"].temperature = 0.2  # Lower for consistency
    quality_config.validation.min_confidence_score = 0.8  # Higher bar
    quality_config.validation.required_scores = {
        "accuracy": 8,
        "completeness": 8,
        "clarity": 7,
        "training_value": 8,
    }
    configs["quality"] = quality_config

    # Experimental configuration
    experimental_config = AppConfig.get_educational_config()
    experimental_config.models["extraction"].temperature = 0.5  # Higher creativity
    experimental_config.processing.max_pairs_per_chunk = 7  # More pairs
    experimental_config.validation.min_confidence_score = (
        0.6  # Lower bar for experimentation
    )
    configs["experimental"] = experimental_config

    return configs
