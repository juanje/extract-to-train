"""
Test fixtures and sample data for extract-to-train tests.

This module provides reusable test data and utilities for testing
the extract-to-train functionality.
"""

from pathlib import Path

# Fixture file paths
FIXTURES_DIR = Path(__file__).parent
SAMPLE_CONFIG_PATH = FIXTURES_DIR / "sample_config.yaml"
SAMPLE_DATASET_PATH = FIXTURES_DIR / "sample_dataset.json"
SAMPLE_MARKDOWN_PATH = FIXTURES_DIR / "sample_markdown.md"


def get_fixture_path(filename: str) -> Path:
    """
    Get the absolute path to a fixture file.
    
    Args:
        filename: Name of the fixture file
        
    Returns:
        Absolute path to the fixture file
    """
    return FIXTURES_DIR / filename


def read_fixture_content(filename: str) -> str:
    """
    Read the content of a fixture file.
    
    Args:
        filename: Name of the fixture file
        
    Returns:
        Content of the fixture file as string
    """
    fixture_path = get_fixture_path(filename)
    return fixture_path.read_text(encoding="utf-8")
