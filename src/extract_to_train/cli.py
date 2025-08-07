"""
Educational CLI interface for the extract-to-train tool.

This module provides a comprehensive command-line interface with educational
features, detailed help, and transparent processing insights.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Lazy imports - moved to functions to improve CLI startup time
# from .core.pipeline import ExtractToTrainPipeline  # -> moved to functions
# from .models.config import AppConfig               # -> moved to functions
# from .models.dataset import DatasetFormat          # -> moved to functions

# Initialize rich console for beautiful output
console = Console()
app = typer.Typer(
    name="extract-to-train",
    help="📚 Extract information from PDFs to create training datasets for LLM fine-tuning",
    add_completion=False,
    rich_markup_mode="rich",
)


def setup_logging(verbose: bool = False, log_level: str = "INFO") -> None:
    """
    Setup educational logging with rich formatting.

    Args:
        verbose: Enable verbose educational logging
        log_level: Logging level
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure rich logging handler
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    # Adjust specific loggers
    if not verbose:
        # Reduce noise from external libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


@app.command()
def extract(
    document_path: Path = typer.Argument(
        ...,
        help="Path to the document file to process (PDF or Markdown)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        "dataset.jsonl",
        "--output",
        "-o",
        help="Output file path for the generated dataset",
    ),
    format_type: str = typer.Option(
        "alpaca",
        "--format",
        "-f",
        help="Output format for fine-tuning (alpaca, sharegpt, openai)",
    ),
    max_pairs: int | None = typer.Option(
        None, "--max-pairs", "-n", help="Maximum number of Q&A pairs to generate", min=1
    ),
    max_chunks: int | None = typer.Option(
        None,
        "--max-chunks",
        help="Maximum number of chunks/pages to process (useful for testing large documents)",
        min=1,
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        help="Text chunk size in characters (default: from config/env)",
        min=100,
        max=5000,
    ),
    chunk_overlap: int | None = typer.Option(
        None,
        "--chunk-overlap",
        help="Overlap between chunks in characters (default: from config/env)",
        min=0,
        max=1000,
    ),
    model_extract: str = typer.Option(
        "llama3.1:8B", "--model-extract", help="Model for Q&A generation"
    ),
    model_validate: str = typer.Option(
        "deepseek-r1:8B", "--model-validate", help="Model for dataset validation"
    ),
    temperature_extract: float = typer.Option(
        0.3,
        "--temperature-extract",
        help="Temperature for generation model",
        min=0.0,
        max=2.0,
    ),
    temperature_validate: float = typer.Option(
        0.1,
        "--temperature-validate",
        help="Temperature for validation model",
        min=0.0,
        max=1.0,
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip dataset validation step for faster processing",
    ),
    enable_auto_correction: bool = typer.Option(
        False, "--auto-correct", help="Enable automatic correction of failed Q&A pairs"
    ),
    include_context: bool = typer.Option(
        True,
        "--include-context/--no-include-context",
        help="Include source context in instruction formats",
    ),
    system_prompt: str | None = typer.Option(
        None, "--system-prompt", help="Custom system prompt for conversational formats"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed educational logging"
    ),
    explain: bool = typer.Option(
        False, "--explain", help="Show detailed explanations of each processing step"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Document language (e.g., 'en', 'es', 'fr'). Auto-detected if not specified",
    ),
    progress_file: str | None = typer.Option(
        None,
        "--progress-file",
        help="File to save progressive Q&A pairs (default: auto-generated .progress.jsonl)",
    ),
) -> None:
    """
    🚀 Extract information from documents and create fine-tuning datasets.

    This command processes PDF or Markdown documents through the complete pipeline:
    1. Extract structured content using docling (PDF) or native parsing (Markdown)
    2. Generate diverse Q&A pairs using local LLMs
    3. Validate dataset quality with reasoning models
    4. Export in formats ready for fine-tuning

    [bold green]Examples:[/bold green]

    [dim]# Basic usage with educational output[/dim]
    extract-to-train extract document.pdf --verbose
    extract-to-train extract guide.md --verbose

    [dim]# Generate Alpaca format for Axolotl/Unsloth[/dim]
    extract-to-train extract document.pdf --format alpaca --max-pairs 100

    [dim]# Test with limited chunks from large document[/dim]
    extract-to-train extract large_document.pdf --max-chunks 10 --verbose

    [dim]# Use custom models and settings[/dim]
    extract-to-train extract tutorial.md --model-extract mistral:7B --temperature-extract 0.4

    [dim]# Process with progress saving and language specification[/dim]
    extract-to-train extract documento.pdf --language es --progress-file progress.jsonl

    [dim]# Skip validation for faster processing[/dim]
    extract-to-train extract document.pdf --skip-validation --max-pairs 50
    """
    # Lazy imports to improve CLI startup time
    from .models.dataset import DatasetFormat

    # Setup logging based on options
    setup_logging(verbose=verbose or explain, log_level=log_level)

    try:
        # Convert string format to enum
        try:
            dataset_format = DatasetFormat(format_type.lower())
        except ValueError:
            console.print(
                f"❌ Invalid format type: {format_type}. Valid options: alpaca, sharegpt, openai",
                style="red",
            )
            sys.exit(1)

        # Create custom configuration
        config = _create_custom_config(
            model_extract=model_extract,
            model_validate=model_validate,
            temperature_extract=temperature_extract,
            temperature_validate=temperature_validate,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if explain:
            _show_configuration_explanation(config, dataset_format)

        # Run the extraction pipeline
        asyncio.run(
            _run_extraction_pipeline(
                document_path=document_path,
                output_path=output,
                config=config,
                format_type=dataset_format,
                max_pairs=max_pairs,
                max_chunks=max_chunks,
                skip_validation=skip_validation,
                enable_auto_correction=enable_auto_correction,
                include_context=include_context,
                system_prompt=system_prompt,
                language=language,
                progress_file=progress_file,
                verbose=verbose or explain,
            )
        )

    except KeyboardInterrupt:
        console.print("\n❌ Process interrupted by user", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ Error: {str(e)}", style="red")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def analyze(
    dataset_path: Path = typer.Argument(
        ...,
        help="Path to the dataset file to analyze",
        exists=True,
        file_okay=True,
        readable=True,
    ),
    detailed: bool = typer.Option(
        False, "--detailed", help="Show detailed analysis with recommendations"
    ),
    export_stats: Path | None = typer.Option(
        None, "--export-stats", help="Export statistics to JSON file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed logging"
    ),
) -> None:
    """
    📊 Analyze a generated dataset and provide educational insights.

    This command provides comprehensive analysis of dataset quality,
    diversity, and suitability for fine-tuning applications.

    [bold green]Examples:[/bold green]

    [dim]# Basic analysis[/dim]
    extract-to-train analyze dataset.jsonl

    [dim]# Detailed analysis with recommendations[/dim]
    extract-to-train analyze dataset.jsonl --detailed

    [dim]# Export statistics for further analysis[/dim]
    extract-to-train analyze dataset.jsonl --export-stats stats.json
    """
    setup_logging(verbose=verbose)

    try:
        _analyze_dataset(dataset_path, detailed, export_stats)
    except Exception as e:
        console.print(f"❌ Analysis failed: {str(e)}", style="red")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def setup() -> None:
    """
    🔧 Verify and setup the environment for optimal learning.

    This command checks your local setup and provides educational guidance
    for configuring the extract-to-train environment.
    """
    setup_logging(verbose=True)

    console.print("🔧 [bold blue]Extract-to-Train Environment Setup[/bold blue]")
    console.print()

    asyncio.run(_verify_environment_setup())


@app.command()
def config(
    show_current: bool = typer.Option(
        False, "--show-current", help="Show current configuration"
    ),
    show_explanations: bool = typer.Option(
        False, "--explain", help="Show detailed explanations of configuration options"
    ),
    export_template: Path | None = typer.Option(
        None, "--export-template", help="Export configuration template to file"
    ),
) -> None:
    """
    ⚙️ Display and manage configuration options.

    This command helps understand and customize the configuration
    for optimal dataset generation results.
    """
    if show_explanations:
        _show_configuration_explanations()

    if show_current:
        _show_current_configuration()

    if export_template:
        _export_configuration_template(export_template)


async def _run_extraction_pipeline(
    document_path: Path,
    output_path: Path,
    config: Any,  # AppConfig - lazy import
    format_type: Any,  # DatasetFormat - lazy import
    max_pairs: int | None,
    max_chunks: int | None,
    skip_validation: bool,
    enable_auto_correction: bool,
    include_context: bool,
    system_prompt: str | None,
    language: str | None,
    progress_file: str | None,
    verbose: bool,
) -> None:
    """Run the complete extraction pipeline with progress tracking."""
    # Lazy imports to improve CLI startup time
    from .core.pipeline import ExtractToTrainPipeline
    from .models.dataset import DatasetFormat

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Initialize pipeline
        task = progress.add_task("Initializing pipeline...", total=None)
        pipeline = ExtractToTrainPipeline(config, verbose=verbose)

        # Set document language if specified
        if language:
            pipeline.set_document_language(language)

        # Setup progress file
        if progress_file is None:
            # Auto-generate progress file name
            base_name = document_path.stem
            progress_file = f".{base_name}_progress.jsonl"

        if verbose:
            console.print(f"📝 Progress will be saved to: {progress_file}", style="dim")

        # Test components if verbose
        if verbose:
            progress.update(task, description="Testing pipeline components...")
            test_results = await pipeline.test_pipeline_components()

            if not all(test_results.values()):
                failed_components = [k for k, v in test_results.items() if not v]
                raise RuntimeError(
                    f"Component tests failed: {', '.join(failed_components)}"
                )

        # Process document
        progress.update(task, description="Processing document...")

        dataset, stats = await pipeline.process_pdf(
            pdf_path=document_path,
            max_pairs=max_pairs,
            max_chunks=max_chunks,
            skip_validation=skip_validation,
            enable_auto_correction=enable_auto_correction,
            progress_file=progress_file,
        )

        # Export dataset
        progress.update(task, description="Exporting dataset...")

        format_options: dict[str, Any] = {}
        if format_type == DatasetFormat.ALPACA:
            format_options["include_context"] = include_context
        elif format_type == DatasetFormat.OPENAI and system_prompt is not None:
            format_options["system_prompt"] = system_prompt

        pipeline.export_dataset(
            dataset=dataset,
            output_path=output_path,
            format_type=format_type,
            **format_options,
        )

        progress.update(task, description="Complete!", completed=True)

    # Show results summary
    _show_results_summary(dataset, stats, output_path, format_type)


def _create_custom_config(
    model_extract: str,
    model_validate: str,
    temperature_extract: float,
    temperature_validate: float,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> Any:  # AppConfig return type - lazy import
    """Create configuration with custom parameters."""
    # Lazy imports to improve CLI startup time
    from extract_to_train.utils.config import get_config_from_env

    config = get_config_from_env()

    # Update model configurations
    config.models["extraction"].name = model_extract
    config.models["extraction"].temperature = temperature_extract
    config.models["validation"].name = model_validate
    config.models["validation"].temperature = temperature_validate

    # Update processing configuration (only if explicitly provided)
    if chunk_size is not None:
        config.processing.chunk_size = chunk_size
    if chunk_overlap is not None:
        config.processing.chunk_overlap = chunk_overlap

    return config


def _show_configuration_explanation(config: Any, format_type: Any) -> None:
    """Show detailed explanation of the current configuration."""
    # Lazy imports to improve CLI startup time
    from .models.dataset import DatasetFormat

    console.print("🎓 [bold blue]Configuration Explanation[/bold blue]")
    console.print()

    # Model explanations
    console.print("🤖 [bold]Model Configuration:[/bold]")
    console.print(f"📝 Generation: {config.models['extraction'].name}")
    console.print(f"   {config.models['extraction'].explanation}")
    console.print()
    console.print(f"🔍 Validation: {config.models['validation'].name}")
    console.print(f"   {config.models['validation'].explanation}")
    console.print()

    # Format explanation
    console.print(f"📊 [bold]Output Format: {format_type}[/bold]")
    format_explanations = {
        DatasetFormat.ALPACA: "Universal instruction format compatible with Axolotl, Unsloth, and HuggingFace+PEFT",
        DatasetFormat.SHAREGPT: "Conversational format ideal for chat model fine-tuning with Axolotl and Unsloth",
        DatasetFormat.OPENAI: "API-compatible format for HuggingFace Transformers + PEFT workflows",
    }
    console.print(f"   {format_explanations[format_type]}")
    console.print()


def _show_results_summary(
    dataset: Any, stats: Any, output_path: Path, format_type: Any
) -> None:
    """Show a beautiful summary of processing results."""
    console.print("\n🎉 [bold green]Processing Complete![/bold green]")
    console.print()

    # Create results table
    table = Table(
        title="Dataset Generation Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Notes", style="dim")

    table.add_row(
        "Source Document", stats.source_filename, f"{stats.source_pages} pages"
    )
    table.add_row(
        "Processing Time", f"{stats.total_execution_time:.1f}s", "Total pipeline time"
    )
    table.add_row(
        "Q&A Pairs Generated",
        str(len(dataset.qa_pairs)),
        f"{stats.final_pass_rate:.1%} pass rate",
    )
    table.add_row("Output Format", format_type.value, "Ready for fine-tuning")
    table.add_row(
        "Output File", str(output_path), f"{output_path.stat().st_size / 1024:.1f} KB"
    )
    table.add_row(
        "Estimated Tokens",
        f"~{dataset.stats.estimated_tokens:,}",
        "For cost estimation",
    )

    console.print(table)
    console.print()

    # Quality indicators
    console.print("📈 [bold]Quality Indicators:[/bold]")
    console.print(f"   • Average confidence: {dataset.stats.avg_confidence_score:.2f}")
    console.print(
        f"   • Question diversity: {len(dataset.stats.question_type_distribution)} types"
    )
    console.print(
        f"   • Difficulty balance: {len(dataset.stats.difficulty_distribution)} levels"
    )
    console.print()

    # Next steps
    console.print("🚀 [bold]Next Steps:[/bold]")
    console.print(f"   1. Review your dataset: [cyan]{output_path}[/cyan]")
    console.print(
        f"   2. Analyze quality: [cyan]extract-to-train analyze {output_path} --detailed[/cyan]"
    )
    console.print("   3. Use for fine-tuning with your preferred framework")
    console.print()


def _analyze_dataset(
    dataset_path: Path, detailed: bool, export_stats: Path | None
) -> None:
    """Analyze a dataset file and show insights."""
    console.print(f"📊 [bold blue]Analyzing Dataset: {dataset_path.name}[/bold blue]")
    console.print()

    try:
        # Load dataset
        if dataset_path.suffix == ".jsonl":
            # Load JSONL format
            entries = []
            with open(dataset_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        else:
            # Load JSON format
            with open(dataset_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "qa_pairs" in data:
                    entries = data["qa_pairs"]
                elif isinstance(data, list):
                    entries = data
                else:
                    entries = [data]

        # Analyze entries
        console.print(f"📈 Found {len(entries)} entries")

        # Basic statistics
        if detailed:
            _show_detailed_analysis(entries)
        else:
            _show_basic_analysis(entries)

        # Export statistics if requested
        if export_stats:
            stats = _calculate_analysis_stats(entries)
            with open(export_stats, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            console.print(f"📁 Statistics exported to: {export_stats}")

    except Exception as e:
        raise RuntimeError(f"Failed to analyze dataset: {str(e)}") from e


def _show_basic_analysis(entries: list[Any]) -> None:
    """Show basic dataset analysis."""
    if not entries:
        console.print("❌ No entries found in dataset")
        return

    # Calculate basic metrics
    len(entries)

    # Try to extract text lengths
    question_lengths = []
    answer_lengths = []

    for entry in entries:
        if isinstance(entry, dict):
            # Handle different formats
            question = entry.get("instruction") or entry.get("question") or ""
            answer = entry.get("output") or entry.get("answer") or ""

            if "conversations" in entry:  # ShareGPT format
                for conv in entry["conversations"]:
                    if conv.get("from") == "human":
                        question = conv.get("value", "")
                    elif conv.get("from") == "gpt":
                        answer = conv.get("value", "")

            if "messages" in entry:  # OpenAI format
                for msg in entry["messages"]:
                    if msg.get("role") == "user":
                        question = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        answer = msg.get("content", "")

            question_lengths.append(len(question))
            answer_lengths.append(len(answer))

    if question_lengths and answer_lengths:
        avg_q_len = sum(question_lengths) / len(question_lengths)
        avg_a_len = sum(answer_lengths) / len(answer_lengths)

        console.print(f"📏 Average question length: {avg_q_len:.0f} characters")
        console.print(f"📏 Average answer length: {avg_a_len:.0f} characters")
        console.print(
            f"🎯 Total estimated tokens: ~{(sum(question_lengths) + sum(answer_lengths)) // 4:,}"
        )


def _show_detailed_analysis(entries: list[Any]) -> None:
    """Show detailed dataset analysis with recommendations."""
    console.print("🔍 [bold]Detailed Analysis[/bold]")

    _show_basic_analysis(entries)

    # Add more detailed metrics here
    console.print("\n💡 [bold]Recommendations:[/bold]")
    console.print("   • Review question diversity for balanced training")
    console.print("   • Check answer quality and consistency")
    console.print("   • Validate format compatibility with your fine-tuning framework")


def _calculate_analysis_stats(entries: list[Any]) -> dict[str, Any]:
    """Calculate comprehensive statistics for export."""
    return {
        "total_entries": len(entries),
        "analysis_timestamp": str(Path().cwd()),
        "format_detected": "mixed",  # Could be enhanced to detect format
    }


async def _verify_environment_setup() -> None:
    """Verify and guide environment setup."""
    # Lazy imports to improve CLI startup time
    from .core.pipeline import ExtractToTrainPipeline
    from .models.config import AppConfig

    console.print("🔍 Checking Ollama connection...")

    try:
        # Test basic setup
        config = AppConfig.get_educational_config()
        pipeline = ExtractToTrainPipeline(config, verbose=True)

        test_results = await pipeline.test_pipeline_components()

        # Show results
        table = Table(title="Environment Status", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Notes")

        for component, status in test_results.items():
            status_icon = "✅" if status else "❌"
            table.add_row(component.replace("_", " ").title(), status_icon, "")

        console.print(table)

        if all(test_results.values()):
            console.print("\n🎉 [bold green]Environment setup complete![/bold green]")
            console.print("You're ready to process PDFs and create training datasets.")
        else:
            console.print("\n⚠️  [bold yellow]Setup issues detected[/bold yellow]")
            console.print("Please check the failed components above.")

    except Exception as e:
        console.print(f"\n❌ Setup verification failed: {str(e)}")


def _show_configuration_explanations() -> None:
    """Show detailed explanations of all configuration options."""
    console.print("⚙️ [bold blue]Configuration Options Explained[/bold blue]")
    console.print()

    console.print(
        "ℹ️ [bold yellow]Note:[/bold yellow] These options are used with the [cyan]extract[/cyan] command, not [cyan]config[/cyan]"
    )
    console.print(
        "📝 Example: [dim]extract-to-train extract document.pdf --model-extract llama3.1:8B[/dim]"
    )
    console.print()

    # Model configurations
    console.print("🤖 [bold]Model Settings (for 'extract' command):[/bold]")
    console.print("   • [cyan]--model-extract[/cyan]: LLM for generating Q&A pairs")
    console.print("     Recommended: llama3.1:8B (best instruction following)")
    console.print("   • [cyan]--model-validate[/cyan]: LLM for quality validation")
    console.print("     Recommended: deepseek-r1:8B (superior reasoning)")
    console.print()

    # Temperature settings
    console.print("🌡️ [bold]Temperature Settings (for 'extract' command):[/bold]")
    console.print(
        "   • [cyan]--temperature-extract[/cyan]: Controls creativity (0.3 recommended)"
    )
    console.print("     Lower = more consistent, Higher = more creative")
    console.print(
        "   • [cyan]--temperature-validate[/cyan]: Controls validation strictness (0.1 recommended)"
    )
    console.print("     Lower = more consistent validation")
    console.print()

    # Processing settings
    console.print("📄 [bold]Processing Settings (for 'extract' command):[/bold]")
    console.print(
        "   • [cyan]--chunk-size[/cyan]: Text chunk size in characters (default: 512)"
    )
    console.print(
        "   • [cyan]--chunk-overlap[/cyan]: Overlap between chunks (default: 100)"
    )
    console.print("   • [cyan]--max-pairs[/cyan]: Maximum Q&A pairs to generate")
    console.print(
        "   • [cyan]--max-chunks[/cyan]: Maximum chunks to process (for testing)"
    )
    console.print()

    # Format options
    console.print("📊 [bold]Output Formats (for 'extract' command):[/bold]")
    console.print(
        "   • [cyan]--format alpaca[/cyan]: Universal format (Axolotl, Unsloth, HF+PEFT)"
    )
    console.print(
        "   • [cyan]--format sharegpt[/cyan]: Conversational format (Axolotl, Unsloth)"
    )
    console.print("   • [cyan]--format openai[/cyan]: API format (HuggingFace + PEFT)")
    console.print()

    # Environment variables
    console.print("🌍 [bold]Environment Variables:[/bold]")
    console.print("   • [cyan]CHUNK_SIZE[/cyan]: Default chunk size")
    console.print("   • [cyan]CHUNK_OVERLAP[/cyan]: Default chunk overlap")
    console.print("   • [cyan]MODEL_EXTRACT[/cyan]: Default extraction model")
    console.print("   • [cyan]MODEL_VALIDATE[/cyan]: Default validation model")
    console.print(
        "   📁 Copy [cyan]env.example[/cyan] to [cyan].env[/cyan] to customize defaults"
    )
    console.print()

    # Usage examples
    console.print("💡 [bold]Usage Examples:[/bold]")
    console.print("   [dim]# Basic usage with defaults[/dim]")
    console.print("   [cyan]extract-to-train extract document.pdf[/cyan]")
    console.print()
    console.print("   [dim]# Custom model and temperature[/dim]")
    console.print(
        "   [cyan]extract-to-train extract document.pdf --model-extract granite3.3:8b --temperature-extract 0.4[/cyan]"
    )
    console.print()
    console.print("   [dim]# Environment variable override[/dim]")
    console.print(
        "   [cyan]CHUNK_SIZE=256 extract-to-train extract document.pdf[/cyan]"
    )


def _show_current_configuration() -> None:
    """Show the current default configuration."""
    # Lazy imports to improve CLI startup time
    from .models.config import AppConfig

    config = AppConfig.get_educational_config()

    console.print("⚙️ [bold blue]Current Configuration[/bold blue]")
    console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Purpose", style="dim")

    table.add_row(
        "Generation Model", config.models["extraction"].name, "Q&A pair generation"
    )
    table.add_row(
        "Validation Model", config.models["validation"].name, "Quality assessment"
    )
    table.add_row(
        "Chunk Size", str(config.processing.chunk_size), "Text processing unit"
    )
    table.add_row(
        "Chunk Overlap", str(config.processing.chunk_overlap), "Context preservation"
    )

    console.print(table)


def _export_configuration_template(template_path: Path) -> None:
    """Export a configuration template file."""
    # Lazy imports to improve CLI startup time
    from .models.config import AppConfig

    config = AppConfig.get_educational_config()

    template_data = {
        "models": {
            "extraction": {
                "name": config.models["extraction"].name,
                "temperature": config.models["extraction"].temperature,
                "explanation": "Model for generating Q&A pairs",
            },
            "validation": {
                "name": config.models["validation"].name,
                "temperature": config.models["validation"].temperature,
                "explanation": "Model for validating dataset quality",
            },
        },
        "processing": {
            "chunk_size": config.processing.chunk_size,
            "chunk_overlap": config.processing.chunk_overlap,
            "max_pairs_per_chunk": config.processing.max_pairs_per_chunk,
        },
    }

    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(template_data, f, indent=2)

    console.print(f"📁 Configuration template exported to: {template_path}")


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
