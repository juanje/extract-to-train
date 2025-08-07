# Extract-to-Train Project Specifications

## 🎯 Project Overview

**Extract-to-Train** is an educational CLI tool that extracts information from PDF and Markdown documents and generates high-quality Q&A datasets in formats optimized for LLM fine-tuning with LoRA/QLoRA techniques. The project emphasizes learning through transparency, providing educational insights into each step of the dataset creation process.

**Key capabilities for large-scale processing:**
- Progressive saving with recovery from interruptions
- Chunk limitation for testing and quality assessment
- Multilingual document processing with language-aware generation

### Primary Objectives

1. **Extract** structured information from PDF and Markdown documents using docling and native parsers
2. **Generate** diverse question-answer pairs using local LLMs via Ollama
3. **Validate** and critique the generated dataset for quality assurance
4. **Export** in standard formats compatible with popular fine-tuning frameworks
5. **Educate** users about dataset creation best practices and fine-tuning workflows

### Educational Goals

- Demonstrate PDF and Markdown document processing techniques
- Show how to create high-quality training datasets from various sources
- Explain LLM prompt engineering for dataset generation
- Illustrate dataset validation and quality control
- Provide hands-on experience with fine-tuning data formats

## 🏗️ Project Architecture

```
extract-to-train/
├── src/
│   └── extract_to_train/
│       ├── __init__.py
│       ├── cli.py                    # Main CLI entry point
│       ├── core/
│       │   ├── __init__.py
│       │   ├── pipeline.py           # Main processing pipeline
│       │   ├── extractor.py          # PDF extraction with docling
│       │   ├── generator.py          # Q&A generation with LLMs
│       │   └── validator.py          # Dataset validation and quality control
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py             # Ollama client with LangChain
│       │   └── prompts.py            # Educational prompt templates
│       ├── models/
│       │   ├── __init__.py
│       │   ├── dataset.py            # Pydantic models for dataset formats
│       │   └── config.py             # Configuration models
│       └── utils/
│           ├── __init__.py
│           ├── config.py             # Application configuration
│           ├── stats.py              # Dataset statistics and analysis
│           ├── logger.py             # Educational logging setup
│           └── file_handler.py       # File I/O utilities
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── examples/
│   ├── sample_academic_paper.pdf
│   ├── sample_tutorial.pdf
│   ├── sample_report.pdf
│   ├── expected_outputs/
│   │   ├── academic_alpaca.jsonl
│   │   ├── tutorial_sharegpt.jsonl
│   │   └── report_openai.jsonl
│   └── tutorial.md
├── docs/
│   ├── getting_started.md
│   ├── configuration.md
│   └── best_practices.md
├── pyproject.toml
├── README.md
├── SPEC.md
└── .gitignore
```

## 🔧 Technology Stack

### Core Dependencies
- **PDF Processing**: `docling` - Robust PDF extraction preserving structure
- **Markdown Processing**: `markdown` - Native Markdown parsing and processing
- **LLM Framework**: `langchain` + `langchain-community` - LLM orchestration
- **Local LLMs**: `ollama` - Local model serving
- **Data Validation**: `pydantic` v2 - Type-safe data models
- **CLI Framework**: `typer` - Modern, educational CLI interface
- **HTTP Client**: `httpx` - Async HTTP for Ollama API

### Development Dependencies
- **Environment Management**: `uv` - Dependency and environment management
- **Code Quality**: `ruff` - Linting and formatting
- **Testing**: `pytest` + `pytest-asyncio` + `pytest-cov`
- **Type Checking**: `mypy`

### Model Requirements
- **Ollama**: Local installation required
- **Optimized Models** (based on available local models): 
  - `llama3.1:8B` - Primary for Q&A generation (improved instruction following)
  - `deepseek-r1:8B` - Primary for validation (superior reasoning and analysis)
  - `mistral:7B` - Alternative for generation (speed-optimized)
  - `llama3:8B` - Fallback option

## 📊 Dataset Formats

The tool supports three industry-standard formats compatible with major fine-tuning frameworks:

### 1. Alpaca Format
**Compatible with**: Axolotl, Unsloth, HuggingFace Transformers + PEFT

```jsonl
{"instruction": "What is machine learning according to the document?", "input": "", "output": "Machine learning is a branch of artificial intelligence that enables computers to learn and improve automatically from experience without being explicitly programmed for each specific task."}
{"instruction": "Explain the concept mentioned in the document", "input": "convolutional neural networks", "output": "Convolutional Neural Networks (CNNs) are a specialized type of artificial neural network designed specifically for processing grid-structured data like images. They use convolution operations to detect local features and spatial patterns."}
```

### 2. ShareGPT Format
**Compatible with**: Axolotl, Unsloth

```jsonl
{"conversations": [{"from": "human", "value": "According to the document, what advantages do neural networks have?"}, {"from": "gpt", "value": "According to the document, the main advantages of neural networks are: 1) Ability to learn complex non-linear patterns, 2) Adaptability to different data types, 3) Automatic improvement with more training data."}]}
```

### 3. OpenAI Format
**Compatible with**: HuggingFace Transformers + PEFT

```jsonl
{"messages": [{"role": "system", "content": "You are an expert who answers questions based on technical documents."}, {"role": "user", "content": "What does the document say about optimization algorithms?"}, {"role": "assistant", "content": "The document describes several optimization algorithms including SGD, Adam, and RMSprop, explaining that Adam combines the advantages of AdaGrad and RMSprop to achieve faster and more stable convergence."}]}
```

## 🤖 LLM Configuration

### Model Selection Strategy

#### Generation Model
- **Primary**: `llama3.1:8B`
- **Temperature**: `0.3`
- **Context Window**: `4096 tokens`
- **Rationale**: Latest Llama model with improved instruction following and consistency for diverse question generation

#### Validation Model  
- **Primary**: `deepseek-r1:8B`
- **Temperature**: `0.1`
- **Context Window**: `4096 tokens`
- **Rationale**: Superior reasoning model specifically designed for analysis, critique, and quality assessment

### Educational Model Configurations

```python
EDUCATIONAL_CONFIGS = {
    "extraction": {
        "name": "llama3.1:8B",
        "temperature": 0.3,
        "context_window": 4096,
        "explanation": "Latest Llama model with improved instruction following and balanced creativity for diverse question generation"
    },
    "validation": {
        "name": "deepseek-r1:8B", 
        "temperature": 0.1,
        "context_window": 4096,
        "explanation": "Superior reasoning model designed for analysis and critique - provides more thorough quality assessment than general models"
    }
}
```

### Optimized Configuration Benefits

This configuration leverages the best available local models for each specific task:

#### Why Llama 3.1:8B for Generation?
- **Enhanced Instruction Following**: Improved training for complex, structured prompts
- **Better Context Understanding**: Superior ability to maintain coherence across long contexts
- **Consistent Output Formatting**: More reliable JSON structure generation
- **Reduced Hallucination**: Better adherence to source document content

#### Why DeepSeek-R1:8B for Validation?
- **Specialized Reasoning**: Purpose-built for analysis and critical thinking tasks
- **Superior Quality Assessment**: More nuanced evaluation of Q&A pair quality
- **Detailed Feedback**: Better at identifying specific issues and providing actionable suggestions
- **Objectivity**: Separate model ensures unbiased validation independent of generation

#### Alternative Configurations
```python
# Speed-optimized (faster processing)
SPEED_CONFIG = {
    "extraction": {"name": "mistral:7B", "temperature": 0.3},
    "validation": {"name": "mistral:7B", "temperature": 0.1}
}

# Consistency-focused (same model for both tasks)
CONSISTENCY_CONFIG = {
    "extraction": {"name": "llama3.1:8B", "temperature": 0.3},
    "validation": {"name": "llama3.1:8B", "temperature": 0.05}
}
```

## 📋 Data Models

### Core Models

```python
class DatasetFormat(str, Enum):
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt" 
    OPENAI = "openai"

class QAPair(BaseModel):
    id: str
    question: str
    answer: str
    context: str
    difficulty: Literal["easy", "medium", "hard"]
    question_type: Literal["factual", "inferential", "comparative", "analytical"]
    confidence_score: float
    source_page: Optional[int] = None
    
class ValidationResult(BaseModel):
    is_valid: bool
    accuracy_score: int  # 0-10
    completeness_score: int  # 0-10
    clarity_score: int  # 0-10
    training_value_score: int  # 0-10
    issues: List[str]
    suggestions: List[str]

class DatasetStats(BaseModel):
    total_pairs: int
    avg_question_length: int
    avg_answer_length: int
    difficulty_distribution: Dict[str, int]
    question_type_distribution: Dict[str, int]
    pages_covered: int
    estimated_tokens: int

class Dataset(BaseModel):
    metadata: Dict[str, Any]
    qa_pairs: List[QAPair]
    validation_summary: ValidationResult
    stats: DatasetStats
```

## 🎛️ CLI Interface

### Main Commands

#### Extract Command
```bash
extract-to-train extract [OPTIONS] PDF_PATH
```

**Options:**
- `--output, -o`: Output file path (default: `dataset.jsonl`)
- `--format`: Dataset format (`alpaca`|`sharegpt`|`openai`, default: `alpaca`)
- `--max-pairs`: Maximum Q&A pairs to generate (default: 100)
- `--chunk-size`: Text chunk size for processing (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--model-extract`: Model for Q&A generation (default: `llama3.1:8b`)
- `--model-validate`: Model for validation (default: `llama3.1:8b`)
- `--temperature-extract`: Temperature for generation (default: 0.3)
- `--temperature-validate`: Temperature for validation (default: 0.1)
- `--skip-validation`: Skip dataset validation step
- `--include-context/--no-include-context`: Include context in instructions (default: True)
- `--system-prompt`: Custom system prompt for conversational formats
- `--verbose, -v`: Show detailed educational information
- `--explain`: Explain each step and show prompts used

#### Analysis Command
```bash
extract-to-train analyze [OPTIONS] DATASET_PATH
```

**Options:**
- `--detailed`: Show detailed analysis with recommendations
- `--export-stats`: Export statistics to JSON file

#### Setup Command
```bash
extract-to-train setup
```

Verifies environment setup and provides educational guidance.

### Usage Examples

```bash
# Basic usage with educational output
extract-to-train extract document.pdf --verbose

# Generate Alpaca format for Axolotl/Unsloth
extract-to-train extract document.pdf --format alpaca --output train_data.jsonl

# Generate conversational format
extract-to-train extract document.pdf --format sharegpt --max-pairs 50

# Generate OpenAI format with custom system prompt
extract-to-train extract document.pdf \
    --format openai \
    --system-prompt "You are a technical expert specializing in AI concepts." \
    --output openai_train.jsonl

# Quick generation without validation
extract-to-train extract document.pdf --skip-validation --max-pairs 20

# Analyze existing dataset
extract-to-train analyze train_data.jsonl --detailed
```

## 🔄 Processing Pipeline

### 1. PDF Extraction Phase
- **Tool**: docling
- **Process**: Extract text, tables, metadata while preserving structure
- **Output**: Structured document content with page references

### 2. Content Chunking Phase  
- **Strategy**: Logical section-based chunking
- **Size**: Configurable (default 1000 chars)
- **Overlap**: Configurable (default 200 chars)
- **Preservation**: Maintain context and relationships

### 3. Q&A Generation Phase
- **Model**: Configurable LLM via Ollama
- **Strategy**: Generate diverse question types and difficulties
- **Validation**: Real-time quality checks during generation
- **Output**: Structured Q&A pairs with metadata

### 4. Dataset Validation Phase
- **Model**: Separate LLM instance for objectivity
- **Criteria**: Accuracy, completeness, clarity, training value
- **Process**: Individual pair validation with scoring
- **Output**: Validation results and improvement suggestions

### 5. Export Phase
- **Formats**: Alpaca, ShareGPT, OpenAI
- **Output**: JSONL files optimized for fine-tuning tools
- **Statistics**: Comprehensive dataset analysis

## 🧪 Testing Strategy

### Coverage Target: >90%

#### Unit Tests
- PDF extraction with mocked files
- Q&A generation with mocked LLM responses  
- Data model validation and serialization
- Format conversion functions
- Utility functions and helpers

#### Integration Tests
- End-to-end pipeline with sample PDFs
- Ollama integration (with Docker container or mocks)
- CLI command testing
- File I/O operations

#### Test Data
- Sample PDFs of different types (academic, technical, reports)
- Expected outputs for regression testing
- Edge cases (empty PDFs, corrupted files, large documents)

### Test Structure
```
tests/
├── unit/
│   ├── test_extractor.py
│   ├── test_generator.py
│   ├── test_validator.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/
│   ├── test_pipeline.py
│   ├── test_cli.py
│   └── test_ollama_integration.py
└── fixtures/
    ├── sample_pdfs/
    ├── expected_outputs/
    └── mock_responses/
```

## 📚 Educational Features

### Verbose Mode
- Step-by-step explanations of the process
- Rationale behind configuration choices
- Real-time display of prompts and model responses
- Educational tips and best practices

### Prompt Transparency
- Version-controlled prompt templates with explanations
- Comments explaining design decisions
- Alternative prompt examples for learning

### Statistics and Analysis
- Comprehensive dataset metrics
- Quality distribution analysis
- Token count estimation for fine-tuning cost calculation
- Recommendations for improvement

### Educational Logging
- Informative log messages explaining each step
- Context about why certain decisions are made
- Performance insights and optimization tips

## 🔧 Configuration

### Environment Variables
```bash
OLLAMA_HOST=http://localhost:11434
LOG_LEVEL=INFO
MAX_RETRIES=3
TIMEOUT_SECONDS=30
```

### Configuration File (`config.yaml`)
```yaml
models:
  extraction:
    name: "llama3.1:8B"
    temperature: 0.3
    context_window: 4096
  validation:
    name: "deepseek-r1:8B"
    temperature: 0.1
    context_window: 4096

processing:
  chunk_size: 1000
  chunk_overlap: 200
  max_pairs_per_chunk: 5

generation:
  question_types: ["factual", "inferential", "comparative", "analytical"]
  difficulty_levels: ["easy", "medium", "hard"]
  min_answer_length: 50
  max_answer_length: 500

validation:
  min_confidence_score: 0.7
  required_scores:
    accuracy: 7
    completeness: 6
    clarity: 6
    training_value: 7
```

## 📊 Quality Metrics

### Dataset Quality Indicators
- **Diversity Score**: Distribution across question types and difficulties
- **Coherence Score**: Question-answer-context alignment
- **Coverage Score**: Percentage of source document covered
- **Training Readiness**: Compatibility with fine-tuning requirements

### Validation Criteria
- **Accuracy**: Factual correctness based on source context
- **Completeness**: Comprehensive coverage of the question
- **Clarity**: Clear, well-structured answers
- **Training Value**: Effectiveness for fine-tuning scenarios

## 🚀 Implementation Phases

### Phase 1: MVP (Core Functionality)
- Basic PDF extraction with docling
- Simple Q&A generation with one model
- Alpaca format export
- Basic CLI interface

### Phase 2: Enhanced (Full Features)
- Multi-format export support
- Dataset validation pipeline
- Educational verbose mode
- Comprehensive testing

### Phase 3: Polish (Production Ready)
- Performance optimizations
- Advanced error handling
- Complete documentation
- Example tutorials and best practices

## 📈 Success Metrics

### Technical Metrics
- **Code Coverage**: >90%
- **Performance**: Process 50-page PDF in <5 minutes
- **Quality**: >80% of generated pairs pass validation
- **Compatibility**: Support for 3 major fine-tuning frameworks

### Educational Metrics
- **Clarity**: Users understand each processing step
- **Learning**: Users can modify prompts and configurations
- **Reproducibility**: Consistent results across runs
- **Documentation**: Complete examples and tutorials

## 🔒 Security Considerations

- **Input Validation**: Sanitize PDF inputs and file paths
- **Resource Limits**: Prevent excessive memory usage with large PDFs
- **API Security**: Secure communication with Ollama instance
- **File Handling**: Safe file operations with proper error handling

## 📝 Documentation Requirements

### User Documentation
- **Getting Started Guide**: Setup and first run
- **Configuration Guide**: Detailed parameter explanations
- **Best Practices**: Tips for optimal dataset generation
- **Troubleshooting**: Common issues and solutions

### Developer Documentation
- **API Reference**: Complete function and class documentation
- **Architecture Guide**: System design and component interaction
- **Contributing Guide**: Development setup and contribution process
- **Testing Guide**: How to run and extend tests

## 🎯 Project Deliverables

1. **Functional CLI Tool**: Ready-to-use command-line application
2. **Educational Examples**: Sample PDFs with expected outputs
3. **Comprehensive Tests**: >90% coverage with integration tests
4. **Documentation**: Complete user and developer guides
5. **Configuration Templates**: Optimized settings for different use cases
6. **Tutorial Materials**: Step-by-step learning resources

This specification provides a complete roadmap for building an educational, production-ready tool for creating fine-tuning datasets from PDF documents while maintaining simplicity and focusing on learning outcomes.