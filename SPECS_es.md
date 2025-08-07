# Especificaciones del Proyecto Extract-to-Train

## 🎯 Resumen del Proyecto

**Extract-to-Train** es una herramienta CLI educativa que extrae información de documentos PDF y Markdown y genera datasets de Q&A de alta calidad en formatos optimizados para fine-tuning de LLM con técnicas LoRA/QLoRA. El proyecto enfatiza el aprendizaje a través de la transparencia, proporcionando insights educativos en cada paso del proceso de creación de datasets.

**Capacidades clave para procesamiento a gran escala:**
- Guardado progresivo con recuperación de interrupciones
- Limitación de chunks para testing y evaluación de calidad
- Procesamiento multiidioma de documentos con generación consciente del idioma

### Objetivos Principales

1. **Extraer** información estructurada de documentos PDF y Markdown usando docling y parsers nativos
2. **Generar** pares diversos de pregunta-respuesta usando LLMs locales vía Ollama
3. **Validar** y criticar el dataset generado para asegurar la calidad
4. **Exportar** en formatos estándar compatibles con frameworks populares de fine-tuning
5. **Educar** a los usuarios sobre mejores prácticas de creación de datasets y flujos de trabajo de fine-tuning

### Objetivos Educativos

- Demostrar técnicas de procesamiento de documentos PDF y Markdown
- Mostrar cómo crear datasets de entrenamiento de alta calidad desde varias fuentes
- Explicar ingeniería de prompts para LLM en generación de datasets
- Ilustrar validación de datasets y control de calidad
- Proporcionar experiencia práctica con formatos de datos para fine-tuning

## 🏗️ Arquitectura del Proyecto

```
extract-to-train/
├── src/
│   └── extract_to_train/
│       ├── __init__.py
│       ├── cli.py                    # Punto de entrada CLI principal
│       ├── core/
│       │   ├── __init__.py
│       │   ├── pipeline.py           # Pipeline principal de procesamiento
│       │   ├── extractor.py          # Extracción PDF con docling
│       │   ├── generator.py          # Generación Q&A con LLMs
│       │   └── validator.py          # Validación y control de calidad del dataset
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py             # Cliente Ollama con LangChain
│       │   └── prompts.py            # Plantillas de prompts educativas
│       ├── models/
│       │   ├── __init__.py
│       │   ├── dataset.py            # Modelos Pydantic para formatos de dataset
│       │   └── config.py             # Modelos de configuración
│       └── utils/
│           ├── __init__.py
│           ├── config.py             # Configuración de aplicación
│           ├── stats.py              # Estadísticas y análisis de dataset
│           ├── logger.py             # Configuración de logging educativo
│           └── file_handler.py       # Utilidades de I/O de archivos
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

## 🔧 Stack Tecnológico

### Dependencias Principales
- **Procesamiento PDF**: `docling` - Extracción robusta de PDF preservando estructura
- **Procesamiento Markdown**: `markdown` - Parsing y procesamiento nativo de Markdown
- **Framework LLM**: `langchain` + `langchain-community` - Orquestación de LLM
- **LLMs Locales**: `ollama` - Servicio de modelos locales
- **Validación de Datos**: `pydantic` v2 - Modelos de datos type-safe
- **Framework CLI**: `typer` - Interfaz CLI moderna y educativa
- **Cliente HTTP**: `httpx` - HTTP asíncrono para API Ollama

### Dependencias de Desarrollo
- **Gestión de Entorno**: `uv` - Gestión de dependencias y entorno
- **Calidad de Código**: `ruff` - Linting y formateo
- **Testing**: `pytest` + `pytest-asyncio` + `pytest-cov`
- **Verificación de Tipos**: `mypy`

### Requerimientos de Modelos
- **Ollama**: Instalación local requerida
- **Modelos Optimizados** (basado en modelos locales disponibles): 
  - `llama3.1:8B` - Principal para generación Q&A (mejor seguimiento de instrucciones)
  - `deepseek-r1:8B` - Principal para validación (razonamiento y análisis superior)
  - `mistral:7B` - Alternativa para generación (optimizado para velocidad)
  - `llama3:8B` - Opción de respaldo

## 📊 Formatos de Dataset

La herramienta soporta tres formatos estándar de la industria compatibles con los principales frameworks de fine-tuning:

### 1. Formato Alpaca
**Compatible con**: Axolotl, Unsloth, HuggingFace Transformers + PEFT

```jsonl
{"instruction": "¿Qué es machine learning según el documento?", "input": "", "output": "Machine learning es una rama de la inteligencia artificial que permite a las computadoras aprender y mejorar automáticamente de la experiencia sin ser explícitamente programadas para cada tarea específica."}
{"instruction": "Explica el concepto mencionado en el documento", "input": "redes neuronales convolucionales", "output": "Las Redes Neuronales Convolucionales (CNNs) son un tipo especializado de red neuronal artificial diseñado específicamente para procesar datos estructurados en grilla como imágenes. Usan operaciones de convolución para detectar características locales y patrones espaciales."}
```

### 2. Formato ShareGPT
**Compatible con**: Axolotl, Unsloth

```jsonl
{"conversations": [{"from": "human", "value": "Según el documento, ¿qué ventajas tienen las redes neuronales?"}, {"from": "gpt", "value": "Según el documento, las principales ventajas de las redes neuronales son: 1) Capacidad de aprender patrones complejos no lineales, 2) Adaptabilidad a diferentes tipos de datos, 3) Mejora automática con más datos de entrenamiento."}]}
```

### 3. Formato OpenAI
**Compatible con**: HuggingFace Transformers + PEFT

```jsonl
{"messages": [{"role": "system", "content": "Eres un experto que responde preguntas basadas en documentos técnicos."}, {"role": "user", "content": "¿Qué dice el documento sobre algoritmos de optimización?"}, {"role": "assistant", "content": "El documento describe varios algoritmos de optimización incluyendo SGD, Adam y RMSprop, explicando que Adam combina las ventajas de AdaGrad y RMSprop para lograr una convergencia más rápida y estable."}]}
```

## 🤖 Configuración LLM

### Estrategia de Selección de Modelos

#### Modelo de Generación
- **Principal**: `llama3.1:8B`
- **Temperatura**: `0.3`
- **Ventana de Contexto**: `4096 tokens`
- **Justificación**: Modelo Llama más reciente con mejor seguimiento de instrucciones y consistencia para generación diversa de preguntas

#### Modelo de Validación  
- **Principal**: `deepseek-r1:8B`
- **Temperatura**: `0.1`
- **Ventana de Contexto**: `4096 tokens`
- **Justificación**: Modelo de razonamiento superior diseñado específicamente para análisis, crítica y evaluación de calidad

### Configuraciones de Modelos Educativas

```python
EDUCATIONAL_CONFIGS = {
    "extraction": {
        "name": "llama3.1:8B",
        "temperature": 0.3,
        "context_window": 4096,
        "explanation": "Modelo Llama más reciente con mejor seguimiento de instrucciones y creatividad balanceada para generación diversa de preguntas"
    },
    "validation": {
        "name": "deepseek-r1:8B", 
        "temperature": 0.1,
        "context_window": 4096,
        "explanation": "Modelo de razonamiento superior diseñado para análisis y crítica - proporciona evaluación de calidad más exhaustiva que modelos generales"
    }
}
```

### Beneficios de la Configuración Optimizada

Esta configuración aprovecha los mejores modelos locales disponibles para cada tarea específica:

#### ¿Por qué Llama 3.1:8B para Generación?
- **Seguimiento de Instrucciones Mejorado**: Entrenamiento mejorado para prompts complejos y estructurados
- **Mejor Comprensión de Contexto**: Capacidad superior para mantener coherencia a través de contextos largos
- **Formato de Salida Consistente**: Generación más confiable de estructura JSON
- **Reducción de Alucinaciones**: Mejor adherencia al contenido del documento fuente

#### ¿Por qué DeepSeek-R1:8B para Validación?
- **Razonamiento Especializado**: Construido específicamente para tareas de análisis y pensamiento crítico
- **Evaluación de Calidad Superior**: Evaluación más matizada de la calidad de pares Q&A
- **Retroalimentación Detallada**: Mejor en identificar problemas específicos y proporcionar sugerencias accionables
- **Objetividad**: Modelo separado asegura validación imparcial independiente de la generación

#### Configuraciones Alternativas
```python
# Optimizado para velocidad (procesamiento más rápido)
SPEED_CONFIG = {
    "extraction": {"name": "mistral:7B", "temperature": 0.3},
    "validation": {"name": "mistral:7B", "temperature": 0.1}
}

# Enfocado en consistencia (mismo modelo para ambas tareas)
CONSISTENCY_CONFIG = {
    "extraction": {"name": "llama3.1:8B", "temperature": 0.3},
    "validation": {"name": "llama3.1:8B", "temperature": 0.05}
}
```

## 📋 Modelos de Datos

### Modelos Principales

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

## 🎛️ Interfaz CLI

### Comandos Principales

#### Comando Extract
```bash
extract-to-train extract [OPTIONS] PDF_PATH
```

**Opciones:**
- `--output, -o`: Ruta del archivo de salida (por defecto: `dataset.jsonl`)
- `--format`: Formato de dataset (`alpaca`|`sharegpt`|`openai`, por defecto: `alpaca`)
- `--max-pairs`: Máximo de pares Q&A a generar (por defecto: 100)
- `--chunk-size`: Tamaño de chunk de texto para procesamiento (por defecto: 1000)
- `--chunk-overlap`: Solapamiento entre chunks (por defecto: 200)
- `--model-extract`: Modelo para generación Q&A (por defecto: `llama3.1:8b`)
- `--model-validate`: Modelo para validación (por defecto: `llama3.1:8b`)
- `--temperature-extract`: Temperatura para generación (por defecto: 0.3)
- `--temperature-validate`: Temperatura para validación (por defecto: 0.1)
- `--skip-validation`: Omitir paso de validación de dataset
- `--include-context/--no-include-context`: Incluir contexto en instrucciones (por defecto: True)
- `--system-prompt`: Prompt de sistema personalizado para formatos conversacionales
- `--verbose, -v`: Mostrar información educativa detallada
- `--explain`: Explicar cada paso y mostrar prompts utilizados

#### Comando Analysis
```bash
extract-to-train analyze [OPTIONS] DATASET_PATH
```

**Opciones:**
- `--detailed`: Mostrar análisis detallado con recomendaciones
- `--export-stats`: Exportar estadísticas a archivo JSON

#### Comando Setup
```bash
extract-to-train setup
```

Verifica la configuración del entorno y proporciona guía educativa.

### Ejemplos de Uso

```bash
# Uso básico con salida educativa
extract-to-train extract document.pdf --verbose

# Generar formato Alpaca para Axolotl/Unsloth
extract-to-train extract document.pdf --format alpaca --output train_data.jsonl

# Generar formato conversacional
extract-to-train extract document.pdf --format sharegpt --max-pairs 50

# Generar formato OpenAI con prompt de sistema personalizado
extract-to-train extract document.pdf \
    --format openai \
    --system-prompt "Eres un experto técnico especializado en conceptos de IA." \
    --output openai_train.jsonl

# Generación rápida sin validación
extract-to-train extract document.pdf --skip-validation --max-pairs 20

# Analizar dataset existente
extract-to-train analyze train_data.jsonl --detailed
```

## 🔄 Pipeline de Procesamiento

### 1. Fase de Extracción PDF
- **Herramienta**: docling
- **Proceso**: Extraer texto, tablas, metadatos preservando estructura
- **Salida**: Contenido de documento estructurado con referencias de página

### 2. Fase de Chunking de Contenido  
- **Estrategia**: Chunking basado en secciones lógicas
- **Tamaño**: Configurable (por defecto 1000 chars)
- **Solapamiento**: Configurable (por defecto 200 chars)
- **Preservación**: Mantener contexto y relaciones

### 3. Fase de Generación Q&A
- **Modelo**: LLM configurable vía Ollama
- **Estrategia**: Generar tipos de preguntas y dificultades diversas
- **Validación**: Verificaciones de calidad en tiempo real durante generación
- **Salida**: Pares Q&A estructurados con metadatos

### 4. Fase de Validación de Dataset
- **Modelo**: Instancia LLM separada para objetividad
- **Criterios**: Precisión, completitud, claridad, valor de entrenamiento
- **Proceso**: Validación de pares individuales con puntuación
- **Salida**: Resultados de validación y sugerencias de mejora

### 5. Fase de Exportación
- **Formatos**: Alpaca, ShareGPT, OpenAI
- **Salida**: Archivos JSONL optimizados para herramientas de fine-tuning
- **Estadísticas**: Análisis integral de dataset

## 🧪 Estrategia de Testing

### Objetivo de Cobertura: >90%

#### Tests Unitarios
- Extracción PDF con archivos mock
- Generación Q&A con respuestas LLM mock  
- Validación y serialización de modelos de datos
- Funciones de conversión de formato
- Funciones de utilidad y helpers

#### Tests de Integración
- Pipeline end-to-end con PDFs de muestra
- Integración Ollama (con contenedor Docker o mocks)
- Testing de comandos CLI
- Operaciones de I/O de archivos

#### Datos de Test
- PDFs de muestra de diferentes tipos (académicos, técnicos, reportes)
- Salidas esperadas para testing de regresión
- Casos extremos (PDFs vacíos, archivos corruptos, documentos grandes)

### Estructura de Tests
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

## 📚 Características Educativas

### Modo Verbose
- Explicaciones paso a paso del proceso
- Justificación detrás de las elecciones de configuración
- Visualización en tiempo real de prompts y respuestas del modelo
- Tips educativos y mejores prácticas

### Transparencia de Prompts
- Plantillas de prompts con control de versiones y explicaciones
- Comentarios explicando decisiones de diseño
- Ejemplos alternativos de prompts para aprendizaje

### Estadísticas y Análisis
- Métricas integrales de dataset
- Análisis de distribución de calidad
- Estimación de conteo de tokens para cálculo de costo de fine-tuning
- Recomendaciones para mejora

### Logging Educativo
- Mensajes de log informativos explicando cada paso
- Contexto sobre por qué se toman ciertas decisiones
- Insights de rendimiento y tips de optimización

## 🔧 Configuración

### Variables de Entorno
```bash
OLLAMA_HOST=http://localhost:11434
LOG_LEVEL=INFO
MAX_RETRIES=3
TIMEOUT_SECONDS=30
```

### Archivo de Configuración (`config.yaml`)
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

## 📊 Métricas de Calidad

### Indicadores de Calidad del Dataset
- **Puntuación de Diversidad**: Distribución a través de tipos de preguntas y dificultades
- **Puntuación de Coherencia**: Alineación pregunta-respuesta-contexto
- **Puntuación de Cobertura**: Porcentaje del documento fuente cubierto
- **Preparación para Entrenamiento**: Compatibilidad con requerimientos de fine-tuning

### Criterios de Validación
- **Precisión**: Corrección factual basada en contexto fuente
- **Completitud**: Cobertura integral de la pregunta
- **Claridad**: Respuestas claras y bien estructuradas
- **Valor de Entrenamiento**: Efectividad para escenarios de fine-tuning

## 🚀 Fases de Implementación

### Fase 1: MVP (Funcionalidad Central)
- Extracción básica de PDF con docling
- Generación simple Q&A con un modelo
- Exportación formato Alpaca
- Interfaz CLI básica

### Fase 2: Mejorada (Características Completas)
- Soporte de exportación multi-formato
- Pipeline de validación de dataset
- Modo verbose educativo
- Testing integral

### Fase 3: Pulida (Lista para Producción)
- Optimizaciones de rendimiento
- Manejo avanzado de errores
- Documentación completa
- Tutoriales de ejemplo y mejores prácticas

## 📈 Métricas de Éxito

### Métricas Técnicas
- **Cobertura de Código**: >90%
- **Rendimiento**: Procesar PDF de 50 páginas en <5 minutos
- **Calidad**: >80% de pares generados pasan validación
- **Compatibilidad**: Soporte para 3 frameworks principales de fine-tuning

### Métricas Educativas
- **Claridad**: Los usuarios entienden cada paso de procesamiento
- **Aprendizaje**: Los usuarios pueden modificar prompts y configuraciones
- **Reproducibilidad**: Resultados consistentes a través de ejecuciones
- **Documentación**: Ejemplos completos y tutoriales

## 🔒 Consideraciones de Seguridad

- **Validación de Entrada**: Sanitizar entradas PDF y rutas de archivos
- **Límites de Recursos**: Prevenir uso excesivo de memoria con PDFs grandes
- **Seguridad de API**: Comunicación segura con instancia Ollama
- **Manejo de Archivos**: Operaciones de archivo seguras con manejo apropiado de errores

## 📝 Requerimientos de Documentación

### Documentación de Usuario
- **Guía de Inicio**: Configuración y primera ejecución
- **Guía de Configuración**: Explicaciones detalladas de parámetros
- **Mejores Prácticas**: Tips para generación óptima de datasets
- **Resolución de Problemas**: Problemas comunes y soluciones

### Documentación de Desarrollador
- **Referencia de API**: Documentación completa de funciones y clases
- **Guía de Arquitectura**: Diseño del sistema e interacción de componentes
- **Guía de Contribución**: Configuración de desarrollo y proceso de contribución
- **Guía de Testing**: Cómo ejecutar y extender tests

## 🎯 Entregables del Proyecto

1. **Herramienta CLI Funcional**: Aplicación de línea de comandos lista para usar
2. **Ejemplos Educativos**: PDFs de muestra con salidas esperadas
3. **Tests Integrales**: >90% cobertura con tests de integración
4. **Documentación**: Guías completas de usuario y desarrollador
5. **Plantillas de Configuración**: Configuraciones optimizadas para diferentes casos de uso
6. **Materiales de Tutorial**: Recursos de aprendizaje paso a paso

Esta especificación proporciona un roadmap completo para construir una herramienta educativa y lista para producción para crear datasets de fine-tuning desde documentos PDF mientras mantiene simplicidad y enfoque en resultados de aprendizaje.
