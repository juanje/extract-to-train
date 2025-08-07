# Extract-to-Train Tutorial

Welcome to the Extract-to-Train tutorial! This guide will walk you through creating your first training dataset from a PDF document.

## 🎯 Learning Objectives

By the end of this tutorial, you'll understand:
- How to extract structured information from PDFs
- The process of generating Q&A pairs for training
- Different output formats and their use cases
- Quality validation and improvement techniques
- Best practices for dataset creation

## 📋 Prerequisites

1. **Extract-to-Train installed** with all dependencies
2. **Ollama running** with required models:
   ```bash
   ollama pull llama3.1:8B
   ollama pull deepseek-r1:8B
   ```
3. **A PDF document** to process (we'll use a sample ML paper)

## 🚀 Step 1: Environment Setup

First, verify your environment is ready:

```bash
extract-to-train setup
```

This command checks:
- ✅ Ollama connectivity
- ✅ Required models availability
- ✅ Model response testing

If everything passes, you're ready to proceed!

## 📄 Step 2: Basic Extraction

Let's start with a simple extraction using verbose mode to see what's happening:

```bash
extract-to-train extract sample_paper.pdf --verbose
```

**What you'll see:**
- 📄 PDF content extraction with docling
- ✂️ Intelligent chunking preserving context
- 🤖 Q&A generation with llama3.1:8B
- 🔍 Quality validation with deepseek-r1:8B
- 📊 Comprehensive statistics and insights

**Educational Note:** Verbose mode is perfect for learning! It shows you exactly how each step works and why certain decisions are made.

## 🎯 Step 3: Understanding Output Formats

### Alpaca Format (Universal)
```bash
extract-to-train extract sample_paper.pdf --format alpaca --max-pairs 20
```

**Best for:** Axolotl, Unsloth, HuggingFace+PEFT
**Structure:** instruction/input/output
**Use case:** General instruction following

### ShareGPT Format (Conversational)
```bash
extract-to-train extract sample_paper.pdf --format sharegpt --max-pairs 20
```

**Best for:** Axolotl, Unsloth
**Structure:** conversation turns
**Use case:** Chat model fine-tuning

### OpenAI Format (API Compatible)
```bash
extract-to-train extract sample_paper.pdf \
    --format openai \
    --system-prompt "You are an AI research expert" \
    --max-pairs 20
```

**Best for:** HuggingFace Transformers+PEFT
**Structure:** system/user/assistant messages
**Use case:** API-compatible fine-tuning

## 🔧 Step 4: Optimization and Customization

### Quality-Focused Processing
```bash
extract-to-train extract sample_paper.pdf \
    --temperature-extract 0.2 \
    --chunk-size 800 \
    --chunk-overlap 300 \
    --auto-correct \
    --verbose
```

**Why these settings?**
- Lower temperature (0.2) = more consistent outputs
- Smaller chunks (800) = better context focus
- More overlap (300) = better context preservation
- Auto-correct = improves borderline examples

### Speed-Optimized Processing
```bash
extract-to-train extract sample_paper.pdf \
    --model-extract mistral:7B \
    --model-validate mistral:7B \
    --skip-validation \
    --chunk-size 1500 \
    --max-pairs 50
```

**Why these settings?**
- Mistral:7B = faster processing
- Skip validation = immediate results
- Larger chunks = fewer API calls
- Same model = consistency but less validation objectivity

## 📊 Step 5: Dataset Analysis

After generating your dataset, analyze its quality:

```bash
extract-to-train analyze dataset.jsonl --detailed
```

**You'll see:**
- 📈 Question type distribution
- 📊 Difficulty level balance
- 🎯 Quality metrics and scores
- 💡 Specific recommendations for improvement

## 🎓 Step 6: Understanding the Educational Output

Let's run with the explain flag to understand the process:

```bash
extract-to-train extract sample_paper.pdf --explain
```

**Educational Insights:**
- **Model Selection:** Why llama3.1:8B for generation and deepseek-r1:8B for validation
- **Temperature Settings:** How 0.3 balances creativity with consistency
- **Prompt Engineering:** See the actual prompts used and why they work
- **Quality Criteria:** Understand the validation scoring system

## 🔍 Step 7: Troubleshooting Common Issues

### Issue: Low Pass Rate (<70%)
```bash
# Try lower temperature for more consistent outputs
extract-to-train extract sample_paper.pdf --temperature-extract 0.2
```

### Issue: Questions Too Similar
```bash
# Increase temperature for more diversity
extract-to-train extract sample_paper.pdf --temperature-extract 0.4
```

### Issue: Processing Too Slow
```bash
# Use faster model and skip validation
extract-to-train extract sample_paper.pdf \
    --model-extract mistral:7B \
    --skip-validation
```

## 🎯 Step 8: Fine-tuning Framework Integration

### For Axolotl
```yaml
# axolotl config
datasets:
  - path: dataset.jsonl
    type: alpaca
```

### For Unsloth
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit"
)
model = FastLanguageModel.get_peft_model(model, ...)

# Load your dataset
dataset = load_dataset("json", data_files="dataset.jsonl")
```

### For HuggingFace + PEFT
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Your generated dataset works directly with standard training loops
```

## 💡 Best Practices Learned

1. **Start Small:** Begin with 20-50 pairs to test your pipeline
2. **Use Verbose Mode:** Educational output helps optimize settings
3. **Validate Quality:** Always review generated examples manually
4. **Iterate on Prompts:** The tool's prompts are educational starting points
5. **Balance Speed vs Quality:** Choose settings based on your use case
6. **Analyze Results:** Use the analysis command to understand your dataset

## 🎉 Congratulations!

You've successfully:
- ✅ Extracted content from a PDF
- ✅ Generated training-ready Q&A pairs
- ✅ Understood different output formats
- ✅ Learned optimization techniques
- ✅ Analyzed dataset quality
- ✅ Prepared data for fine-tuning

## 🚀 Next Steps

1. **Try Different PDFs:** Test with various document types
2. **Experiment with Settings:** Find optimal configurations for your use case
3. **Fine-tune a Model:** Use your dataset with your preferred framework
4. **Share Your Results:** Contribute back to the community

## 📚 Additional Resources

- [Configuration Reference](../docs/configuration.md)
- [Best Practices Guide](../docs/best_practices.md)
- [Troubleshooting Guide](../docs/troubleshooting.md)
- [Community Forum](https://github.com/your-username/extract-to-train/discussions)

Happy fine-tuning! 🎯