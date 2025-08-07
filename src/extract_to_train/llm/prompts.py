"""
Educational prompt templates for Q&A generation and validation.

This module contains carefully crafted prompts with detailed explanations
of design decisions and best practices for educational purposes.
"""

from typing import Any

from langchain.prompts import PromptTemplate


class PromptTemplates:
    """
    Collection of educational prompt templates with design explanations.

    Educational Note: Well-designed prompts are crucial for consistent,
    high-quality outputs from LLMs. Each template here demonstrates
    specific prompt engineering techniques.
    """

    # Version control for prompts - important for reproducibility
    VERSION = "1.0.0"

    @staticmethod
    def get_qa_generation_prompt() -> PromptTemplate:
        """
        Prompt for generating diverse Q&A pairs from document content.

        Design Principles:
        1. Clear role definition ("You are creating training data...")
        2. Specific output format requirements
        3. Constraints to prevent hallucination
        4. Examples of desired question types
        5. Structured JSON output for parsing

        Educational Note: This prompt balances creativity with constraints
        to ensure useful training data while preventing common issues.
        """

        template = """
# Educational Prompt Design Note:
# This prompt uses several key techniques:
# 1. Role-based instruction for context setting
# 2. Explicit constraints to prevent hallucination
# 3. Diverse question type examples for variety
# 4. Structured output format for reliability
# 5. Clear quality criteria for consistency

You are creating high-quality training data for fine-tuning a language model. The model will learn this information as its own knowledge, so generate natural question-answer pairs that represent expertise in the subject matter.

## CONTENT TO LEARN:
{content}

## YOUR TASK:
Generate {num_questions} diverse questions and expert answers that represent deep knowledge of the subject matter. The answers should sound like they come from an expert who has internalized this knowledge.

## REQUIREMENTS:
1. **NATURAL EXPERTISE**: Answers should sound like an expert explaining concepts naturally
2. **NO DOCUMENT REFERENCES**: Never mention "the document", "according to the text", or similar phrases
3. **NO PAGE/SECTION REFERENCES**: Never mention "página X", "capítulo Y", "sección Z", or any location references
4. **DIRECT KNOWLEDGE**: Present information as established facts and concepts
5. **DIVERSE QUESTION TYPES**: Include different types as shown below
6. **VARIED DIFFICULTY**: Mix easy, medium, and hard questions
7. **COMPLETE ANSWERS**: Provide comprehensive but concise responses

## QUESTION TYPE EXAMPLES:
- **Factual**: "What is [specific concept]?"
- **Inferential**: "Why is [concept] important?"
- **Comparative**: "How does [concept A] differ from [concept B]?"
- **Analytical**: "What are the implications of [concept]?"

## OUTPUT FORMAT:
Respond with a JSON array containing exactly {num_questions} objects. Each object must have:

```json
[
    {{
        "question": "Clear, specific question about the subject matter",
        "answer": "Direct, expert answer presenting the information as established knowledge",
        "question_type": "factual|inferential|comparative|analytical",
        "difficulty": "easy|medium|hard",
        "confidence": 0.95
    }}
]
```

## QUALITY CRITERIA:
- Questions should be meaningful and educational
- Answers should teach concepts as established knowledge
- Use natural, expert language without any source references
- Ensure variety in question structure and focus
- Maintain professional, authoritative tone
- Present information as facts, not as "according to a source"
- NEVER include location references (pages, chapters, sections)
- Sound like an expert who naturally knows the information

Generate your {num_questions} expert Q&A pairs now:
"""

        return PromptTemplate(
            input_variables=["content", "num_questions"], template=template.strip()
        )

    @staticmethod
    def get_validation_prompt() -> PromptTemplate:
        """
        Prompt for validating Q&A pair quality for training purposes.

        Design Principles:
        1. Expert role for authority
        2. Specific evaluation criteria
        3. Quantitative scoring for objectivity
        4. Constructive feedback orientation
        5. Focus on training value

        Educational Note: This prompt creates a "second opinion" system
        that helps maintain dataset quality through systematic evaluation.
        """

        template = """
# Educational Prompt Design Note:
# This validation prompt implements several quality control techniques:
# 1. Expert role establishment for authoritative evaluation
# 2. Multi-criteria assessment for comprehensive quality check
# 3. Quantitative scoring reduces subjective bias
# 4. Specific feedback categories guide improvement
# 5. Training-focused perspective ensures practical value
# 6. Natural expertise assessment for fine-tuning quality

You are a quality control expert specializing in LLM training dataset validation. Your task is to rigorously evaluate Q&A pairs for fine-tuning quality, ensuring they represent natural expert knowledge.

## Q&A PAIR TO EVALUATE:

**Question:** {question}

**Answer:** {answer}

**Source Context:** {context}

## EVALUATION CRITERIA:

Evaluate this Q&A pair on each criterion using a 0-10 scale:

### 1. ACCURACY (0-10)
- Is the answer factually correct based on the provided context?
- Are there any hallucinations or invented facts?
- Does the answer stay within the bounds of the source material?

### 2. COMPLETENESS (0-10)
- Does the answer fully address the question asked?
- Are important aspects of the topic covered adequately?
- Is the response comprehensive without being verbose?

### 3. CLARITY (0-10)
- Is the answer well-structured and easy to understand?
- Does it use clear, professional language?
- Would this help a model learn to communicate effectively?

### 4. TRAINING VALUE (0-10)
- Would this example help a model learn useful patterns?
- Does it demonstrate good question-answering behavior?
- Is it representative of desirable model outputs?
- **CRITICAL**: Does the answer sound like expert knowledge rather than document reference?

## ADDITIONAL ASSESSMENT:
- **Natural Expertise**: Does the answer present information as established knowledge?
- **Reference-Free**: Is the answer completely free of document, page, section, or chapter references?
- **Expert Tone**: Does the answer sound like it comes from someone who knows the subject matter?
- **Self-Contained**: Can the answer stand alone without pointing to external sources or locations?
- **Educational Value**: Would this example teach something valuable for fine-tuning?

## CRITICAL ISSUES TO CHECK:
- ❌ **Document references**: "according to the document", "the text mentions", "as stated in the document"
- ❌ **Source citations**: "the author says", "this document explains", "based on the source"
- ❌ **Location references**: "página X", "capítulo Y", "sección Z", "page X", "chapter Y", "section Z"
- ❌ **Structural references**: "as shown in the table", "in the previous section", "see figure X"
- ✅ **Natural expertise**: Direct statements, confident explanations, expert-level knowledge
- ✅ **Established facts**: Presenting information as known concepts and principles
- ✅ **Self-contained knowledge**: Information presented without external references

## OUTPUT FORMAT:
Provide your evaluation as a JSON object:

```json
{{
    "is_valid": true/false,
    "accuracy_score": 0-10,
    "completeness_score": 0-10,
    "clarity_score": 0-10,
    "training_value_score": 0-10,
    "overall_score": calculated_average,
    "issues": ["specific problems identified"],
    "suggestions": ["specific improvement recommendations"],
    "reasoning": "Brief explanation of your evaluation"
}}
```

## VALIDATION GUIDELINES:
- **PASS THRESHOLD**: Overall score ≥ 7.0 with no score below 6
- **ISSUES**: Be specific about problems (e.g., "Answer includes information not in context")
- **SUGGESTIONS**: Provide actionable improvement advice
- **REASONING**: Explain your scoring rationale briefly

Evaluate this Q&A pair now:
"""

        return PromptTemplate(
            input_variables=["question", "answer", "context"], template=template.strip()
        )

    @staticmethod
    def get_correction_prompt() -> PromptTemplate:
        """
        Prompt for correcting identified issues in Q&A pairs.

        Design Principles:
        1. Specific correction guidance
        2. Preserve original intent
        3. Maintain educational value
        4. Focus on fixable issues

        Educational Note: Automatic correction can improve borderline
        examples rather than discarding them entirely.
        """

        template = """
# Educational Prompt Design Note:
# This correction prompt demonstrates:
# 1. Preservation of original educational intent
# 2. Targeted improvement rather than complete rewriting
# 3. Explicit instruction to maintain training value
# 4. Structured approach to systematic improvement

You are an expert editor specializing in improving training data for language models. Your task is to correct the identified issues while preserving the educational value and intent of the original Q&A pair.

## ORIGINAL Q&A PAIR:
**Question:** {question}
**Answer:** {answer}
**Context:** {context}

## IDENTIFIED ISSUES:
{issues}

## IMPROVEMENT SUGGESTIONS:
{suggestions}

## CORRECTION GUIDELINES:
1. **Preserve Intent**: Keep the original question's purpose and educational goal
2. **Fix Accuracy**: Ensure all facts align with the provided context
3. **Improve Clarity**: Enhance readability without changing meaning
4. **Maintain Training Value**: Keep the example useful for model learning
5. **Minimal Changes**: Make only necessary corrections

## OUTPUT FORMAT:
Provide the corrected Q&A pair:

```json
{{
    "corrected_question": "Improved question if needed",
    "corrected_answer": "Improved answer addressing the issues",
    "changes_made": ["list of specific changes"],
    "confidence": 0.0-1.0
}}
```

Provide your correction now:
"""

        return PromptTemplate(
            input_variables=["question", "answer", "context", "issues", "suggestions"],
            template=template.strip(),
        )

    @classmethod
    def get_system_prompts(cls) -> dict[str, str]:
        """
        Get system prompts for different conversation formats.

        Returns:
            Dictionary of system prompts for various use cases

        Educational Note: System prompts set the overall behavior
        and personality of the model in conversational formats.
        """

        return {
            "expert_tutor": (
                "You are an expert tutor who provides clear, accurate explanations "
                "based on educational materials. You focus on helping students "
                "understand concepts deeply while staying faithful to source content."
            ),
            "technical_assistant": (
                "You are a knowledgeable technical assistant who answers questions "
                "based on documentation and technical materials. You provide precise, "
                "well-structured responses that help users apply information effectively."
            ),
            "research_analyst": (
                "You are a research analyst who examines documents and provides "
                "insightful answers to questions. You focus on accuracy, completeness, "
                "and helping users understand complex information."
            ),
            "general_assistant": (
                "You are a helpful assistant who answers questions based on provided "
                "information. You strive to be accurate, clear, and educational in "
                "your responses."
            ),
        }

    @classmethod
    def get_prompt_explanations(cls) -> dict[str, str]:
        """
        Get detailed explanations of prompt design decisions.

        Returns:
            Dictionary explaining why each prompt works

        Educational Note: Understanding prompt design helps users
        modify and improve prompts for their specific needs.
        """

        return {
            "qa_generation": (
                "🎯 Q&A GENERATION PROMPT DESIGN:\n\n"
                "KEY TECHNIQUES:\n"
                "• Role Definition: Establishes context and authority\n"
                "• Explicit Constraints: Prevents hallucination and ensures accuracy\n"
                "• Example Patterns: Shows desired question types for variety\n"
                "• Structured Output: JSON format enables reliable parsing\n"
                "• Quality Criteria: Sets clear expectations for output\n\n"
                "WHY IT WORKS:\n"
                "• Clear instructions reduce ambiguity\n"
                "• Examples guide without constraining creativity\n"
                "• Constraints ensure training data quality\n"
                "• Structure enables automated processing"
            ),
            "validation": (
                "🔍 VALIDATION PROMPT DESIGN:\n\n"
                "KEY TECHNIQUES:\n"
                "• Expert Role: Creates authority for quality assessment\n"
                "• Multi-Criteria Scoring: Comprehensive evaluation approach\n"
                "• Quantitative Metrics: Reduces subjective bias\n"
                "• Specific Guidelines: Ensures consistent evaluation\n"
                "• Constructive Feedback: Enables improvement rather than just rejection\n\n"
                "WHY IT WORKS:\n"
                "• Systematic approach ensures thorough evaluation\n"
                "• Numeric scores enable automated filtering\n"
                "• Detailed feedback guides improvements\n"
                "• Training-focused perspective maintains relevance"
            ),
            "correction": (
                "🛠️ CORRECTION PROMPT DESIGN:\n\n"
                "KEY TECHNIQUES:\n"
                "• Preservation Focus: Maintains original educational intent\n"
                "• Targeted Improvements: Fixes specific issues efficiently\n"
                "• Minimal Change Principle: Avoids unnecessary modifications\n"
                "• Confidence Scoring: Indicates reliability of corrections\n\n"
                "WHY IT WORKS:\n"
                "• Preserves valuable training examples\n"
                "• Improves quality without starting over\n"
                "• Maintains consistency with original content\n"
                "• Provides transparency about changes made"
            ),
        }

    @classmethod
    def get_temperature_recommendations(cls) -> dict[str, dict[str, float|str]]:
        """
        Get temperature recommendations for different prompt types.

        Returns:
            Dictionary with recommended temperatures for various tasks

        Educational Note: Temperature significantly affects output quality
        and consistency. Different tasks require different settings.
        """

        return {
            "qa_generation": {
                "recommended": 0.3,
                "min": 0.2,
                "max": 0.5,
                "explanation": "Balanced creativity for diverse questions while maintaining consistency",
            },
            "validation": {
                "recommended": 0.1,
                "min": 0.0,
                "max": 0.2,
                "explanation": "Low temperature ensures consistent, objective evaluation",
            },
            "correction": {
                "recommended": 0.2,
                "min": 0.1,
                "max": 0.3,
                "explanation": "Slight creativity for improvements while maintaining accuracy",
            },
        }

    @classmethod
    def validate_prompt_inputs(cls, prompt_type: str, inputs: dict[str, Any]) -> list[str]:
        """
        Validate inputs for specific prompt types.

        Args:
            prompt_type: Type of prompt being used
            inputs: Dictionary of input variables

        Returns:
            List of validation errors (empty if valid)

        Educational Note: Input validation prevents runtime errors
        and ensures prompts receive the data they expect.
        """

        errors = []

        if prompt_type == "qa_generation":
            if "content" not in inputs or not inputs["content"].strip():
                errors.append("Content must be provided and non-empty")
            if "num_questions" not in inputs or inputs["num_questions"] < 1:
                errors.append("Number of questions must be positive integer")
            if len(inputs.get("content", "")) < 100:
                errors.append(
                    "Content should be at least 100 characters for meaningful Q&A generation"
                )

        elif prompt_type == "validation":
            required_fields = ["question", "answer", "context"]
            for field in required_fields:
                if field not in inputs or not inputs[field].strip():
                    errors.append(
                        f"{field.capitalize()} must be provided and non-empty"
                    )

        elif prompt_type == "correction":
            required_fields = ["question", "answer", "context", "issues", "suggestions"]
            for field in required_fields:
                if field not in inputs:
                    errors.append(f"{field.capitalize()} must be provided")

        return errors
