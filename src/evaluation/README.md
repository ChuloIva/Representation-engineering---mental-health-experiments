# 🎯 Evaluation

Scripts for evaluating model responses using various LLM-as-judge approaches.

## Scripts

### `llm_judge_base.py`
**Base LLM judge functionality using local models**
- Core evaluation framework
- Local model integration via Ollama
- Customizable evaluation criteria

### `llm_judge_local.py`
**Local model evaluation pipeline**
- Uses local models for evaluation (Emollama-7b)
- Offline evaluation capability
- Cost-effective for large datasets

### `llm_judge_openai.py`
**OpenAI-based evaluation pipeline**
- Uses GPT models for evaluation
- Higher quality but cost considerations
- API rate limiting and error handling

### `llm_judge_enhanced.py`
**Enhanced evaluation with multiple metrics**
- Multi-dimensional scoring
- Aggregated evaluation metrics
- Advanced prompt engineering

### `question_analyzer.py`
**Question analysis and response processing**
- Question categorization and analysis
- Response preprocessing
- Statistical analysis of results

## Evaluation Dimensions

The evaluation framework assesses responses across multiple dimensions:

1. **Mood Health** - Emotional positivity and mental health indicators
2. **Relevance** - Appropriateness and on-topic nature of responses
3. **Emotional Competence** - Emotional intelligence and self-awareness
4. **Thought Adaptiveness** - Cognitive flexibility and adaptive thinking

## Usage

```python
from evaluation.llm_judge_openai import evaluate_responses

results = evaluate_responses(
    questions=questions,
    responses=responses,
    criteria=["mood_health", "relevance", "emotional_competence"]
)
```

## Configuration

- Model selection via configuration
- Evaluation criteria customizable
- Batch processing for efficiency
- Results saved in structured format