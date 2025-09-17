# 🔧 Source Code

This directory contains all Python source code organized by functionality. All scripts follow consistent naming conventions and are well-documented.

## 📁 Module Structure

### `data_processing/`
**Data ingestion, embedding generation, and preprocessing**
- `openai_embeddings.py` - OpenAI API embedding generation with rate limiting
- `clustering_explorer.py` - Clustering algorithms and data exploration tools

### `evaluation/`
**Model evaluation and assessment tools**
- `llm_judge_base.py` - Base LLM judge functionality (local models)
- `llm_judge_local.py` - Local model evaluation using Ollama
- `llm_judge_openai.py` - OpenAI-based evaluation pipeline
- `llm_judge_enhanced.py` - Enhanced evaluation with multiple metrics
- `question_analyzer.py` - Question analysis and response processing

### `visualization/`
**Data visualization and result presentation**
- `cluster_3d_viewer.py` - 3D cluster visualization with UMAP
- `embedding_visualizer.py` - Embedding space visualization tools
- `valence_plotter.py` - Valence and sentiment plotting utilities

### `utils/`
**Common utilities and helper functions**
- `valence_utils.py` - Valence normalization and processing utilities

## 🏗️ Code Architecture

### Design Principles
- **Modular**: Each script has a single, clear responsibility
- **Documented**: Every function includes docstrings and type hints
- **Configurable**: Parameters exposed through constants and configuration
- **Reusable**: Functions designed for use across multiple notebooks

### Naming Conventions
- **Files**: `snake_case.py`
- **Functions**: `snake_case()`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`

### Dependencies
Common dependencies across modules:
```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
```

## 🚀 Usage Examples

### Data Processing
```python
from src.data_processing.openai_embeddings import generate_embeddings
embeddings = generate_embeddings(texts, model="text-embedding-3-large")
```

### Evaluation
```python
from src.evaluation.llm_judge_openai import evaluate_responses
scores = evaluate_responses(questions, responses, criteria)
```

### Visualization
```python
from src.visualization.cluster_3d_viewer import create_3d_cluster_plot
fig = create_3d_cluster_plot(embeddings, labels, title="Concept Clusters")
```

## 🔗 Integration

These modules are designed to work seamlessly with:
- **Notebooks**: Import and use functions directly
- **Scripts**: Can be run standalone with proper configuration
- **Pipeline**: Modular design allows easy pipeline construction

## 📋 Development Standards

- **Type hints**: All function parameters and returns typed
- **Error handling**: Comprehensive exception handling
- **Logging**: Structured logging for debugging and monitoring
- **Testing**: Unit tests for critical functions (when applicable)