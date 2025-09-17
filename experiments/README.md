# 🧪 Experiments

This directory contains experimental configurations, input data, and results from the cognitive representation engineering experiments.

## 📁 Directory Structure

### `cognitive_pairs/`
**Cognitive concept pairs and experimental configurations**

- `positive_concepts.csv` - Positive psychological concepts for vector extraction
- `negative_concepts.csv` - Negative psychological concepts for contrast
- `therapeutic_techniques.csv` - Therapeutic intervention concepts and techniques

### `results/`
**Experimental outputs and generated artifacts**

#### `embeddings/`
- Generated embedding vectors
- Representation engineering vectors
- Cached embedding results

#### `visualizations/`
- Generated HTML visualizations
- Interactive plots and charts
- Export-ready figures

## 🎯 Experimental Design

### Concept Pairs
The experiments use contrastive pairs spanning different psychological domains:

**Basic Psychological Needs (Intrinsic Motivation):**
- Alienation/Isolation ↔ Community/Engagement
- Powerlessness/Helplessness ↔ Power/Influence
- Stagnation/Complacency ↔ Learning/Curiosity

**Socially Constructed Goals (Extrinsic Motivation):**
- Obscurity/Anonymity ↔ Fame/Recognition
- Neglect/Disregard ↔ Beauty/Appearance
- Selfishness/Greed ↔ Generosity/Altruism

### Vector Extraction Process
1. **Contrastive Prompting**: Generate positive/negative prompt pairs
2. **Activation Capture**: Record model activations during generation
3. **PCA Analysis**: Extract directional vectors from activation differences
4. **Vector Application**: Apply vectors at varying intensities (0-1.5x)

### Evaluation Framework
- **25 Depression-related Questions**: Covering mood, activity, cognition, coping
- **Multi-dimensional Scoring**: Mood health, relevance, emotional competence, adaptiveness
- **Statistical Analysis**: Regression, correlation, effectiveness patterns

## 📊 Data Files

### Concept Files Format
```csv
concept,category,description,examples
Community,intrinsic,"Social connection and belonging","friendship, support, belonging"
Isolation,intrinsic,"Social disconnection and loneliness","lonely, alone, excluded"
```

### Results Format
- **Embeddings**: NumPy arrays or pickled tensors
- **Scores**: JSON/CSV with evaluation metrics
- **Visualizations**: HTML interactive plots

## 🔗 Integration

This directory integrates with:
- **Notebooks**: Load configurations and analyze results
- **Source Code**: Input data for processing scripts
- **Documentation**: Referenced in methodology docs

## 📋 Experimental Protocols

1. **Reproducibility**: All random seeds documented
2. **Version Control**: Model versions and parameters tracked
3. **Validation**: Cross-validation and statistical significance testing
4. **Documentation**: Detailed experimental logs and metadata