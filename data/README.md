# 📊 Data Directory

This directory contains experimental data files used throughout the project. **Note: All data files are in .gitignore to keep the repository lightweight while preserving local analysis capability.**

## 📁 Expected Structure

```
data/
├── raw/                          # Original, unprocessed data
│   ├── cognitive_experiment_*.xlsx     # Raw experimental results
│   ├── baseline_answers*.xlsx          # Baseline model responses
│   └── ...
├── processed/                    # Cleaned and processed data
│   ├── embeddings/                     # Generated embedding files
│   ├── scores/                         # Evaluation scores and metrics
│   └── aggregated/                     # Summary and aggregated datasets
└── external/                     # External datasets and references
    ├── depression_datasets/            # Training/validation data
    └── psychological_scales/           # Standard psychological measures
```

## 📋 Data Files

### Raw Experimental Data
- `cognitive_experiment_results*.xlsx` - Raw experimental results from representation engineering
- `baseline_answers*.xlsx` - Baseline model responses without interventions
- `cognitive_experiment_therapy_*.xlsx` - Therapeutic intervention results

### Generated Data
- **Embeddings**: Vector representations from OpenAI and local models
- **Evaluation Scores**: LLM-as-judge evaluation results
- **Statistical Results**: Regression analysis, correlation matrices

## 🔒 Data Privacy & Ethics

- **No Personal Data**: All datasets are synthetic or anonymized
- **Model Outputs**: Generated responses, not human subjects
- **Ethical Use**: Research purposes only, not clinical application

## 📊 Data Formats

### Excel Files (.xlsx)
- **Experiment Results**: Questions, responses, scores, metadata
- **Columns**: `question_id`, `response`, `vector_strength`, `concept_pair`, `scores`

### CSV Files (.csv)
- **Processed Data**: Clean, analysis-ready formats
- **Concept Definitions**: Psychological concept pairs and categories

### Binary Files
- **Embeddings**: `.pkl`, `.npy` - NumPy arrays and pickled objects
- **Models**: `.pt`, `.pth` - PyTorch model states (if applicable)

## 🔧 Data Processing Pipeline

1. **Raw Data Ingestion** → Load experimental results
2. **Cleaning & Validation** → Remove invalid responses, normalize scores
3. **Feature Engineering** → Create derived metrics, aggregate scores
4. **Embedding Generation** → Create vector representations
5. **Statistical Analysis** → Correlation, regression, significance testing

## 📈 Usage in Analysis

### Loading Data
```python
import pandas as pd
from pathlib import Path

# Load experimental results
data_dir = Path("../data")
results = pd.read_excel(data_dir / "cognitive_experiment_results007_big.xlsx")

# Load processed embeddings
embeddings = np.load(data_dir / "processed/embeddings/concept_embeddings.npy")
```

### Data Validation
- **Completeness**: Check for missing values
- **Consistency**: Validate score ranges and formats
- **Quality**: Remove outliers and invalid responses

## 🔗 Integration

This data directory integrates with:
- **Notebooks**: Primary data source for all analysis
- **Source Code**: Input for processing and visualization scripts
- **Experiments**: Results storage and configuration data

## 📋 Data Management

- **Version Control**: Data files excluded from git but documented
- **Backup**: Local backups recommended for important datasets
- **Reproducibility**: Processing scripts document exact data transformations
- **Documentation**: Metadata files describe data structure and provenance