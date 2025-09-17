# 📊 Analysis Notebooks

This directory contains all Jupyter notebooks for the cognitive representation engineering experiments, organized by analysis type and execution order.

## 📁 Directory Structure

### `01-main-experiments/`
**Main experimental analysis and representation engineering**
- `representation_engineering_main.ipynb` - Core experiments testing cognitive pairs on fine-tuned model

### `02-baseline-analysis/`
**Baseline model evaluation and comparison**
- `baseline_model_analysis.ipynb` - Analysis of unmodified model responses for comparison

### `03-therapeutic-analysis/`
**Therapeutic concept and intervention analysis**
- `therapeutic_concept_analysis.ipynb` - Deep dive into therapeutic effectiveness of different concepts

### `04-judge-evaluation/`
**LLM-as-judge evaluation methodology**
- `llm_judge_evaluation.ipynb` - Evaluation metrics and judge model performance analysis

### `05-regression-analysis/`
**Statistical analysis and pattern identification**
- `linear_regression_analysis.ipynb` - Regression analysis of concept effectiveness patterns

### `06-visualization/`
**Data visualization and result presentation**
- `valence_normalization_viz.ipynb` - Valence normalization and visualization methods
- `general_analysis_viz.ipynb` - General analysis visualizations and exploratory plots

## 🚀 Quick Start

1. **Recommended execution order**: Follow the numbered directories (01 → 06)
2. **Main results**: Start with `01-main-experiments/representation_engineering_main.ipynb`
3. **Dependencies**: Ensure all required packages are installed (see main README)
4. **Data**: All notebooks expect data to be in the `../data/` directory

## 📋 Notebook Standards

- **Cell outputs cleared**: For GitHub compatibility
- **Markdown documentation**: Each section explained
- **Reproducible**: All random seeds set where applicable
- **Self-contained**: Each notebook can run independently with proper data setup

## 🔗 Related Directories

- `../src/` - Source code used by notebooks
- `../data/` - Input data files
- `../experiments/` - Experiment configurations and results
- `../assets/` - Generated figures and screenshots