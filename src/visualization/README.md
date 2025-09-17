# 📈 Visualization

Scripts for creating interactive and static visualizations of results.

## Scripts

### `cluster_3d_viewer.py`
**3D cluster visualization with UMAP**
- Interactive 3D plots using Plotly
- UMAP dimensionality reduction
- Multiple clustering algorithm comparison
- Quality metrics visualization
- Auto-opens in browser

**Features:**
- Real-time cluster interaction
- Quality metrics overlay
- Customizable color schemes
- Export capabilities

### `embedding_visualizer.py`
**Embedding space visualization tools**
- 2D/3D embedding projections
- Concept relationship visualization
- Vector arithmetic visualization
- Semantic space exploration

### `valence_plotter.py`
**Valence and sentiment plotting utilities**
- Valence score distributions
- Concept effectiveness comparisons
- Statistical analysis plots
- Regression visualization

## Visualization Types

### Interactive Plots
- **3D Cluster Galaxy**: Interactive exploration of concept clusters
- **Embedding Spaces**: Navigate semantic relationships
- **Performance Dashboards**: Real-time metric monitoring

### Static Analysis
- **Heatmaps**: Concept effectiveness patterns
- **Line Plots**: Performance across vector strengths
- **Bar Charts**: Comparative analysis
- **Regression Plots**: Statistical relationships

## Usage

```python
from visualization.cluster_3d_viewer import create_3d_cluster_plot
from visualization.valence_plotter import plot_effectiveness_comparison

# Create interactive 3D visualization
fig = create_3d_cluster_plot(embeddings, labels, title="Cognitive Concepts")

# Plot concept effectiveness
effectiveness_plot = plot_effectiveness_comparison(results_df)
```

## Output Formats

- **HTML**: Interactive Plotly visualizations
- **PNG/SVG**: High-quality static exports
- **PDF**: Publication-ready figures
- **JSON**: Plotly figure specifications

## Dependencies

- `plotly` - Interactive visualizations
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualizations
- `umap-learn` - Dimensionality reduction