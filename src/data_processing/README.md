# 📊 Data Processing

Scripts for data ingestion, preprocessing, and embedding generation.

## Scripts

### `openai_embeddings.py`
**OpenAI API embedding generation with rate limiting**
- Generates embeddings using OpenAI's text-embedding models
- Implements rate limiting for API compliance
- Batch processing for efficiency
- Error handling and retry logic

**Key Functions:**
- `generate_embeddings()` - Main embedding generation
- `batch_embed()` - Batch processing with rate limits
- `save_embeddings()` - Save to various formats

### `clustering_explorer.py`
**Clustering algorithms and data exploration**
- UMAP dimensionality reduction
- K-means, hierarchical, and DBSCAN clustering
- Quality metrics (silhouette score, etc.)
- Interactive visualization generation

**Key Functions:**
- `reduce_dimensions()` - UMAP reduction
- `cluster_data()` - Multiple clustering algorithms
- `evaluate_clusters()` - Quality metrics
- `create_interactive_plot()` - 3D visualizations

## Usage

```python
from data_processing.openai_embeddings import generate_embeddings
from data_processing.clustering_explorer import reduce_dimensions, cluster_data

# Generate embeddings
embeddings = generate_embeddings(text_list)

# Reduce dimensions and cluster
reduced_data = reduce_dimensions(embeddings)
clusters = cluster_data(reduced_data)
```

## Configuration

- OpenAI API key required in environment
- Rate limits configured for API tier
- Output paths configurable via constants