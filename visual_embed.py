import pandas as pd
import umap
import plotly.express as px
import plotly.graph_objects as go

# ─── Configuration ─────────────────────────────────────────────────────────────
TSV_PATH = "/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/embeddings/response_embeddings_with_features.tsv"

# ─── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(TSV_PATH, sep="\t")
emb_cols = [c for c in df.columns if c.startswith("emb_dim_")]
X = df[emb_cols].values
strength = df['strength'].values

# ─── 2D UMAP ────────────────────────────────────────────────────────────────────
reducer_2d = umap.UMAP(n_components=2, random_state=42)
X_2d = reducer_2d.fit_transform(X)
df['UMAP1_2D'] = X_2d[:, 0]
df['UMAP2_2D'] = X_2d[:, 1]

fig_2d = px.scatter(
    df, x='UMAP1_2D', y='UMAP2_2D', color='strength',
    color_continuous_scale='viridis',
    title='2D UMAP of Response Embeddings'
)
fig_2d.show()

# ─── 3D UMAP ────────────────────────────────────────────────────────────────────
reducer_3d = umap.UMAP(n_components=3, random_state=42)
X_3d = reducer_3d.fit_transform(X)
df['UMAP1_3D'] = X_3d[:, 0]
df['UMAP2_3D'] = X_3d[:, 1]
df['UMAP3_3D'] = X_3d[:, 2]

fig_3d = px.scatter_3d(
    df, x='UMAP1_3D', y='UMAP2_3D', z='UMAP3_3D',
    color='strength', color_continuous_scale='viridis',
    title='3D UMAP of Response Embeddings'
)
fig_3d.show()

# ─── Histogram of Strength ─────────────────────────────────────────────────────
fig_hist = px.histogram(df, x='strength', nbins=20, title='Distribution of Strength')
fig_hist.update_layout(xaxis_title='strength', yaxis_title='count')
fig_hist.show()
