"""
umap_cluster_viewer.py
---------------------------------------------
UMAP dimensionality reduction → three clustering algorithms →
quality metrics → interactive Plotly 3-D galaxy (auto-opens).
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import webbrowser
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score,
                             calinski_harabasz_score,
                             davies_bouldin_score)
import umap                                   # UMAP dimensionality reducer  :contentReference[oaicite:4]{index=4}
import plotly.express as px                   # 3-D scatter  :contentReference[oaicite:5]{index=5}

# ── Configuration ─────────────────────────────────────────────────────────────
TSV_PATH   = Path("/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/embeddings/response_embeddings_with_features.tsv")  # adjust if needed
HTML_FILE  = Path("umap_kmeans_gallery.html").resolve()
N_CLUSTERS = 6          # k for K-Means & hierarchical

# ── 1 · Load data & build feature matrix ──────────────────────────────────────
df = pd.read_csv(TSV_PATH, sep="\t")
emb_cols = [c for c in df.columns if c.startswith("emb_dim_")]

features = (
    df[emb_cols]
    .assign(strength=df["strength"],
            question_index=df["question_index"],
            pair_enc=LabelEncoder().fit_transform(df["pair"].astype(str)))
)

X = StandardScaler(with_mean=False).fit_transform(features.values)  # preserves cosine geometry

# ── 2 · 3-D UMAP reduction ────────────────────────────────────────────────────
umap3 = umap.UMAP(
    n_components=3, n_neighbors=25, min_dist=0.1,
    metric="cosine", random_state=42
)  # parameters explained in docs :contentReference[oaicite:6]{index=6}
coords3d = umap3.fit_transform(X)
df[["ux", "uy", "uz"]] = coords3d

# ── 3 · Clustering algorithms ────────────────────────────────────────────────
df["cluster_kmeans"] = KMeans(n_clusters=N_CLUSTERS, n_init="auto",
                              random_state=42).fit_predict(X)  # :contentReference[oaicite:7]{index=7}
df["cluster_hier"]   = AgglomerativeClustering(n_clusters=N_CLUSTERS,
                                               linkage="ward").fit_predict(X)
df["cluster_dbscan"] = DBSCAN(eps=3.0, min_samples=10).fit_predict(X)  # tune eps

# ── 4 · Quality metrics (printed) ─────────────────────────────────────────────
def quality(name, labels):
    if len(set(labels)) < 2 or (len(set(labels)) == 1 and -1 in labels):
        return f"{name}: not enough clusters"
    return (f"{name}: silhouette={silhouette_score(X, labels):.3f}  "
            f"CH={calinski_harabasz_score(X, labels):.1f}  "
            f"DB={davies_bouldin_score(X, labels):.3f}")

print("\n".join([
    quality("K-Means",      df["cluster_kmeans"]),
    quality("Hierarchical", df["cluster_hier"]),
    quality("DBSCAN",       df["cluster_dbscan"]),
]))
# Metrics docs: silhouette :contentReference[oaicite:8]{index=8}, CH :contentReference[oaicite:9]{index=9}, DB :contentReference[oaicite:10]{index=10}

# ── 5 · Interactive Plotly galaxy ─────────────────────────────────────────────
COLOR_FIELD = "cluster_kmeans"   # change to 'cluster_hier' or 'cluster_dbscan'

fig = px.scatter_3d(
    df, x="ux", y="uy", z="uz",
    color=df[COLOR_FIELD].astype(str),   # colours by chosen cluster labels
    hover_name="response",
    size="strength",
    opacity=0.75,
    height=800,
    title=f"UMAP + {COLOR_FIELD} (k={N_CLUSTERS})"
)  # plotly scatter_3d guide :contentReference[oaicite:11]{index=11}
fig.update_traces(marker=dict(line=dict(width=0)))
fig.update_layout(scene=dict(aspectmode="data"))
fig.write_html(HTML_FILE)
webbrowser.open_new_tab(HTML_FILE.as_uri())     # auto-launch  :contentReference[oaicite:12]{index=12}
print(f"✓ interactive viewer → {HTML_FILE}")
