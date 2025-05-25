"""
Clustering Explorer – hierarchical, K-Means & DBSCAN
Automatically opens each visualisation in your browser.
"""

from pathlib import Path
import webbrowser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA         # ← swap for UMAP if installed
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# ─── PATHS ──────────────────────────────────────────────────────────────────────
EMB_FILE = Path("/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/embeddings/response_embeddings_with_features.tsv")  # adjust if needed
HTML_DIR = Path("cluster_viz_out").resolve()
HTML_DIR.mkdir(exist_ok=True)

# ─── 1 · Load data & build feature matrix ──────────────────────────────────────
df = pd.read_csv(EMB_FILE, sep="\t")
emb_cols = [c for c in df.columns if c.startswith("emb_dim_")]

# full feature-set: embeddings + numeric + categorical-encoded
features = (
    df[emb_cols]
    .assign(strength=df["strength"],
            question_index=df["question_index"],
            pair_enc=LabelEncoder().fit_transform(df["pair"].astype(str)))
)
X = StandardScaler(with_mean=False).fit_transform(features.values)      # cos-geometry

# ─── 2 · Dim-reduce *once* for all scatter plots ───────────────────────────────
coords2d = PCA(n_components=2, random_state=42).fit_transform(X)        # fast & built-in

# ─── 3 · Cluster with three algorithms ─────────────────────────────────────────
models = {
    "Hierarchical (Ward, k=6)": AgglomerativeClustering(n_clusters=6, linkage="ward"),
    "KMeans (k=6)":             KMeans(n_clusters=6, random_state=42, n_init="auto"),
    "DBSCAN":                   DBSCAN(eps=3.0, min_samples=10)
}

labels_dict = {name: model.fit_predict(X) for name, model in models.items()}

# ─── 4 · Visualise clusters – one figure per method ────────────────────────────
for name, labels in labels_dict.items():
    fig = plt.figure()
    plt.scatter(coords2d[:, 0], coords2d[:, 1], c=labels, s=6, alpha=.8)
    plt.xlabel("PC-1"); plt.ylabel("PC-2"); plt.title(name)
    out_png = HTML_DIR / f"{name.replace(' ', '_').replace('/', '')}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    webbrowser.open_new_tab(out_png.as_uri())      # instant preview

# ─── 5 · Hierarchical dendrogram (truncated to depth=25) ───────────────────────
linkage_matrix = linkage(X, method="ward")
fig = plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode="level", p=25, no_labels=True)
plt.title("Ward Hierarchical Dendrogram (top 25 merges)")
plt.xlabel("Sample index or (cluster size)"); plt.ylabel("Distance")
out_png = HTML_DIR / "hierarchical_dendrogram.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
webbrowser.open_new_tab(out_png.as_uri())
print("✓ All visuals exported to", HTML_DIR)
