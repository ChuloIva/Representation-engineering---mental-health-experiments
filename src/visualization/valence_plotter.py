"""
3-D UMAP → Plotly visualiser for OpenAI response embeddings
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import pandas as pd                       # dataframe IO                     (TSV)      \
from pathlib import Path                                                               # |
import umap                                                                      # ⇠ UMAP
import plotly.express as px                                                     # ⇠ 3-D scatter
from sklearn.preprocessing import StandardScaler                                # normalise
from sklearn.metrics.pairwise import cosine_similarity                          # option

# ── Config ─────────────────────────────────────────────────────────────────────
EMB_FILE = Path(
    "/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/"
    "Cogexp/embeddings/response_embeddings_with_features.tsv"
)
DIM_COLS = [c for c in pd.read_csv(EMB_FILE, nrows=0, sep="\t").columns if c.startswith("emb_dim_")]

COLOR_COL   = "pair"                 # categorical colouring (string or int)
SIZE_COL    = "strength"             # numeric bubble sizes
LABEL_COL   = "response"             # shows on hover

UMAP_NCOMP  = 3                      # 3-D
UMAP_NN     = 25                     # neighbours
UMAP_MIND   = .1                     # min_dist

OUTPUT_HTML = "embedding_cloud.html"

# ── Load TSV ───────────────────────────────────────────────────────────────────
df = pd.read_csv(EMB_FILE, sep="\t")                                           # pandas read_csv supports TSV via sep. :contentReference[oaicite:2]{index=2}
X  = df[DIM_COLS].values

# ── Optional: length-normalise for cosine geometry ─────────────────────────────
X = StandardScaler(with_mean=False).fit_transform(X)                            # keeps sparsity

# ── Reduce with UMAP ───────────────────────────────────────────────────────────
reducer = umap.UMAP(
    n_components=UMAP_NCOMP,
    n_neighbors=UMAP_NN,
    min_dist=UMAP_MIND,
    metric="cosine",
    random_state=42,
)
coords = reducer.fit_transform(X)

df[["x", "y", "z"]] = coords                                                    # attach low-D axes

# ── Plot with Plotly Express ───────────────────────────────────────────────────
fig = px.scatter_3d(
    df,
    x="x", y="y", z="z",
    hover_name=LABEL_COL,
    color=df[COLOR_COL].astype(str),                                            # ensure discrete palette. :contentReference[oaicite:3]{index=3}
    size=SIZE_COL,
    opacity=0.75,
    height=800,
)
fig.update_traces(marker=dict(line=dict(width=0)))  # remove black borders
fig.update_layout(scene=dict(aspectmode="data"))

fig.write_html(OUTPUT_HTML, auto_open=True)                                     # opens browser
