import re
from pathlib import Path
from typing import Dict, Any, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import requests

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH_IN  = Path(
    "/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/" \
    "Cogexp/Valence/cognitive_experiment_results007_big.xlsx"
)
DATA_PATH_OUT = Path(
    "/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/"
    "Cogexp/Valence/cognitive_experiment_results007_big_normalised.xlsx"
)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME     = "hf.co/mradermacher/Emollama-7b-GGUF:Q8_0"   # ← your model

_PROMPT_TMPL = (
    "Human:\n"
    "Task: Evaluate the valence intensity of the subject’s mental state based "
    "on the text, assigning it a real-valued score from 0 (most negative) "
    "to 1 (most positive).\n"
    "Text: {text}\n"
    "Intensity Score:\n\n"
    "Assistant:\n"
)
_VALENCE_RE = re.compile(r"([01]\.\d+|0|1)(?=\D|$)")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _score_valence(text: str) -> float | None:
    """Return a single float in ⟦0,1⟧ or None on failure."""
    if not isinstance(text, str) or not text.strip():
        return None

    payload = {"model": MODEL_NAME,
               "prompt": _PROMPT_TMPL.format(text=text.strip()),
               "stream": False}

    try:
        r = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        r.raise_for_status()
        m = _VALENCE_RE.search(r.json().get("response", ""))
        return float(m.group(1)) if m else None
    except (requests.RequestException, ValueError):
        return None

# ──────────────────────────────────────────────────────────────────────────────
# 3.  LOAD ORIGINAL DATA
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH_IN)
assert "pair" in df.columns and "valence_intensity" in df.columns, \
    "Expected at least ‘pair’ and ‘valence_intensity’ columns."

# ──────────────────────────────────────────────────────────────────────────────
# 4.  EXTRACT UNIQUE LATTER SIDES + SCORE THEM
# ──────────────────────────────────────────────────────────────────────────────
def latter_side(pair: str) -> str:
    return pair.split("vs", 1)[-1].strip()     # robust to “ vs ” / “vs”

latter_df = (
    pd.Series(df["pair"].unique(), name="pair")
      .to_frame()
      .assign(latter=lambda d: d["pair"].map(latter_side))
)

print(f"⚙️  Scoring valence for {len(latter_df)} unique latter-side items …")
tqdm.pandas(desc="Valence")
latter_df["latter_valence"] = latter_df["latter"].progress_apply(_score_valence)

# Any failures?  (Optionally alert yourself here)
if latter_df["latter_valence"].isna().any():
    failures = latter_df[latter_df["latter_valence"].isna()]
    print("⚠️  Valence scoring failed for:", failures["latter"].tolist())

# Build a mapping: pair  ➝  valence(latter)
pair2valence = dict(zip(latter_df["pair"], latter_df["latter_valence"]))

# ──────────────────────────────────────────────────────────────────────────────
# 5.  NORMALISE ORIGINAL VALENCE BY LATTER-SIDE VALENCE
# ──────────────────────────────────────────────────────────────────────────────
df["latter_valence"] = df["pair"].map(pair2valence)
df["valence_norm"]   = df["valence_intensity"] / df["latter_valence"]

# ──────────────────────────────────────────────────────────────────────────────
# 6.  (OPTIONAL) SAVE THE ENRICHED DATA
# ──────────────────────────────────────────────────────────────────────────────
df.to_excel(DATA_PATH_OUT, index=False)
print(f"💾  Normalised file written to {DATA_PATH_OUT}")

# ──────────────────────────────────────────────────────────────────────────────
# 7.  RE-PIVOT AND VISUALISE
#     (same layout you used originally: pair × strength)
# ──────────────────────────────────────────────────────────────────────────────
# pivot_norm = (
#     df.pivot_table(index="pair",
#                    columns="strength",
#                    values="valence_norm",
#                    aggfunc="mean")
#       .sort_index()
# )

# plt.figure(figsize=(14, 10))
# sns.heatmap(
#     pivot_norm,
#     annot=True,
#     fmt=".2f",
#     cmap="vlag",
#     vmin=0, vmax=2              # tweak as desired; normed values may exceed 1
# )
# plt.title("Normalised Valence Intensity by Pair and Strength")
# plt.xlabel("Strength")
# plt.ylabel("Cognitive Pair")
# plt.tight_layout()
# plt.show()