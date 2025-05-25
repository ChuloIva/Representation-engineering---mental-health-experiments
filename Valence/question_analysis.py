import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── 1. Load data ───────────────────────────────────────────────────────────────
file_path = Path("/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/Valence/cognitive_experiment_results007_big_with_questions_evaluated.xlsx")
df = pd.read_excel(file_path)

# ── 2. Metrics to visualise ────────────────────────────────────────────────────
metrics = [
    "relevancy_score",
    "mood_health",
    "emotional_competence",
    "thought_adaptiveness",
]

# ── 3. Make a heat‑map per metric ──────────────────────────────────────────────
plt.style.use("default")           # keep matplotlib’s default style
figsize = (14, 10)                 # same size as your valence plot
cmap = "vlag"                      # diverging palette

for metric in metrics:
    pivot = df.pivot_table(
        index="pair",
        columns="strength",
        values=metric,
        aggfunc="mean",
    )

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )
    plt.title(
        f"Average {metric.replace('_', ' ').title()} by Pair and Strength (All Questions)",
        fontsize=14,
    )
    plt.xlabel("Strength")
    plt.ylabel("Cognitive Pair")
    plt.tight_layout()
    plt.show()
