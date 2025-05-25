import requests
import re
from pathlib import Path
from typing import Dict, Any, Union

import pandas as pd
from tqdm import tqdm

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "hf.co/mradermacher/Emollama-7b-GGUF:Q8_0"  # Replace with your desired model



_PROMPT_TMPL = (
    "Human:\n"
    "Task: Evaluate the valence intensity of the subjects mental state based on the text, "
    "assigning it a real‑valued score from 0 (most negative) to 1 (most positive).\n"
    "Text: {text}\n"
    "Intensity Score:\n\n"
    "Assistant:\n"
)
_VALENCE_RE = re.compile(r"([01]\.\d+|0|1)(?=\D|$)")  # captures first float in 0‑1 range



def _score_valence(text: str) -> Dict[str, Any]:
    """Return `{valence_intensity: float|None}` for one string."""
    if not isinstance(text, str) or not text.strip():
        return {"valence_intensity": None}

    prompt = _PROMPT_TMPL.format(text=text.strip())
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        generated = result.get("response", "")
        m = _VALENCE_RE.search(generated)
        return {"valence_intensity": float(m.group(1)) if m else None}
    except (requests.RequestException, ValueError):
        return {"valence_intensity": None}


def run_valence_pipeline(
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    *,
    response_col: str = "response",
) -> pd.DataFrame:
    """Load an Excel/CSV, append `valence_intensity`, optionally save, and return DataFrame.

    Parameters
    ----------
    input_path : str | Path
        Path to the original dataset (.xlsx or .csv).
    output_path : str | Path | None, default None
        Where to save the enriched file (same format as input). If None, nothing is saved.
    response_col : str, default "response"
        Column containing free‑text responses to score.

    Returns
    -------
    pd.DataFrame
        Enriched dataframe with an extra `valence_intensity` column.
    """
    input_path = Path(input_path)
    ext = input_path.suffix.lower()
    if ext == ".xlsx":
        df = pd.read_excel(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Unsupported file type – must be .xlsx or .csv")

    print("▶️  Scoring valence intensity…")
    tqdm.pandas(desc="Valence")
    valence_series = df[response_col].astype(str).progress_apply(_score_valence).apply(pd.Series)
    df = pd.concat([df, valence_series], axis=1)

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix.lower() == ".xlsx":
            df.to_excel(output_path, index=False)
        elif output_path.suffix.lower() == ".csv":
            df.to_csv(output_path, index=False)
        else:
            raise ValueError("output_path must end with .xlsx or .csv")
        print(f"💾 Saved enriched file to {output_path}")

    return df

df = run_valence_pipeline(
    input_path="/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/data/cognitive_experiment_results007_big.xlsx",
    output_path="/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/Valence/cognitive_experiment_results007_big.xlsx"
)
df.head()