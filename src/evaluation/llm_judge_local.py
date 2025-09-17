"""
Therapeutic Q&A Evaluator — **Local‑LLM Edition**
=================================================

Evaluate each *question–answer* pair in a dataset with **any Ollama‑served model** and append
four diagnostic columns:

* `relevancy_score` – How relevant the answer is to the question (0 = irrelevant, 1 = direct)
* `mood_health` – Emotional wellbeing and stability indicated in response (0 = negative, 1 = positive)
* `emotional_competence` – Ability to identify and express emotions (0 = poor, 1 = excellent)
* `thought_adaptiveness` – Presence of healthy cognitive patterns (0 = maladaptive, 1 = adaptive)

The script talks to a **locally running Ollama API** (default: `http://localhost:11434/api/generate`).
Just change `MODEL_NAME` to any model you have pulled with `ollama pull …`.

Quick start
-----------
```python
from qa_therapy_evaluator_local import run_qa_evaluation

results = run_qa_evaluation(
    input_path="/mnt/data/therapy_dataset.xlsx",
    output_path="/mnt/data/with_scores.xlsx",   # optional
    question_col="question",                    # change if different
    answer_col="response"                       # change if different
)
results.head()
```

Dependencies
------------
```bash
pip install pandas tqdm requests openpyxl
```
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import requests
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:9b"  # ← Replace with your local model name

SYSTEM_PROMPT = (
    "You are a licensed cognitive‑behavioral therapist specializing in comprehensive mental health "
    "assessment. For each question/answer pair, produce **ONLY** a JSON object with these keys:\n\n"
    "  relevancy_score – float 0‑1: how directly the answer addresses the question's therapeutic intent\n"
    "  mood_health – float 0‑1: emotional wellbeing and stability indicated in response\n"
    "  emotional_competence – float 0‑1: ability to identify and express emotions effectively\n"
    "  thought_adaptiveness – float 0‑1: presence of healthy, adaptive cognitive patterns\n\n"
    "Return strictly valid JSON with exactly those four keys and numeric values between 0 and 1."
)

USER_PROMPT_TMPL = (
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Provide your evaluation now as JSON."
)

# Capture the first JSON object in the response (the model might wrap it in markdown)
_JSON_RE = re.compile(r"\{[\s\S]*?\}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _eval_pair(question: str, answer: str) -> Dict[str, Any]:
    """Evaluate a single Q‑A pair via Ollama and return a dict with four scores."""
    if not isinstance(question, str) or not isinstance(answer, str):
        return {
            "relevancy_score": None,
            "mood_health": None,
            "emotional_competence": None,
            "thought_adaptiveness": None,
        }

    prompt = (
        "System:\n" + SYSTEM_PROMPT + "\n\n" +
        "Human:\n" + USER_PROMPT_TMPL.format(question=question.strip(), answer=answer.strip()) + "\n\n" +
        "Assistant:\n"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json().get("response", "")

        # Extract JSON
        match = _JSON_RE.search(content)
        if not match:
            raise ValueError("No JSON object returned by model.")
        data = json.loads(match.group(0))

        # Ensure all four keys are present
        return {
            "relevancy_score": float(data.get("relevancy_score")) if data.get("relevancy_score") is not None else None,
            "mood_health": float(data.get("mood_health")) if data.get("mood_health") is not None else None,
            "emotional_competence": float(data.get("emotional_competence")) if data.get("emotional_competence") is not None else None,
            "thought_adaptiveness": float(data.get("thought_adaptiveness")) if data.get("thought_adaptiveness") is not None else None,
        }
    except Exception as exc:
        print(f"⚠️  Evaluation failed for a row: {exc}")
        return {
            "relevancy_score": None,
            "mood_health": None,
            "emotional_competence": None,
            "thought_adaptiveness": None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_qa_evaluation(
    *,
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    question_col: str = "question",
    answer_col: str = "response",
) -> pd.DataFrame:
    """Read a CSV/Excel file, score each Q‑A row, optionally save, and return DataFrame."""
    input_path = Path(input_path)
    ext = input_path.suffix.lower()
    if ext == ".xlsx":
        df = pd.read_excel(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("input_path must be .xlsx or .csv")

    pairs: List[Tuple[str, str]] = df[[question_col, answer_col]].astype(str).values.tolist()

    print(f"🔍 Evaluating {len(pairs)} question–answer pairs with model '{MODEL_NAME}'")
    tqdm.pandas(desc="QA Eval")
    results_series = pd.Series(pairs).progress_apply(lambda p: _eval_pair(p[0], p[1]))
    results_df = pd.DataFrame(results_series.tolist())

    df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

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


# Sample usage when executed directly – edit as needed
if __name__ == "__main__":
    EXAMPLE_INPUT = "/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/Valence/Data/cognitive_experiment_therapy_003_big_with_questions.xlsx"
    EXAMPLE_OUTPUT = "/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/Valence/Data/cognitive_experiment_therapy_003_big_with_questions_eval_local.xlsx"

    run_qa_evaluation(
        input_path=EXAMPLE_INPUT,
        output_path=EXAMPLE_OUTPUT,
        question_col="question",
        answer_col="response",
    )
