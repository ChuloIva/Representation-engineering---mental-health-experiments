"""
Therapeutic Q&A Evaluator (OpenAI‑only)
=====================================

Evaluates each **question–answer pair** in a dataset with OpenAI Chat Completions
and appends four diagnostic columns:

* `relevancy_score` – How relevant the answer is to the question (0 = irrelevant, 1 = direct)
* `mood_health` – Emotional wellbeing and stability indicated in response (0 = negative, 1 = positive)
* `emotional_competence` – Ability to identify and express emotions (0 = poor, 1 = excellent)
* `thought_adaptiveness` – Presence of healthy cognitive patterns (0 = maladaptive, 1 = adaptive)

Quick start
-----------
```python
from qa_therapy_evaluator import run_qa_evaluation

# Make sure you exported your key first:
#   export OPENAI_API_KEY="sk‑…"

results = run_qa_evaluation(
    input_path="/mnt/data/your_dataset.xlsx",
    output_path="/mnt/data/with_scores.xlsx",   # optional
    question_col="question",                    # change if your header differs
    answer_col="response"                       # change if your header differs
)
results.head()
```

Dependencies
------------
```bash
pip install pandas openpyxl tqdm python-dotenv "openai>=1.14.0"
```
"""
from __future__ import annotations

import os
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Load env & configure OpenAI client (>= 1.x)
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

try:
    from openai import AsyncOpenAI  # type: ignore
except ImportError as exc:
    raise ImportError(
        "The latest OpenAI Python library is required. Install with\n"
        "    pip install --upgrade \"openai>=1.14.0\""
    ) from exc

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError(
        "Please set your OpenAI API key as an environment variable before running this script.\n"
        "Example (in your terminal):\n"
        "  export OPENAI_API_KEY='sk-...'"
    )

# Instantiate once and share across all async calls
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ──────────────────────────────────────────────────────────────────────────────
# Model & prompts
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "gpt-4o"

SYSTEM_PROMPT = (
    "You are a licensed cognitive-behavioral therapist specializing in comprehensive mental health assessment.\n"
    "For each question/answer pair, produce a JSON object with these keys:\n\n"
    "  relevancy_score – float 0‑1: how directly the answer addresses the question's therapeutic intent\n"
    "  mood_health – float 0‑1: emotional wellbeing and stability indicated in response\n"
    "  emotional_competence – float 0‑1: ability to identify and express emotions effectively\n"
    "  thought_adaptiveness – float 0‑1: presence of healthy, adaptive cognitive patterns\n\n"
    "Return ONLY valid JSON."
)

USER_PROMPT_TMPL = (
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Provide your evaluation now."
)

# ──────────────────────────────────────────────────────────────────────────────
# Async helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _eval_pair(question: str, answer: str) -> Dict[str, Any]:
    """Evaluate a single Q‑A pair and return the parsed JSON with all four scores."""
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TMPL.format(
                        question=question.strip(), answer=answer.strip()
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        # Log once per failed row and fall back to None values so dataframe aligns
        print(f"⚠️  Could not parse JSON for a row: {exc}")
        return {
            "relevancy_score": None,
            "mood_health": None,
            "emotional_competence": None,
            "thought_adaptiveness": None,
        }


async def _eval_all(
    pairs: List[Tuple[str, str]], concurrency: int = 10
) -> List[Dict[str, Any]]:
    """Evaluate many Q‑A pairs concurrently with an asyncio semaphore limit."""
    sem = asyncio.Semaphore(concurrency)

    async def _worker(q: str, a: str) -> Dict[str, Any]:
        async with sem:
            return await _eval_pair(q, a)

    tasks = [_worker(q, a) for q, a in pairs]
    return await asyncio.gather(*tasks)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_qa_evaluation(
    *,
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    question_col: str = "question",
    answer_col: str = "response",
    concurrency: int = 10,
) -> pd.DataFrame:
    """Read a CSV/Excel file, score each Q‑A row, and optionally save the results."""

    input_path = Path(input_path)
    ext = input_path.suffix.lower()
    if ext == ".xlsx":
        df = pd.read_excel(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("input_path must be .xlsx or .csv")

    pairs = df[[question_col, answer_col]].astype(str).values.tolist()

    print("🔍 Evaluating", len(pairs), "question–answer pairs with", MODEL_NAME)
    progress = tqdm(total=len(pairs), desc="Rows processed")

    async def _run() -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        CHUNK = concurrency * 2  # send in manageable batches to avoid rate limits
        for i in range(0, len(pairs), CHUNK):
            chunk = pairs[i : i + CHUNK]
            chunk_results = await _eval_all(chunk, concurrency)
            results.extend(chunk_results)
            progress.update(len(chunk))
        return results

    results = asyncio.run(_run())
    progress.close()

    # Merge back into the original dataframe
    results_df = pd.DataFrame(results)
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


# Allow execution as a script for quick ad‑hoc runs
# if __name__ == "__main__":
#     import argparse

#     ap = argparse.ArgumentParser(description="Score therapeutic Q‑A datasets with OpenAI chat completions.")
#     ap.add_argument("input_path", help="Path to input .csv or .xlsx file")
#     ap.add_argument("--output", dest="output_path", help="Optional path to save with scores")
#     ap.add_argument("--qcol", dest="question_col", default="question", help="Name of the question column")
#     ap.add_argument("--acol", dest="answer_col", default="response", help="Name of the answer column")
#     ap.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")

#     args = ap.parse_args()

#     run_qa_evaluation(
#         input_path=args.input_path,
#         output_path=args.output_path,
#         question_col=args.question_col,
#         answer_col=args.answer_col,
#         concurrency=args.concurrency,
#     )

run_qa_evaluation(
    input_path="/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/Valence/cognitive_experiment_results007_big_with_questions.xlsx",
    output_path="/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/Valence/cognitive_experiment_results007_big_with_questions_evaluated.xlsx",
    question_col="question",
    answer_col="response",
    concurrency=10
)