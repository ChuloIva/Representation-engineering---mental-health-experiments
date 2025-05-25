"""
Therapeutic Q&A Evaluator (OpenAI‑only)
=====================================

This notebook‑friendly module evaluates each **question–answer pair** in a dataset
with OpenAI Chat Completions and appends three new columns:

* `therapy_score`   – How therapeutically constructive/helpful the answer is   (0 = poor, 1 = excellent)
* `relevancy_score` – How relevant the answer is to the question               (0 = irrelevant, 1 = direct)
* `evaluation`      – A short textual justification/explanation (optional)

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
pip install pandas openpyxl tqdm openai>=1.3.8
```
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Union

import pandas as pd
from tqdm import tqdm
import openai

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI model & prompts
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "gpt-4o-mini"  # flip to gpt-4o for highest quality

SYSTEM_PROMPT = (
    "You are a licensed cognitive‑behavioral therapist tasked with evaluating \n"
    "client answers to reflective questions. For each question/answer pair, \n"
    "produce a JSON object with these keys:\n"
    "  therapy_score    – float 0‑1: how therapeutically helpful the answer is\n"
    "  relevancy_score  – float 0‑1: how directly the answer addresses the question\n"
    "  evaluation       – one‑sentence justification (max 30 words)\n"
    "Return ONLY valid JSON."
)

USER_PROMPT_TMPL = (
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Provide your evaluation now."
)


async def _eval_pair(question: str, answer: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TMPL.format(question=question.strip(), answer=answer.strip())},
    ]
    response = await openai.ChatCompletion.acreate(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        print("⚠️  Could not parse JSON for a row:", exc)
        return {"therapy_score": None, "relevancy_score": None, "evaluation": None}


async def _eval_all(pairs: List[tuple[str, str]], concurrency: int = 10) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def _worker(q, a):
        async with sem:
            return await _eval_pair(q, a)

    return await asyncio.gather(*[_worker(q, a) for q, a in pairs])

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_qa_evaluation(
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    *,
    question_col: str = "question",
    answer_col: str = "response",
    concurrency: int = 10,
) -> pd.DataFrame:
    """Append therapeutic & relevancy scores (0‑1) for every row.

    Parameters
    ----------
    input_path : str | Path
    output_path : str | Path | None
    question_col : str – column name containing the question
    answer_col   : str – column name containing the answer/response
    concurrency  : int – how many OpenAI calls in parallel
    """
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
    tqdm_bars = tqdm(total=len(pairs), desc="Rows processed")

    # Split into chunks so we can update progress bar when each chunk completes
    async def _run():
        results: List[Dict[str, Any]] = []
        CHUNK = concurrency * 2
        for i in range(0, len(pairs), CHUNK):
            chunk = pairs[i : i + CHUNK]
            chunk_results = await _eval_all(chunk, concurrency)
            results.extend(chunk_results)
            tqdm_bars.update(len(chunk))
        return results

    results = asyncio.run(_run())
    tqdm_bars.close()

    results_df = pd.DataFrame(results)
    df = pd.concat([df, results_df], axis=1)

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

# -----------------------------------------------------------------------------
# CLI helper
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Therapeutic Q&A evaluator")
    parser.add_argument("input", help="Path to input .xlsx or .csv")
    parser.add_argument("output", nargs="?", help="(Optional) save path for output file")
    parser.add_argument("--qcol", default="question", help="Column containing questions")
    parser.add_argument("--acol", default="response", help="Column containing answers")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel OpenAI calls")
    args = parser.parse_args()

    run_qa_evaluation(
        args.input,
        args.output,
        question_col=args.qcol,
        answer_col=args.acol,
        concurrency=args.concurrency,
    )
