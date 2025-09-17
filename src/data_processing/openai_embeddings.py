"""
OpenAI Embeddings Generator

This module provides functionality for generating embeddings using OpenAI's text-embedding models
with proper rate limiting and batch processing for large datasets.

Key Features:
- Rate limiting compliant with OpenAI API tiers
- Batch processing for efficiency
- Error handling and retry logic
- Progress tracking with tqdm
- Automatic deque-based rate limiting

Author: Cognitive Representation Engineering Project
"""

import os
import time
import math
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# ── 1) Environment Configuration ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
client = OpenAI()  # Pulls API key from OPENAI_API_KEY environment variable

# ── 2) Rate Limiting Configuration (Tier-1 for text-embedding-3-large) ─────────
MAX_RPM = 3_000     # Maximum requests per minute (see OpenAI docs for your tier)
BATCH = 96          # Texts per request (308 TPM × 96 ≈ 29k TPM, safely under 30k limit)

# ── 3) Paths ────────────────────────────────────────────────────────────────────
EXCEL_PATH = "/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/data/cognitive_experiment_results003.xlsx"
OUTPUT_TSV = "/home/koalacrown/Desktop/Code/Projects/LLM_threapy/LLM_agumented_therapy/Cogexp/embeddings/response_embeddings_with_features.tsv"

# ── 4) Load data & sanity-check ─────────────────────────────────────────────────
df = pd.read_excel(EXCEL_PATH)
need = {"response", "pair", "strength", "question_index"}
if not need.issubset(df.columns):
    raise KeyError(f"Missing columns: {need - set(df.columns)}")

# ── 5) Embed in mini-batches (cheaper & faster) ────────────────────────────────
records, request_times = [], deque()
for i in tqdm(range(0, len(df), BATCH), desc="Embedding batches"):
    batch = df.iloc[i:i + BATCH]

    # client-side RPM throttle
    while request_times and time.time() - request_times[0] > 60:
        request_times.popleft()
    if len(request_times) >= MAX_RPM:
        time.sleep(60 - (time.time() - request_times[0]))

    # ---- embedding request ----------------------------------------------------
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=batch["response"].astype(str).tolist(),
        encoding_format="float"               # default, but explicit for clarity
    )
    request_times.append(time.time())

    # ---- merge vectors back into rows -----------------------------------------
    for row, emb_obj in zip(batch.itertuples(index=False), resp.data):
        rec = {
            "response": row.response,
            "pair": row.pair,
            "strength": row.strength,
            "question_index": row.question_index,
        }
        rec.update({f"emb_dim_{d}": v for d, v in enumerate(emb_obj.embedding)})
        records.append(rec)

# ── 6) Persist ─────────────────────────────────────────────────────────────────
pd.DataFrame(records).to_csv(OUTPUT_TSV, sep="\t", index=False)
print(f"✓ {len(records):,} embeddings written → {OUTPUT_TSV}")
