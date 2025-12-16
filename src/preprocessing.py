# src/preprocessing.py
from __future__ import annotations

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import config, utils

def build_index() -> None:
    """
    Load Train dataset and build knowledge base (Index) in the form of 'Question+Options+Answer'.
    """
    # 1. Find Train dataset
    train_path = config.DATA_DIR / "Criminal-Law-train.csv"
    
    if not train_path.exists():
        # Fallback path in case filename is different
        train_path = config.DATA_DIR / "train.csv"

    if not train_path.exists():
        print(f"Error: Training data file not found. (Check {config.DATA_DIR})")
        return

    print(f"Loading knowledge base from {train_path}...")
    df = pd.read_csv(train_path)

    # 2. Text construction: Question + Options + Answer (for model reference)
    texts: list[str] = []
    
    for _, row in df.iterrows():
        # Safely extract data
        q = str(row.get("question", "")).strip()
        opt_a = str(row.get("A", "")).strip()
        opt_b = str(row.get("B", "")).strip()
        opt_c = str(row.get("C", "")).strip()
        opt_d = str(row.get("D", "")).strip()
        
        # Convert answer label (1 -> A)
        raw_ans = str(row.get("answer", "")).strip()
        ans_label = utils.map_label(raw_ans)

        full_text = (
            f"질문: {q}\n"
            f"선택지:\n"
            f"A) {opt_a}\n"
            f"B) {opt_b}\n"
            f"C) {opt_c}\n"
            f"D) {opt_d}\n"
            # f"정답: {ans_label}" # Remove answer label to prevent bias
        )
        texts.append(full_text)

    if not texts:
        print("Error: Data is empty.")
        return

    # 3. Generate embeddings and save index
    print(f"Embedding {len(texts)} documents... (Wait a moment)")
    embeddings = utils.openai_embed(texts)

    print("Building Index...")
    nn = NearestNeighbors(n_neighbors=min(10, len(texts)), metric="cosine")
    nn.fit(embeddings)

    # Save (Create outputs folder)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save raw text
    config.INDEX_TXT.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    
    # Save embedding vectors
    np.save(config.OUT_DIR / "embeddings.npy", embeddings)

    # Save search model
    joblib.dump(nn, config.INDEX_NN)
    
    print(f"Index successfully built & saved to {config.OUT_DIR}")

if __name__ == "__main__":
    build_index()