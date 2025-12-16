from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Index file paths
INDEX_EMB = OUT_DIR / "embeddings.npy"
INDEX_TXT = OUT_DIR / "texts.json"
INDEX_NN = OUT_DIR / "nn.joblib"

# Model configuration
MODEL_EMBEDDING = "text-embedding-3-small"
MODEL_GENERATION = "gpt-4o-mini"

# RAG configuration
TOP_K = 5
MAX_CONTEXT_LEN = 15000


# Get API key
def get_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please check your .env file or shell environment variables."
        )
    return key
