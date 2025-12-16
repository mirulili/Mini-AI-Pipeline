import argparse
import random
import pandas as pd
from tqdm import tqdm

# Import central config and utilities
from . import config
from .utils import map_label

def run_baseline(mode: str = "random"):
    """
    mode='random': Randomly choose from A, B, C, D (approx. 25%)
    mode='majority': Always predict the most frequent label from training data
    """
    print(f"--- Running Naive Baseline (Mode: {mode}) ---")
    
    # Load data
    dev_path = config.DATA_DIR / "Criminal-Law-test.csv"
    train_path = config.DATA_DIR / "Criminal-Law-train.csv"
    
    if not dev_path.exists():
        raise FileNotFoundError("data/dev.csv is required.")
        
    dev_df = pd.read_csv(dev_path)
    
    # Training data statistics for Majority vote (if needed)
    majority_label = "A"
    if mode == "majority":
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            # Check label distribution
            train_df["norm_answer"] = train_df["answer"].apply(map_label)
            majority_label = train_df["norm_answer"].mode()[0]
            print(f"Train data majority label: {majority_label}")
        else:
            print("Warning: train.csv not found for majority voting. Defaulting to 'A'.")

    preds = []
    correct = 0
    total = 0
    
    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df)):
        gold = map_label(row["answer"])
        
        if mode == "random":
            pred = random.choice(["A", "B", "C", "D"])
        elif mode == "majority":
            pred = majority_label
        else:
            pred = "A" # Default fallback
            
        preds.append(pred)
        
        if pred == gold:
            correct += 1
        total += 1

    acc = correct / max(1, total)
    print(f"[{mode}] Baseline Accuracy: {acc:.4f} ({correct}/{total})")
    
    # Save results
    out_df = dev_df.copy()
    out_df["pred"] = preds
    save_path = config.OUT_DIR / f"baseline_{mode}_eval.csv"
    out_df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="random", choices=["random", "majority"], help="Baseline mode")
    args = parser.parse_args()
    run_baseline(mode=args.mode)