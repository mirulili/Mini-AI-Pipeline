import pandas as pd
from src.utils import map_label
import sys

try:
    df = pd.read_csv("outputs/dev_eval.csv")
    print(f"Total rows: {len(df)}")
    for idx, row in df.iterrows():
        q_start = row['question'][:20]
        gold = row['answer']
        try:
            gold_letter = map_label(gold)
        except:
            gold_letter = str(gold)
            
        pred_full = str(row['pred'])
        # Try to parse pred letter again to be sure
        import re
        match = re.search(r"정답\s*[:]\s*([ABCD])", pred_full, re.IGNORECASE)
        if match:
            pred_letter = match.group(1).upper()
        else:
            # Fallback
            lines = pred_full.splitlines()
            if lines and re.search(r"[ABCD]", lines[-1], re.IGNORECASE):
                pred_letter = re.search(r"[ABCD]", lines[-1], re.IGNORECASE).group(0).upper()
            else:
                pred_letter = "NONE"

        print(f"{idx} | Q: {q_start}... | Gold: {gold_letter} | Pred: {pred_letter}")
        
except Exception as e:
    print(e)
