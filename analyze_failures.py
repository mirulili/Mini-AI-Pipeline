import pandas as pd
from src.utils import map_label
import re

pd.set_option('display.max_colwidth', None)

try:
    df = pd.read_csv("outputs/dev_eval.csv")
    print(f"Total rows: {len(df)}")
    
    failures = []
    
    for idx, row in df.iterrows():
        gold = row['answer']
        try:
            gold_letter = map_label(gold)
        except:
            gold_letter = str(gold)
            
        pred_full = str(row['pred'])
        
        match = re.search(r"정답\s*[:]\s*([ABCD])", pred_full, re.IGNORECASE)
        if match:
            pred_letter = match.group(1).upper()
        else:
            lines = pred_full.splitlines()
            if lines and re.search(r"[ABCD]", lines[-1], re.IGNORECASE):
                pred_letter = re.search(r"[ABCD]", lines[-1], re.IGNORECASE).group(0).upper()
            else:
                pred_letter = "NONE"

        if pred_letter != gold_letter:
            failures.append({
                "idx": idx,
                "question": row['question'],
                "gold": gold_letter,
                "pred": pred_letter,
                "reasoning": pred_full
            })

    print(f"Found {len(failures)} failures.")
    
    # Print first 3 failures
    for f in failures[:3]:
        print(f"\n--- Failure {f['idx']} ---")
        print(f"Q: {f['question']}")
        print(f"Gold: {f['gold']} | Pred: {f['pred']}")
        print(f"Reasoning:\n{f['reasoning']}")

except Exception as e:
    print(e)
