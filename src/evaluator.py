import argparse
from typing import cast

import pandas as pd
from pandas.core.series import Series
from tqdm import tqdm

from . import config
from .rag_pipeline import answer_question
from .utils import map_label


def evaluate(subset: int = -1) -> None:
    """Evaluate on dev dataset"""
    dev_path = config.DATA_DIR / "Criminal-Law-test.csv"
    if not dev_path.exists():
        raise FileNotFoundError("Place dev.csv under data/")
    dev: pd.DataFrame = pd.read_csv(dev_path)
    if subset > 0:
        dev = dev.head(subset)

    preds: list[str] = []
    correct = 0
    total = 0

    import concurrent.futures

    # Helper function for parallel processing
    def process_row(row_tuple):
        idx, row = row_tuple
        row_series: Series = cast(Series, row)
        q_text = (
            str(row_series.get("question", "") or "") 
            + "\n"
            + f"A) {str(row_series.get('A', '') or '')}"
            + "\n"
            + f"B) {str(row_series.get('B', '') or '')}"
            + "\n"
            + f"C) {str(row_series.get('C', '') or '')}"
            + "\n"
            + f"D) {str(row_series.get('D', '') or '')}"
        )
        gold = map_label(str(row_series.get("answer", "")))
        
        raw_pred = answer_question(q_text)
        
        # Extract only the answer character (A, B, C, D) for evaluation
        try:
            from .generator import parse_answer
            pred_letter = parse_answer(raw_pred)
        except Exception as e:
            print(f"Parse error: {e}")
            pred_letter = "Unknown"
            
        return idx, raw_pred, pred_letter, gold

    # Parallel execution (max_workers adjustable)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # returns (raw_pred, pred_letter, gold)
        future_to_row = {executor.submit(process_row, row): row for row in dev.iterrows()}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(dev)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing row: {e}")
                # Tuple must match loop return size: (idx, raw_pred, pred_letter, gold)
                # However, we can't easily get idx here if we don't pass it.
                # Only way is to fetch from future_to_row, but simpler to just handle it inside process_row
                # or let it sort at the end. 
                # Actually, let's look at how to get idx safely. 
                # We can retrieve 'row' from future_to_row[future] which is (idx, series).
                idx_err = future_to_row[future][0]
                results.append((idx_err, "", "Error", ""))

    # Aggregate results (Sort by index to match original dataframe)
    results.sort(key=lambda x: x[0])
    
    preds = [r[1] for r in results]
    # r[2] is pred_letter, r[3] is gold
    correct = sum(1 for r in results if r[2] == r[3])
    total = len(results)

    acc = correct / max(1, total)
    print(f"dev accuracy: {acc:.4f} ({correct}/{total})")

    out: pd.DataFrame = dev.copy()
    out["pred"] = preds
    out.to_csv(config.OUT_DIR / "dev_eval.csv", index=False)
    print(f"Saved results to {config.OUT_DIR / 'dev_eval.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=-1, help="Number of samples to evaluate (-1 = all)")
    args = parser.parse_args()
    evaluate(subset=args.subset)
