import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("MODEL EVALUATION COMPARISON")

pre_df = pd.read_csv(P('pre_eval.csv'))
post_df = pd.read_csv(P('post_eval.csv'))

print(f"\nPre-eval users: {len(pre_df)}")
print(f"Post-eval users: {len(post_df)}")

def calc_diversity(df):
    all_recs = set()
    for col in ['rec1', 'rec2']:
        if col in df.columns:
            all_recs.update(df[col].dropna().unique())
    return len(all_recs)

pre_diversity = calc_diversity(pre_df)
post_diversity = calc_diversity(post_df)

print(f"\n--- Diversity ---")
print(f"Pre-eval unique items: {pre_diversity}")
print(f"Post-eval unique items: {post_diversity}")
print(f"Change: {post_diversity - pre_diversity:+d}")

pre_set = set(zip(pre_df['adventurer_id'], pre_df['rec1'])) | \
          set(zip(pre_df['adventurer_id'], pre_df['rec2']))
post_set = set(zip(post_df['adventurer_id'], post_df['rec1'])) | \
           set(zip(post_df['adventurer_id'], post_df['rec2']))

overlap = len(pre_set & post_set)
print(f"\n--- Overlap ---")
print(f"Identical recommendations: {overlap}/{len(pre_set)} ({overlap/len(pre_set)*100:.1f}%)")