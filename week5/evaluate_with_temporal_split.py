import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("EVALUATION WITH TEMPORAL SPLIT")
print("="*60)

# Load data
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

# Get publisher scope
pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"\nPublisher: {publisher_id}")

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())
views_pub = df_views[df_views["adventurer_id"].isin(sub_ids)]
publisher_content_scope = set(views_pub['content_id'].unique())
print(f"Content scope: {len(publisher_content_scope)} items")

# Calculate watch_pct
df_merged = views_pub.merge(df_metadata[['content_id', 'minutes']], on='content_id', how='left')
denom = (df_merged['minutes'] * 60).replace(0, np.nan)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / denom).clip(0, 1)

# Key insight: We need to split each user's views into train and test
# Let's use 80% of views for training, 20% for testing
WATCH_THRESHOLD = 0.5

def evaluate_method(rec_file, method_name):
    """Evaluate a recommendation method using temporal split"""
    recs = pd.read_csv(P(rec_file))
    
    hits = 0
    total_recs = 0
    total_possible = 0
    
    for _, row in recs.iterrows():
        user_id = row['adventurer_id']
        
        # Get user's viewing history in publisher scope
        user_views = df_merged[
            (df_merged['adventurer_id'] == user_id) &
            (df_merged['content_id'].isin(publisher_content_scope))
        ].copy()
        
        if len(user_views) == 0:
            continue
        
        # Sort by view_date to get temporal ordering
        user_views = user_views.sort_values('view_date')
        
        # Split: 80% train, 20% test
        split_idx = int(len(user_views) * 0.8)
        train_views = user_views.iloc[:split_idx]
        test_views = user_views.iloc[split_idx:]
        
        # "Liked" items are those watched >50% in the TEST set
        test_liked = set(test_views[test_views['watch_pct'] >= WATCH_THRESHOLD]['content_id'].unique())
        
        # Count recommendations
        recs_list = [row['rec1'], row['rec2']]
        total_recs += len(recs_list)
        total_possible += len(test_liked)
        
        # Check hits
        for rec in recs_list:
            if rec in test_liked:
                hits += 1
    
    precision = hits / total_recs if total_recs > 0 else 0
    recall = hits / total_possible if total_possible > 0 else 0
    
    return precision, recall, total_recs, total_possible

# Evaluate all methods
methods = [
    ('collaborative_eval.csv', 'Collaborative Filtering'),
    ('content_based_eval.csv', 'Content-Based'),
    ('heuristic_eval.csv', 'Global Heuristic'),
    ('post_eval.csv', 'Hybrid (60/40)'),
    ('pre_eval.csv', 'Baseline KNN')
]

print("\n" + "="*60)
print("RESULTS (with temporal train/test split)")
print("="*60)
print(f"{'Method':<30} {'Precision@2':<15} {'Recall@2':<15}")
print("-" * 60)

results = []
for rec_file, method_name in methods:
    try:
        precision, recall, total_recs, total_possible = evaluate_method(rec_file, method_name)
        results.append((method_name, precision, recall))
        print(f"{method_name:<30} {precision:<15.3f} {recall:<15.3f}")
    except Exception as e:
        print(f"{method_name:<30} Error: {e}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
print("This evaluation tests if recommendations match what users")
print("watched AFTER their training period (future likes).")
print("This is a more realistic evaluation than testing on all history.")
print("="*60)