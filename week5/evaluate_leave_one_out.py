import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("LEAVE-ONE-OUT EVALUATION")
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

WATCH_THRESHOLD = 0.5

def evaluate_method(rec_file, method_name):
    """
    For each user, we check:
    1. What items did they watch with >50% completion?
    2. Of those, how many appear in the recommendations?
    
    This is "recall" - did we recommend items they liked?
    """
    recs = pd.read_csv(P(rec_file))
    
    hits = 0
    total_recs = 0
    users_evaluated = 0
    
    hit_details = []
    
    for _, row in recs.iterrows():
        user_id = row['adventurer_id']
        
        # Get all items user liked (in publisher scope)
        user_views = df_merged[
            (df_merged['adventurer_id'] == user_id) &
            (df_merged['content_id'].isin(publisher_content_scope))
        ]
        
        liked_items = set(user_views[user_views['watch_pct'] >= WATCH_THRESHOLD]['content_id'].unique())
        
        if len(liked_items) == 0:
            continue
        
        users_evaluated += 1
        
        # Get recommendations
        recs_list = [row['rec1'], row['rec2']]
        total_recs += len(recs_list)
        
        # Check hits
        user_hits = 0
        for rec in recs_list:
            if rec in liked_items:
                hits += 1
                user_hits += 1
        
        hit_details.append({
            'user': user_id,
            'liked_count': len(liked_items),
            'hits': user_hits,
            'recs': recs_list
        })
    
    precision = hits / total_recs if total_recs > 0 else 0
    
    return precision, users_evaluated, hit_details

# Evaluate all methods
methods = [
    ('collaborative_eval.csv', 'Collaborative Filtering'),
    ('content_based_eval.csv', 'Content-Based'),
    ('heuristic_eval.csv', 'Global Heuristic'),
    ('post_eval.csv', 'Hybrid (60/40)'),
    ('pre_eval.csv', 'Baseline KNN')
]

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"{'Method':<30} {'Precision@2':<15} {'Users':<10}")
print("-" * 60)

all_details = {}
for rec_file, method_name in methods:
    try:
        precision, users, details = evaluate_method(rec_file, method_name)
        all_details[method_name] = details
        print(f"{method_name:<30} {precision:<15.3f} {users:<10}")
    except Exception as e:
        print(f"{method_name:<30} Error: {e}")

print("\n" + "="*60)
print("DETAILED ANALYSIS - First 3 users")
print("="*60)

for method_name, details in all_details.items():
    print(f"\n{method_name}:")
    for i, d in enumerate(details[:3]):
        print(f"  User {d['user']}: {d['hits']}/2 hits (liked {d['liked_count']} items)")
        print(f"    Recommended: {d['recs']}")

print("\n" + "="*60)
print("KEY INSIGHT")
print("="*60)
print("The problem: Your models recommend NEW items (good for discovery!)")
print("but evaluation only counts items users ALREADY watched.")
print("")
print("Solution: Either:")
print("  1. Accept lower precision (you're discovering new content)")
print("  2. Add 'similar items' penalty (reward recommending similar to liked)")
print("  3. Use a proper train/test split")
print("="*60)