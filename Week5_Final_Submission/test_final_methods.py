import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("TESTING ALL METHODS ON FINAL 20 USERS")
print("="*60)

# Load data
df_views = pd.read_parquet(P("content_views.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))

# Get publisher wn32 subscribers
pub_subs = df_subs[df_subs['publisher_id'] == 'wn32']

# Calculate user activity
user_activity = df_views[
    df_views['adventurer_id'].isin(pub_subs['adventurer_id'])
].groupby('adventurer_id')['content_id'].count().sort_values(ascending=False)

# Select same 20 users as your final submission
top_users = user_activity.head(15).index.tolist()
medium_users = user_activity.iloc[20:25].index.tolist()
selected_users = top_users + medium_users

print(f"\nSelected {len(selected_users)} users")
print(f"Activity range: {user_activity[selected_users].min():.0f} to {user_activity[selected_users].max():.0f} views\n")

# Test each method
from recommender import recommend_for_user
from heuristic_recommender import recommend_trending
from advanced_recommender_week5 import recommend_hybrid

methods = {
    'Collaborative': recommend_for_user,
    'Heuristic': recommend_trending,
    'Hybrid': recommend_hybrid
}

results = {}

for method_name, recommend_func in methods.items():
    print(f"Testing {method_name}...")
    success = 0
    recs_list = []
    
    for user_id in selected_users:
        try:
            recs = recommend_func(user_id, n_recs=2)
            if len(recs) >= 2:
                success += 1
                recs_list.extend(recs[:2])
        except Exception as e:
            print(f"  Error for {user_id}: {e}")
    
    unique_items = len(set(recs_list))
    coverage = unique_items / 38 * 100  # 38 items in publisher wn32
    
    results[method_name] = {
        'success_rate': success / len(selected_users),
        'unique_items': unique_items,
        'coverage': coverage
    }
    
    print(f"  Success: {success}/{len(selected_users)} users")
    print(f"  Unique items: {unique_items} ({coverage:.1f}% coverage)")
    print()

# Recommendation
print("="*60)
print("RECOMMENDATION")
print("="*60)

best_method = max(results.items(), key=lambda x: (x[1]['success_rate'], x[1]['unique_items']))

print(f"\nBEST METHOD: {best_method[0]}")
print(f"   - Success rate: {best_method[1]['success_rate']:.1%}")
print(f"   - Diversity: {best_method[1]['unique_items']} unique items")
print(f"   - Coverage: {best_method[1]['coverage']:.1f}%")

if best_method[0] == 'Collaborative':
    print("\nUse pure COLLABORATIVE filtering")
    print("   Your 20 users are all highly active (good for collaborative)")
    
elif best_method[0] == 'Heuristic':
    print("\nHeuristic won, but only recommends 2 items to everyone")
    print("   Consider using COLLABORATIVE for personalization")
    
else:
    print("\nUse HYBRID (60% collaborative + 40% content)")
    print("   Best balance of accuracy and diversity")
