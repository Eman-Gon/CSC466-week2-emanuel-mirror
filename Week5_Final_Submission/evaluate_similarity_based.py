import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("Running similarity-based evaluation")

df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())
views_pub = df_views[df_views["adventurer_id"].isin(sub_ids)]
publisher_content_scope = set(views_pub['content_id'].unique())

df_merged = views_pub.merge(df_metadata[['content_id', 'minutes']], on='content_id', how='left')
denom = (df_merged['minutes'] * 60).replace(0, np.nan)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / denom).clip(0, 1)

print("Building item similarity matrix")
user_item = views_pub.groupby(['adventurer_id', 'content_id']).size().unstack(fill_value=0)
item_sim = cosine_similarity(user_item.T)
item_sim_df = pd.DataFrame(item_sim, index=user_item.columns, columns=user_item.columns)
print(f"Similarity matrix shape: {item_sim_df.shape}")

WATCH_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.3

def evaluate_with_similarity(rec_file, method_name):
    recs = pd.read_csv(P(rec_file))
    
    exact_hits = 0
    similarity_score = 0
    total_recs = 0
    users_evaluated = 0
    
    for _, row in recs.iterrows():
        user_id = row['adventurer_id']
        user_views = df_merged[
            (df_merged['adventurer_id'] == user_id) &
            (df_merged['content_id'].isin(publisher_content_scope))
        ]
        liked_items = set(user_views[user_views['watch_pct'] >= WATCH_THRESHOLD]['content_id'].unique())
        if len(liked_items) == 0:
            continue
        users_evaluated += 1
        recs_list = [row['rec1'], row['rec2']]
        total_recs += len(recs_list)
        for rec in recs_list:
            if rec in liked_items:
                exact_hits += 1
                similarity_score += 1.0
            elif rec in item_sim_df.index:
                max_sim = 0
                for liked_item in liked_items:
                    if liked_item in item_sim_df.columns:
                        sim = item_sim_df.loc[rec, liked_item]
                        max_sim = max(max_sim, sim)
                if max_sim >= SIMILARITY_THRESHOLD:
                    similarity_score += max_sim
    
    exact_precision = exact_hits / total_recs if total_recs > 0 else 0
    sim_precision = similarity_score / total_recs if total_recs > 0 else 0
    return exact_precision, sim_precision, users_evaluated

methods = [
    ('collaborative_eval.csv', 'Collaborative Filtering'),
    ('content_based_eval.csv', 'Content-Based'),
    ('heuristic_eval.csv', 'Global Heuristic'),
    ('post_eval.csv', 'Hybrid (60/40)'),
    ('pre_eval.csv', 'Baseline KNN')
]

print("\nResults:")
print(f"{'Method':<30} {'Exact P@2':<15} {'Similar P@2':<15} {'Users':<10}")
print("-" * 70)

results = []
for rec_file, method_name in methods:
    try:
        exact_p, sim_p, users = evaluate_with_similarity(rec_file, method_name)
        results.append((method_name, exact_p, sim_p))
        print(f"{method_name:<30} {exact_p:<15.3f} {sim_p:<15.3f} {users:<10}")
    except Exception as e:
        print(f"{method_name:<30} Error: {e}")

results.sort(key=lambda x: x[2], reverse=True)

print("\nRanked by Similarity Score:")
for method, exact_p, sim_p in results:
    improvement = (sim_p - exact_p) * 100
    print(f"{method:<30} {sim_p:.3f} (+{improvement:.1f}%)")

print("\nInterpretation:")
print("Exact P@2 counts only exact matches to viewed items.")
print("Similar P@2 also credits items similar to liked content.")
print("High Similar P@2 indicates better discovery performance.")
