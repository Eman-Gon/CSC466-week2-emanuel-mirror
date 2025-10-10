import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("WEEK 3: IMPROVED KNN RECOMMENDER")
print("Based on Emanuel's Week 2 Item-Item Collaborative Filtering")

df_views = pd.read_parquet(P("content_views.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))

print(f"Initial views: {len(df_views):,}")

before = len(df_views)
dup_count = df_views.duplicated(subset=['adventurer_id', 'content_id']).sum()
df_views = df_views.sort_values('seconds_viewed', ascending=False).drop_duplicates(subset=['adventurer_id', 'content_id'], keep='first')
print(f"Removed duplicates: {dup_count:,}. After: {len(df_views):,} views")

cols = ['content_id', 'minutes']
if 'publisher_id' in df_metadata.columns:
    cols.append('publisher_id')
df_merged = df_views.merge(df_metadata[cols], on='content_id', how='left')
denom = (df_merged['minutes'] * 60).replace(0, np.nan)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / denom).clip(lower=0)

before_filter = len(df_merged)
low_engagement = ((df_merged['watch_pct'] < 0.05) & (df_merged['seconds_viewed'] < 30)).sum()
df_views_clean = df_merged[(df_merged['watch_pct'].fillna(0) >= 0.05) | (df_merged['seconds_viewed'] >= 30)].copy()
print(f"Low engagement removed: {low_engagement:,} of {before_filter:,}. After filtering: {len(df_views_clean):,} views")

print("Selecting top publisher...")
pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"Publisher: {publisher_id} with {pub_counts.max():,} subscribers")

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())

if 'publisher_id' not in df_views_clean.columns:
    if 'publisher_id' in df_metadata.columns:
        df_views_clean = df_views_clean
    else:
        raise KeyError("publisher_id not found in content_metadata; cannot scope views to a publisher.")

views_pub = df_views_clean[(df_views_clean.get("publisher_id") == publisher_id) & (df_views_clean["adventurer_id"].isin(sub_ids))].copy()
print(f"Views from subscribers of target publisher: {len(views_pub):,}")

print("Building item-item KNN model...")
views_pub["value"] = 1
user_item = (
    views_pub.groupby(["adventurer_id", "content_id"])["value"]
    .max()
    .unstack(fill_value=0)
    .astype(np.float32)
)

if user_item.shape[1] == 0:
    raise ValueError("No items available after filtering to build the KNN model.")

item_user = user_item.T
n_neighbors = max(1, min(20, len(item_user)))
knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
knn.fit(item_user.values)
item_matrix = item_user.values

def recommend_for_user(uid, n_recs=10):
    if uid not in user_item.index:
        return []
    seen_idx = np.where(user_item.loc[uid].values > 0)[0]
    if len(seen_idx) == 0:
        return []
    dists, idxs = knn.kneighbors(item_matrix[seen_idx], return_distance=True)
    scores = np.zeros(user_item.shape[1], dtype=np.float32)
    for drow, irow in zip(dists, idxs):
        sims = 1.0 - drow
        np.add.at(scores, irow, sims)
    scores[seen_idx] = -np.inf
    top_indices = np.argsort(-scores)[:n_recs]
    recommendations = [user_item.columns[i] for i in top_indices if np.isfinite(scores[i])]
    return recommendations

print("Generating recommendations for 10 diverse users...")
user_activity = user_item.sum(axis=1).sort_values(ascending=False)
recommendations_list = []
for uid in user_activity.index:
    recs = recommend_for_user(uid, n_recs=10)
    if len(recs) >= 10:
        recommendations_list.append({'adventurer_id': uid, 'recommendations': recs[:10]})
        print(f"{uid}: {len(recs)} recommendations")
    if len(recommendations_list) >= 10:
        break

print("Saving recommendations...")
output_file = P("recommendations.csv")
out_df = pd.DataFrame(
    [
        {"adventurer_id": r["adventurer_id"], **{f"rec{i+1}": str(cid) for i, cid in enumerate(r["recommendations"])}}
        for r in recommendations_list
    ]
)
out_df.to_csv(output_file, index=False)
print(f"Saved {len(recommendations_list)} users with recommendations to {output_file}")

print("IMPROVEMENTS SUMMARY")
print(
    "\n".join(
        [
            "CHANGES FROM WEEK 2:",
            "",
            f"1. DUPLICATE REMOVAL: Removed {dup_count:,} duplicate (adventurer_id, content_id) pairs; kept highest seconds_viewed.",
            f"2. LOW-ENGAGEMENT FILTERING: Removed {low_engagement:,} views with <5% watch AND <30 seconds.",
            "3. SAME CORE ALGORITHM: Item-item CF with cosine similarity; recommends items similar to watched ones.",
            "",
            "EXPECTED RESULTS:",
            "- Better precision via noise reduction",
            "- Cleaner signals (no double-counted duplicates)",
            "- Directly comparable to Week 2"
        ]
    )
)
print("COMPLETE - Ready for evaluation!")
