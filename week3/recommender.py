import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("Running Improved KNN Recommender...")

df_views = pd.read_parquet(P("content_views.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))

before = len(df_views)
dup_count = df_views.duplicated(subset=['adventurer_id', 'content_id']).sum()
df_views = df_views.sort_values('seconds_viewed', ascending=False).drop_duplicates(subset=['adventurer_id', 'content_id'], keep='first')
print(f"Removed {dup_count:,} duplicates")

cols = ['content_id', 'minutes']
if 'publisher_id' in df_metadata.columns:
    cols.append('publisher_id')
df_merged = df_views.merge(df_metadata[cols], on='content_id', how='left')
denom = (df_merged['minutes'] * 60).replace(0, np.nan)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / denom).clip(lower=0)

before_filter = len(df_merged)
low_engagement = ((df_merged['watch_pct'] < 0.05) & (df_merged['seconds_viewed'] < 30)).sum()
df_views_clean = df_merged[(df_merged['watch_pct'].fillna(0) >= 0.05) | (df_merged['seconds_viewed'] >= 30)].copy()
print(f"Removed {low_engagement:,} low-engagement views")

pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"Selected publisher {publisher_id} ({pub_counts.max():,} subs)")

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())

if 'publisher_id' not in df_views_clean.columns:
    if 'publisher_id' in df_metadata.columns:
        df_views_clean = df_views_clean
    else:
        raise KeyError("publisher_id not found in content_metadata")

views_pub = df_views_clean[(df_views_clean.get("publisher_id") == publisher_id) & (df_views_clean["adventurer_id"].isin(sub_ids))].copy()
print(f"Scoped views: {len(views_pub):,}")

views_pub["value"] = 1
user_item = (
    views_pub.groupby(["adventurer_id", "content_id"])["value"]
    .max()
    .unstack(fill_value=0)
    .astype(np.float32)
)

if user_item.shape[1] == 0:
    raise ValueError("No items available after filtering")

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
    return [user_item.columns[i] for i in top_indices if np.isfinite(scores[i])]

user_activity = user_item.sum(axis=1).sort_values(ascending=False)
recommendations_list = []
for uid in user_activity.index:
    recs = recommend_for_user(uid, n_recs=10)
    if len(recs) >= 10:
        recommendations_list.append({'adventurer_id': uid, 'recommendations': recs[:10]})
    if len(recommendations_list) >= 10:
        break

output_file = P("recommendations.csv")
out_df = pd.DataFrame(
    [
        {"adventurer_id": r["adventurer_id"], **{f"rec{i+1}": str(cid) for i, cid in enumerate(r["recommendations"])}}
        for r in recommendations_list
    ]
)
out_df.to_csv(output_file, index=False)
print(f"Saved recommendations for {len(recommendations_list)} users")

recs_df = pd.read_csv(output_file)
all_recs = []
for col in recs_df.columns:
    if col.startswith('rec'):
        all_recs.extend(recs_df[col].dropna().unique())

rec_content = df_metadata[df_metadata['content_id'].isin(all_recs)]
print(f"Unique content recommended: {len(all_recs)} ({len(all_recs)/len(df_metadata)*100:.1f}% of total)")

if 'genre_id' in rec_content.columns:
    rec_genres = rec_content['genre_id'].value_counts()
    overall_genres = df_metadata['genre_id'].value_counts(normalize=True)
    for genre in rec_genres.head().index:
        rec_pct = rec_genres[genre] / len(rec_content) * 100
        overall_pct = overall_genres.get(genre, 0) * 100
        print(f"{genre}: {rec_pct:.1f}% vs {overall_pct:.1f}%")

if 'language_code' in rec_content.columns:
    rec_langs = rec_content['language_code'].value_counts()
    overall_langs = df_metadata['language_code'].value_counts(normalize=True)
    for lang in rec_langs.index:
        rec_pct = rec_langs[lang] / len(rec_content) * 100
        overall_pct = overall_langs.get(lang, 0) * 100
        print(f"{lang}: {rec_pct:.1f}% vs {overall_pct:.1f}%")

rec_users = recs_df['adventurer_id'].unique()
user_meta = pd.read_parquet(P("adventurer_metadata.parquet"))
rec_user_info = user_meta[user_meta['adventurer_id'].isin(rec_users)]

print(f"User age range: {rec_user_info['age'].min():.0f}-{rec_user_info['age'].max():.0f}")
print(f"Primary languages: {rec_user_info['primary_language'].value_counts().to_dict()}")
print(f"Top regions: {rec_user_info['region'].value_counts().head(3).to_dict()}")
print("Done.")
