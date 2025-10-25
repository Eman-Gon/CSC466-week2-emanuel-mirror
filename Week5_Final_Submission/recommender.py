import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("Running Improved KNN Recommender...")

# ----------------------------
# Load data
# ----------------------------
df_views = pd.read_parquet(P("content_views.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))

# Normalize content_id dtype across frames (prevents merge/isin mismatches)
cid_dtype = df_metadata["content_id"].dtype
df_views["content_id"] = df_views["content_id"].astype(cid_dtype, copy=False)

# ----------------------------
# De-dup + engagement filter
# ----------------------------
dup_count = df_views.duplicated(subset=["adventurer_id", "content_id"]).sum()
df_views = (
    df_views.sort_values("seconds_viewed", ascending=False)
    .drop_duplicates(subset=["adventurer_id", "content_id"], keep="first")
)
print(f"Removed {dup_count:,} duplicates")

cols = ["content_id", "minutes"]
if "publisher_id" in df_metadata.columns:
    cols.append("publisher_id")

df_merged = df_views.merge(df_metadata[cols], on="content_id", how="left")

denom = (df_merged["minutes"] * 60).replace(0, np.nan)
df_merged["watch_pct"] = (df_merged["seconds_viewed"] / denom).clip(lower=0)

low_engagement = ((df_merged["watch_pct"] < 0.05) & (df_merged["seconds_viewed"] < 30)).sum()
df_views_clean = df_merged[(df_merged["watch_pct"].fillna(0) >= 0.05) | (df_merged["seconds_viewed"] >= 30)].copy()
print(f"Removed {low_engagement:,} low-engagement views")

# ----------------------------
# Choose publisher with most subs
# ----------------------------
pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"Selected publisher {publisher_id} ({pub_counts.max():,} subs)")

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())

if "publisher_id" not in df_views_clean.columns:
    if "publisher_id" not in df_metadata.columns:
        raise KeyError("publisher_id not found in content_metadata")
    # If present (as we merged above), nothing to do

views_pub = df_views_clean[
    (df_views_clean.get("publisher_id") == publisher_id)
    & (df_views_clean["adventurer_id"].isin(sub_ids))
].copy()
print(f"Scoped views: {len(views_pub):,}")

# ----------------------------
# Build implicit feedback matrix
# ----------------------------
views_pub["value"] = 1
user_item = (
    views_pub.groupby(["adventurer_id", "content_id"])["value"]
    .max()
    .unstack(fill_value=0)
    .astype(np.float32)
)

if user_item.shape[1] == 0:
    raise ValueError("No items available after filtering")

# ----------------------------
# Item-based KNN (cosine)
# ----------------------------
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
    # Do not recommend items already seen
    scores[seen_idx] = -np.inf
    top_indices = np.argsort(-scores)[:n_recs]
    return [user_item.columns[i] for i in top_indices if np.isfinite(scores[i])]

# ----------------------------
# Generate recommendations
# ----------------------------
user_activity = user_item.sum(axis=1).sort_values(ascending=False)
recommendations_list = []

for uid in user_activity.index:
    recs = recommend_for_user(uid, n_recs=10)
    if len(recs) >= 10:
        recommendations_list.append({"adventurer_id": uid, "recommendations": recs[:10]})
    if len(recommendations_list) >= 10:
        break

output_file = P("recommendations.csv")
if recommendations_list:
    out_df = pd.DataFrame(
        [
            {"adventurer_id": r["adventurer_id"], **{f"rec{i+1}": r["recommendations"][i] for i in range(10)}}
            for r in recommendations_list
        ]
    )
    # Ensure content_id dtype consistency when writing/reading
    # We'll keep them as strings when saving, then cast back to original dtype for analysis
    for i in range(10):
        col = f"rec{i+1}"
        out_df[col] = out_df[col].astype(str)

    out_df.to_csv(output_file, index=False)
    print(f"Saved recommendations for {len(recommendations_list)} users")
else:
    print("No users with 10+ recommendations; skipping file save.")

# ----------------------------
# Coverage / genre / language mix
# ----------------------------
if recommendations_list:
    recs_df = pd.read_csv(output_file)

    # Collect unique recommended content_ids (cast back to original dtype)
    rec_set = set()
    for col in recs_df.columns:
        if col.startswith("rec"):
            rec_set.update(recs_df[col].dropna().tolist())

    # Cast to original dtype (handles numeric ids correctly)
    if pd.api.types.is_integer_dtype(cid_dtype):
        rec_ids = pd.Series(list(rec_set), dtype="Int64").dropna().astype(cid_dtype).tolist()
    elif pd.api.types.is_numeric_dtype(cid_dtype):
        rec_ids = pd.Series(list(rec_set), dtype="float64").dropna().astype(cid_dtype).tolist()
    else:
        rec_ids = pd.Series(list(rec_set), dtype="string").astype(cid_dtype).tolist()

    rec_content = df_metadata[df_metadata["content_id"].isin(rec_ids)]

    total_items = len(df_metadata)
    uniq_count = len(rec_ids)
    pct_total = (uniq_count / total_items * 100) if total_items else 0.0
    print(f"Unique content recommended: {uniq_count} ({pct_total:.1f}% of total catalog)")

    if not rec_content.empty and "genre_id" in rec_content.columns:
        rec_genres = rec_content["genre_id"].value_counts()
        overall_genres = df_metadata["genre_id"].value_counts(normalize=True)
        # Show up to 5 most common in recs
        for genre in rec_genres.head(5).index:
            rec_pct = rec_genres[genre] / len(rec_content) * 100
            overall_pct = overall_genres.get(genre, 0) * 100
            print(f"{genre}: {rec_pct:.1f}% vs {overall_pct:.1f}%")

    if not rec_content.empty and "language_code" in rec_content.columns:
        rec_langs = rec_content["language_code"].value_counts()
        overall_langs = df_metadata["language_code"].value_counts(normalize=True)
        for lang in rec_langs.index:
            rec_pct = rec_langs[lang] / len(rec_content) * 100
            overall_pct = overall_langs.get(lang, 0) * 100
            print(f"{lang}: {rec_pct:.1f}% vs {overall_pct:.1f}%")

    # ------------------------
    # User demographics summary
    # ------------------------
    rec_users = recs_df["adventurer_id"].unique()
    user_meta = pd.read_parquet(P("adventurer_metadata.parquet"))
    rec_user_info = user_meta[user_meta["adventurer_id"].isin(rec_users)]

    if not rec_user_info.empty:
        min_age = rec_user_info["age"].min()
        max_age = rec_user_info["age"].max()
        print(f"User age range: {int(min_age)}-{int(max_age)}")
        print(f"Primary languages: {rec_user_info['primary_language'].value_counts().to_dict()}")
        print(f"Top regions: {rec_user_info['region'].value_counts().head(3).to_dict()}")

print("Done.")
