import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

ROOT = Path(__file__).resolve().parent 
P = lambda name: ROOT / name

df_views = pd.read_parquet(P("content_views.parquet"))
df_subs  = pd.read_parquet(P("subscriptions.parquet"))

pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"Publisher chosen: {publisher_id}")

# Restrict to that publisher and its subscribers
subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())

views_pub = df_views[
    (df_views["publisher_id"] == publisher_id)
    & (df_views["adventurer_id"].isin(sub_ids))
].copy()

# 80/20 split per user
def create_train_test_split(df: pd.DataFrame):
    """Split each user's rows into 80% train / 20% test (by content_id order)."""
    train_list, test_list = [], []
    for uid in df["adventurer_id"].unique():
        user_data = df[df["adventurer_id"] == uid].sort_values("content_id")
        if len(user_data) >= 5:
            n_train = max(1, int(len(user_data) * 0.8))
            train_list.append(user_data.iloc[:n_train])
            test_list.append(user_data.iloc[n_train:])
        else:
            #train listed
            train_list.append(user_data)
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=df.columns)
    return train_df, test_df

# Split data
train_views, test_views = create_train_test_split(views_pub)
print(f"Train: {len(train_views)} rows, Test: {len(test_views)} rows")

# matrix on train data
train_views = train_views.copy()
train_views["value"] = 1
user_item = (
    train_views.groupby(["adventurer_id", "content_id"])["value"]
    .max()
    .unstack(fill_value=0)
    .astype(np.float32)
)

print(f"User-item matrix: {user_item.shape}")

item_user = user_item.T
n_neighbors = min(20, len(item_user))
knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
knn.fit(item_user.values)

item_matrix = item_user.values 

def recommend2(uid: str) -> list[str]:
    """Recommend 2 unseen items for a user using item-item cosine KNN."""
    if uid not in user_item.index:
        return []
    seen_idx = np.where(user_item.loc[uid].values > 0)[0]
    if len(seen_idx) == 0:
        return []
    # Get neighbors for each seen item
    dists, idxs = knn.kneighbors(item_matrix[seen_idx], return_distance=True)
    scores = np.zeros(user_item.shape[1], dtype=np.float32) 
    for drow, irow in zip(dists, idxs):
        sims = 1.0 - drow
        np.add.at(scores, irow, sims)
    # Don't recommend items already seen
    scores[seen_idx] = -np.inf
    # Top-2 indices
    top = np.argsort(-scores)[:2]
    return [user_item.columns[i] for i in top if np.isfinite(scores[i])]

user_activity = user_item.sum(axis=1).sort_values(ascending=False).index.tolist()

rows = []
for uid in user_activity:
    recs = recommend2(uid)
    if len(recs) == 2:
        rows.append((uid, recs[0], recs[1]))
    if len(rows) >= 9:
        break

eval_df = pd.DataFrame(rows, columns=["adventurer_id", "rec1", "rec2"])
out_csv = P("emanuel-eval.csv")
eval_df.to_csv(out_csv, index=False)

print("\nTop 9 users with recommendations (saved to emanuel-eval.csv):")
print(eval_df)
