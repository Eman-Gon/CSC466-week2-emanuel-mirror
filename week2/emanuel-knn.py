import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load Week 2 data
df_views = pd.read_parquet("week2/content_views.parquet")
df_subs = pd.read_parquet("week2/subscriptions.parquet")

pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"Publisher chosen: {publisher_id}")

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())

views_pub = df_views[
    (df_views["publisher_id"] == publisher_id) &
    (df_views["adventurer_id"].isin(sub_ids))
].copy()

# ADD 80/20 SPLIT PER USER - THIS IS NEW
def create_train_test_split(df):
    """Split each user's data: 80% train, 20% test"""
    train_list = []
    test_list = []
    
    for uid in df['adventurer_id'].unique():
        user_data = df[df['adventurer_id'] == uid].sort_values('content_id')
        
        if len(user_data) >= 5:
            n_train = int(len(user_data) * 0.8)
            train_list.append(user_data.iloc[:n_train])
            test_list.append(user_data.iloc[n_train:])
        else:
            train_list.append(user_data)
    
    return pd.concat(train_list), pd.concat(test_list)

# Split the data
train_views, test_views = create_train_test_split(views_pub)
print(f"Train: {len(train_views)} rows, Test: {len(test_views)} rows")

# Build user-item matrix on TRAINING DATA ONLY
train_views["value"] = 1
user_item = (
    train_views.groupby(["adventurer_id", "content_id"])["value"]
    .max().unstack(fill_value=0).astype(np.float32)
)

print(f"User-item matrix: {user_item.shape}")

item_user = user_item.T
knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(20, len(item_user)))
knn.fit(item_user.values)

items_index = item_user.index.to_list()
item_matrix = item_user.values

def recommend2(uid: str) -> list[str]:
    """Recommend 2 unseen items for a user"""
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
    top = np.argsort(-scores)[:2]
    return [user_item.columns[i] for i in top if np.isfinite(scores[i])]

# Pick 9 adventurers (not 3, assignment says 9)
user_activity = user_item.sum(axis=1).sort_values(ascending=False).index.tolist()
rows, picked_users = [], []
for uid in user_activity:
    recs = recommend2(uid)
    if len(recs) == 2:
        rows.append((uid, recs[0], recs[1]))
        picked_users.append(uid)
    if len(rows) >= 9:  # Changed from 3 to 9
        break

eval_df = pd.DataFrame(rows, columns=["adventurer_id", "rec1", "rec2"])
eval_df.to_csv("week2/emanuel-eval.csv", index=False)
print("\nTop 9 users with recommendations:")
print(eval_df)