import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
import matplotlib.pyplot as plt
import csv

# Load data
df_views = pd.read_parquet("week2/content_views.parquet")
df_subs = pd.read_parquet("week2/subscriptions.parquet")

pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())

views_pub = df_views[
    (df_views["publisher_id"] == publisher_id) &
    (df_views["adventurer_id"].isin(sub_ids))
].copy()

# Split data per user
def create_splits(df):
    train_list, test_list = [], []
    for uid in df['adventurer_id'].unique():
        user_data = df[df['adventurer_id'] == uid].sort_values('content_id')
        if len(user_data) >= 5:
            n_train = int(len(user_data) * 0.8)
            train_list.append(user_data.iloc[:n_train])
            test_list.append(user_data.iloc[n_train:])
        else:
            train_list.append(user_data)
    return pd.concat(train_list), pd.concat(test_list)

train_views, test_views = create_splits(views_pub)

# Build KNN model on training data
train_views["value"] = 1
user_item = (
    train_views.groupby(["adventurer_id", "content_id"])["value"]
    .max().unstack(fill_value=0).astype(np.float32)
)

item_user = user_item.T
knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(20, len(item_user)))
knn.fit(item_user.values)
item_matrix = item_user.values

def score_item(uid, item_id):
    """Score how likely user would watch this item"""
    if uid not in user_item.index or item_id not in user_item.columns:
        return 0.0
    
    seen_idx = np.where(user_item.loc[uid].values > 0)[0]
    if len(seen_idx) == 0:
        return 0.0
    
    item_idx = user_item.columns.get_loc(item_id)
    dists, idxs = knn.kneighbors([item_matrix[item_idx]], n_neighbors=min(10, len(item_user)))
    
    # Average similarity to user's watched items
    sims = 1.0 - dists[0]
    overlap = np.isin(idxs[0], seen_idx)
    if overlap.sum() == 0:
        return 0.0
    return sims[overlap].mean()

# Evaluate on the 9 users from your recommendations
with open('week2/emanuel-eval.csv', 'r') as f:
    reader = csv.DictReader(f)
    eval_users = [row['adventurer_id'] for row in reader]

y_true, y_scores = [], []

for uid in eval_users:
    # Positive examples: items in test set
    test_items = test_views[test_views['adventurer_id'] == uid]['content_id'].values
    
    if len(test_items) == 0:
        continue
    
    # Score positive examples
    for item in test_items:
        score = score_item(uid, item)
        y_true.append(1)
        y_scores.append(score)
    
    # Negative examples: random unwatched items
    all_items = user_item.columns.values
    watched = train_views[train_views['adventurer_id'] == uid]['content_id'].values
    unwatched = [i for i in all_items if i not in watched and i not in test_items]
    
    if len(unwatched) > 0:
        neg_samples = np.random.choice(unwatched, min(len(test_items) * 4, len(unwatched)), replace=False)
        for item in neg_samples:
            score = score_item(uid, item)
            y_true.append(0)
            y_scores.append(score)

# Calculate metrics
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)

# Test thresholds
thresholds = np.linspace(0.1, 0.9, 9)
f1_scores = []
for thresh in thresholds:
    y_pred = (np.array(y_scores) >= thresh).astype(int)
    f1_scores.append(f1_score(y_true, y_pred))

best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Emanuel's KNN Recommender Evaluation:")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC: {pr_auc:.3f}")
print(f"Best F1: {best_f1:.3f} at threshold {best_thresh:.2f}")

# Plot
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Emanuel - ROC Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(recall, precision, label=f'PR (AUC={pr_auc:.2f})', color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Emanuel - Precision-Recall Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(thresholds, f1_scores, marker='o', color='blue')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('Emanuel - F1 vs Threshold')
plt.grid(True)

plt.tight_layout()
plt.savefig('week2/emanuel_evaluation.png')
print("\nPlots saved to week2/emanuel_evaluation.png")
plt.show()