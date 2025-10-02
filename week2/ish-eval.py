import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, auc, f1_score
)
import matplotlib.pyplot as plt


df_content_views = pd.read_parquet("./week2/content_views.parquet", engine="pyarrow")
df_subscriptions = pd.read_parquet("./week2/subscriptions.parquet", engine="pyarrow")
df_cancellations = pd.read_parquet("./week2/cancellations.parquet", engine="pyarrow")

df_content_views = df_content_views.dropna(subset=['rating'])

df_content_views = df_content_views.drop_duplicates(['adventurer_id', 'content_id'])

train, test = train_test_split(df_content_views, test_size=0.2, random_state=42)
df_matrix = train.pivot(index="content_id", columns="adventurer_id", values="rating").fillna(0)

item_means = df_matrix.mean(axis=1)
df_matrix_centered = df_matrix.sub(item_means, axis=0)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(df_matrix_centered.values)

def recommend(item_id, user_id, k=10):   
    if item_id not in df_matrix_centered.index: #might not be in the train set
        return []

    item = df_matrix_centered.loc[item_id]
    dists, indices = knn_model.kneighbors([item.values], n_neighbors=k)

    candidates = list(df_matrix_centered.index[indices[0]])
    candidates = [c for c in candidates if c != item_id]  #drop content i used
    watched = set(train[train["adventurer_id"] == user_id]["content_id"])
    recs = [c for c in candidates if c not in watched]

    return recs[:k]


def evaluate_with_curves(k=10, n_samples=200):
    y_true_all = []
    y_score_all = []

    users = test["adventurer_id"].unique()
    np.random.shuffle(users) #testing on all users
    users = users[:n_samples] 

    for user_id in users:
        #find content user liked in test
        liked = set(test[(test["adventurer_id"] == user_id) & (test["rating"] >= 4)]["content_id"])
        if len(liked) == 0:
            continue

        #content that user has watched in train
        train_items = train[train["adventurer_id"] == user_id]["content_id"].unique()
        if len(train_items) == 0:
            continue

        #find content to feed in
        c = np.random.choice(train_items)
        if c not in df_matrix_centered.index:
            continue

        item = df_matrix_centered.loc[c]
        dists, indices = knn_model.kneighbors([item.values], n_neighbors=10)  
        recs = list(df_matrix_centered.index[indices[0]])
        
        #had to add in similarity scores to be able to evaluate
        for idx, r in enumerate(recs):
            if r == c:
                continue
            y_true_all.append(1 if r in liked else 0)
            y_score_all.append(1 - dists[0][idx])


    y_actual = np.array(y_true_all)
    y_pred = np.array(y_score_all)

    # -- ROC-AUC --
    fpr, tpr, _ = roc_curve(y_actual, y_pred)
    roc_auc = roc_auc_score(y_actual, y_pred)
    print(f"ROC-AUC = {roc_auc:.4f}")

    # -- PR-AUC --
    precision, recall, thresholds = precision_recall_curve(y_actual, y_pred)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC = {pr_auc:.4f}")

   # -- F1 - Threshold --
    f1_scores = []
    test_thresholds = np.linspace(0.0, 1.0, 11)  # 0.0 to 1.0 step 0.1
    for thresh in test_thresholds:
        preds = (y_pred >= thresh).astype(int)
        f1_scores.append(f1_score(y_actual, preds))

    # --- Plotting ---
    plt.figure(figsize=(18, 5))

    # ROC Curve
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

    # Precision-Recall Curve
    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})', color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)

    # F1 vs Threshold
    plt.subplot(1, 3, 3)
    plt.plot(test_thresholds, f1_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.xticks(test_thresholds)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return roc_auc, pr_auc, f1_scores, test_thresholds

# --------------------------
# Run evaluation
# --------------------------
roc_auc, pr_auc, f1_scores, test_thresholds = evaluate_with_curves(k=10, n_samples=500)
