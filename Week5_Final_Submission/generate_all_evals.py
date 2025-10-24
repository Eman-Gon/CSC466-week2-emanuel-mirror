"""
Generate all evaluation CSV files needed for evaluate_similarity_based.py

This script creates:
1. collaborative_eval.csv - Pure collaborative filtering (from recommender.py)
2. content_based_eval.csv - Pure content-based filtering 
3. heuristic_eval.csv - Global heuristic/popularity
4. (pre_eval.csv and post_eval.csv already exist from advanced_recommender_week5.py)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("GENERATING ALL EVALUATION FILES")
print("="*60)

# Load data (same as your other scripts)
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

# Get publisher scope
pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"Publisher: {publisher_id}")

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())
views_pub = df_views[df_views["adventurer_id"].isin(sub_ids)]

# Calculate watch_pct (same logic as your other scripts)
df_merged = views_pub.merge(
    df_metadata[['content_id', 'minutes']], 
    on='content_id', 
    how='left'
)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / (df_merged['minutes'] * 60)).clip(0, 1)

df_views_clean = df_merged[
    (df_merged['watch_pct'].fillna(0) >= 0.05) | 
    (df_merged['seconds_viewed'] >= 30)
].copy()

# Build user-item matrix for evaluation user selection
user_item = df_views_clean.groupby(['adventurer_id', 'content_id'])['watch_pct']\
    .max().unstack(fill_value=0).astype(np.float32)

# Select same 9 evaluation users (top activity)
user_activity = user_item.sum(axis=1).sort_values(ascending=False)
eval_users = user_activity.index[:9].tolist()

print(f"\nSelected {len(eval_users)} evaluation users")
print(f"Users: {eval_users}\n")

# ============================================================================
# 1. COLLABORATIVE FILTERING (Pure KNN from recommender.py)
# ============================================================================
print("1. Generating collaborative_eval.csv...")

try:
    # Import your KNN recommender
    import sys
    sys.path.insert(0, str(ROOT))
    from recommender import recommend_for_user
    
    collaborative_recs = []
    for uid in eval_users:
        try:
            recs = recommend_for_user(uid, n_recs=2)
            if len(recs) >= 2:
                collaborative_recs.append({
                    'adventurer_id': uid,
                    'rec1': recs[0],
                    'rec2': recs[1]
                })
                print(f"  ✓ {uid}: {recs[:2]}")
            else:
                print(f"  ⚠ {uid}: Only got {len(recs)} recs")
        except Exception as e:
            print(f"  ✗ {uid}: {e}")
    
    df_collab = pd.DataFrame(collaborative_recs)
    df_collab.to_csv(P('collaborative_eval.csv'), index=False)
    print(f"  Saved: {len(collaborative_recs)} users\n")
    
except Exception as e:
    print(f"  ERROR: Could not generate collaborative_eval.csv: {e}\n")

# ============================================================================
# 2. CONTENT-BASED FILTERING (Pure content similarity)
# ============================================================================
print("2. Generating content_based_eval.csv...")

try:
    # Need to rebuild content-based similarity matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get content items
    content_items = df_metadata[
        df_metadata['content_id'].isin(views_pub['content_id'].unique())
    ].copy()
    
    # Feature engineering (simplified version)
    scaler = StandardScaler()
    content_items['duration_scaled'] = scaler.fit_transform(content_items[['minutes']])
    
    # One-hot encode genres and languages
    genre_dummies = pd.get_dummies(content_items['genre_id'], prefix='genre') if 'genre_id' in content_items.columns else pd.DataFrame()
    lang_dummies = pd.get_dummies(content_items['language_code'], prefix='lang') if 'language_code' in content_items.columns else pd.DataFrame()
    
    # Build feature matrix
    feature_matrix = content_items[['content_id', 'duration_scaled']].set_index('content_id')
    
    if not genre_dummies.empty:
        genre_dummies.index = content_items['content_id'].values
        feature_matrix = feature_matrix.join(genre_dummies, how='left')
    
    if not lang_dummies.empty:
        lang_dummies.index = content_items['content_id'].values
        feature_matrix = feature_matrix.join(lang_dummies, how='left')
    
    feature_matrix = feature_matrix.fillna(0)
    
    # Align with user-item matrix
    common_items = user_item.columns.intersection(feature_matrix.index)
    feature_matrix = feature_matrix.loc[common_items]
    user_item_aligned = user_item[common_items]
    
    # Compute content-based similarity
    item_content_sim = cosine_similarity(feature_matrix.values)
    
    print(f"  Feature matrix: {feature_matrix.shape}")
    print(f"  Similarity matrix: {item_content_sim.shape}")
    
    # Recommendation function
    def recommend_content_based(user_id, n_recs=2):
        if user_id not in user_item_aligned.index:
            return []
        
        user_profile = user_item_aligned.loc[user_id].values
        seen_idx = np.where(user_profile > 0)[0]
        
        if len(seen_idx) == 0:
            return []
        
        # Pure content-based scoring
        user_weights = user_profile[seen_idx]
        scores = (item_content_sim[seen_idx].T * user_weights).sum(axis=1)
        scores[seen_idx] = -np.inf
        
        top_idx = np.argsort(-scores)[:n_recs]
        return [user_item_aligned.columns[i] for i in top_idx if np.isfinite(scores[i])]
    
    content_recs = []
    for uid in eval_users:
        try:
            recs = recommend_content_based(uid, n_recs=2)
            if len(recs) >= 2:
                content_recs.append({
                    'adventurer_id': uid,
                    'rec1': recs[0],
                    'rec2': recs[1]
                })
                print(f"  ✓ {uid}: {recs[:2]}")
            else:
                print(f"  ⚠ {uid}: Only got {len(recs)} recs")
        except Exception as e:
            print(f"  ✗ {uid}: {e}")
    
    df_content = pd.DataFrame(content_recs)
    df_content.to_csv(P('content_based_eval.csv'), index=False)
    print(f"  Saved: {len(content_recs)} users\n")
    
except Exception as e:
    print(f"  ERROR: Could not generate content_based_eval.csv: {e}\n")

# ============================================================================
# 3. HEURISTIC/POPULARITY (Global trending)
# ============================================================================
print("3. Generating heuristic_eval.csv...")

try:
    # Import your heuristic recommender
    from heuristic_recommender import recommend_trending
    
    heuristic_recs = []
    for uid in eval_users:
        try:
            recs = recommend_trending(uid, n_recs=2)
            if len(recs) >= 2:
                heuristic_recs.append({
                    'adventurer_id': uid,
                    'rec1': recs[0],
                    'rec2': recs[1]
                })
                print(f"  ✓ {uid}: {recs[:2]}")
            else:
                print(f"  ⚠ {uid}: Only got {len(recs)} recs")
        except Exception as e:
            print(f"  ✗ {uid}: {e}")
    
    df_heuristic = pd.DataFrame(heuristic_recs)
    df_heuristic.to_csv(P('heuristic_eval.csv'), index=False)
    print(f"  Saved: {len(heuristic_recs)} users\n")
    
except Exception as e:
    print(f"  ERROR: Could not generate heuristic_eval.csv: {e}\n")

# ============================================================================
# Summary
# ============================================================================
print("="*60)
print("SUMMARY")
print("="*60)

eval_files = [
    'collaborative_eval.csv',
    'content_based_eval.csv', 
    'heuristic_eval.csv',
    'pre_eval.csv',
    'post_eval.csv'
]

for fname in eval_files:
    fpath = P(fname)
    if fpath.exists():
        df = pd.read_csv(fpath)
        print(f"✓ {fname:<30} {len(df)} users")
    else:
        print(f"✗ {fname:<30} MISSING")

print("\n✅ Ready to run: python evaluate_similarity_based.py")