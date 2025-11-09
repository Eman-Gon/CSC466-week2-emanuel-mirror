import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set up paths
ROOT = Path.cwd()
P = lambda name: ROOT / name

print("="*60)
print("SIMPLE COLLABORATIVE FILTERING SUBMISSION")
print("="*60)

# Load data
print("\n[1] Loading data...")
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

# Clean and prepare data
print("\n[2] Preparing data...")
df_views = df_views.sort_values('seconds_viewed', ascending=False)\
    .drop_duplicates(subset=['adventurer_id', 'content_id'], keep='first')

# Calculate watch percentage
df_merged = df_views.merge(
    df_metadata[['content_id', 'minutes']], 
    on='content_id', 
    how='left'
)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / (df_merged['minutes'] * 60)).clip(0, 1)

# Filter low engagement
df_views_clean = df_merged[
    (df_merged['watch_pct'].fillna(0) >= 0.05) | 
    (df_merged['seconds_viewed'] >= 30)
].copy()

# Get publisher with most subscribers (likely wn32)
pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"  Publisher: {publisher_id} ({pub_counts.max():,} subscribers)")

# Filter to publisher's subscribers
subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())
views_pub = df_views_clean[df_views_clean["adventurer_id"].isin(sub_ids)].copy()

# Build user-item matrix
print("\n[3] Building collaborative filtering model...")
user_item = views_pub.groupby(['adventurer_id', 'content_id'])['watch_pct']\
    .max().unstack(fill_value=0).astype(np.float32)

print(f"  User-item matrix: {user_item.shape}")

# Compute item similarity
item_sim = cosine_similarity(user_item.T.values)

def recommend_collaborative(user_id, n_recs=3):
    """Pure collaborative filtering recommendations"""
    if user_id not in user_item.index:
        return recommend_popular(n_recs)
    
    user_profile = user_item.loc[user_id].values
    seen_idx = np.where(user_profile > 0)[0]
    
    if len(seen_idx) == 0:
        return recommend_popular(n_recs)
    
    # Calculate scores
    scores = (item_sim[seen_idx].T * user_profile[seen_idx]).sum(axis=1)
    scores[seen_idx] = -np.inf
    
    # Get top recommendations
    top_idx = np.argsort(-scores)[:n_recs]
    recs = [user_item.columns[i] for i in top_idx if np.isfinite(scores[i])]
    
    # Fill with popular if not enough
    if len(recs) < n_recs:
        popular = recommend_popular(n_recs - len(recs), exclude=set(recs))
        recs.extend(popular)
    
    return recs[:n_recs]

def recommend_popular(n_recs=3, exclude=None):
    """Most popular items as fallback"""
    popularity = views_pub['content_id'].value_counts()
    
    if exclude:
        popularity = popularity[~popularity.index.isin(exclude)]
    
    return popularity.head(n_recs).index.tolist()

# Select 30 users
print("\n[4] Selecting 30 users...")

# Get users with varying activity levels
user_activity = user_item.sum(axis=1).sort_values(ascending=False)

# Take top 30 most active users (most likely to have good recommendations)
selected_users = user_activity.head(30).index.tolist()

print(f"  Selected {len(selected_users)} users")
print(f"  Activity range: {user_activity[selected_users].min():.0f} - {user_activity[selected_users].max():.0f} views")

# Generate recommendations
print("\n[5] Generating recommendations...")
final_recs = []

for i, user_id in enumerate(selected_users, 1):
    recs = recommend_collaborative(user_id, n_recs=3)
    
    final_recs.append({
        'adventurer_id': user_id,
        'rec1': recs[0] if len(recs) > 0 else '',
        'rec2': recs[1] if len(recs) > 1 else '',
        'rec3': recs[2] if len(recs) > 2 else ''
    })
    
    if i % 10 == 0:
        print(f"  Progress: {i}/30 users completed")

# Save results
df_final = pd.DataFrame(final_recs)
df_final.to_csv(P('eval_simple.csv'), index=False)

print("\n[6] Summary")
print("="*60)
print(f"✅ Generated eval_simple.csv")
print(f"   Users: {len(df_final)}")

# Check diversity
all_recs = []
for col in ['rec1', 'rec2', 'rec3']:
    all_recs.extend(df_final[col].dropna().tolist())

unique_items = len(set(all_recs))
print(f"   Unique items: {unique_items}")
print(f"   Coverage: {unique_items/len(user_item.columns)*100:.1f}%")

print(f"\nFirst 5 rows:")
print(df_final.head().to_string(index=False))

print("\n✓ Submission ready: eval_simple.csv")