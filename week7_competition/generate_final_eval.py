import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set up paths (adjust if needed)
ROOT = Path.cwd()
P = lambda name: ROOT / name

print("="*60)
print("GENERATING FINAL COMPETITION SUBMISSION")
print("="*60)

# Load data
print("\n[1] Loading data...")
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

print(f"  Views: {len(df_views):,}")
print(f"  Content: {len(df_metadata):,}")
print(f"  Adventurers: {len(df_adventurers):,}")
print(f"  Subscriptions: {len(df_subs):,}")

# Data cleaning
print("\n[2] Cleaning data...")
df_views = df_views.sort_values('seconds_viewed', ascending=False)\
    .drop_duplicates(subset=['adventurer_id', 'content_id'], keep='first')

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
print(f"  Clean views: {len(df_views_clean):,}")

# Get publisher with most subscribers
pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"\n[3] Selected publisher: {publisher_id} ({pub_counts.max():,} subscribers)")

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())

# Filter views to this publisher's subscribers
views_pub = df_views_clean[
    df_views_clean["adventurer_id"].isin(sub_ids)
].copy()

print(f"  Publisher views: {len(views_pub):,}")
print(f"  Unique users: {views_pub['adventurer_id'].nunique():,}")
print(f"  Unique content: {views_pub['content_id'].nunique():,}")

# Build feature matrix for content-based filtering
print("\n[4] Building content features...")
content_items = df_metadata[
    df_metadata['content_id'].isin(views_pub['content_id'].unique())
].copy()

# Scale duration
scaler = StandardScaler()
content_items['duration_scaled'] = scaler.fit_transform(content_items[['minutes']])

# Initialize feature matrix
feature_matrix = content_items[['content_id', 'duration_scaled']].set_index('content_id')

# Add genre features
if 'genre_id' in content_items.columns:
    genre_dummies = pd.get_dummies(content_items['genre_id'], prefix='genre')
    genre_dummies.index = content_items['content_id'].values
    feature_matrix = feature_matrix.join(genre_dummies, how='left')
    print(f"  Added {len(genre_dummies.columns)} genre features")

# Add language features
if 'language_code' in content_items.columns:
    lang_dummies = pd.get_dummies(content_items['language_code'], prefix='lang')
    lang_dummies.index = content_items['content_id'].values
    feature_matrix = feature_matrix.join(lang_dummies, how='left')
    print(f"  Added {len(lang_dummies.columns)} language features")

feature_matrix = feature_matrix.fillna(0)
print(f"  Final feature matrix: {feature_matrix.shape}")

# Build user-item matrix for collaborative filtering
print("\n[5] Building user-item matrix...")
user_item = views_pub.groupby(['adventurer_id', 'content_id'])['watch_pct']\
    .max().unstack(fill_value=0).astype(np.float32)

# Align content features with user-item matrix
common_items = user_item.columns.intersection(feature_matrix.index)
feature_matrix = feature_matrix.loc[common_items]
user_item = user_item[common_items]
print(f"  User-item matrix: {user_item.shape}")
print(f"  Aligned items: {len(common_items)}")

# Compute similarity matrices
print("\n[6] Computing similarity matrices...")
item_collab_sim = cosine_similarity(user_item.T.values)
item_content_sim = cosine_similarity(feature_matrix.values)

# Hybrid weights (60% collaborative, 40% content-based)
ALPHA = 0.6
BETA = 0.4
item_hybrid_sim = ALPHA * item_collab_sim + BETA * item_content_sim
print(f"  Hybrid weights: {ALPHA:.0%} collaborative, {BETA:.0%} content-based")

def recommend_hybrid(user_id, n_recs=3):
    """Generate recommendations using hybrid similarity"""
    if user_id not in user_item.index:
        return []
    
    user_profile = user_item.loc[user_id].values
    seen_idx = np.where(user_profile > 0)[0]
    
    if len(seen_idx) == 0:
        return []
    
    # Weight by user's engagement level
    user_weights = user_profile[seen_idx]
    scores = (item_hybrid_sim[seen_idx].T * user_weights).sum(axis=1)
    
    # Exclude already seen items
    scores[seen_idx] = -np.inf
    
    # Get top recommendations
    top_idx = np.argsort(-scores)[:n_recs]
    return [user_item.columns[i] for i in top_idx if np.isfinite(scores[i])]

def recommend_fallback(n_recs=3):
    """Fallback: most popular items"""
    popularity = views_pub['content_id'].value_counts()
    return popularity.head(n_recs).index.tolist()

# Select 30 diverse users for evaluation
print("\n[7] Selecting 30 users for evaluation...")

# Calculate user activity
user_activity = user_item.sum(axis=1).sort_values(ascending=False)
print(f"  Total users in matrix: {len(user_activity)}")

# Strategy: Mix of different activity levels for diversity
# - 10 highly active users (>15 views)
# - 10 moderately active (5-15 views) 
# - 10 less active but engaged (2-5 views)

highly_active = user_activity[user_activity > 15].head(10).index.tolist()
moderately_active = user_activity[(user_activity >= 5) & (user_activity <= 15)].head(10).index.tolist()
less_active = user_activity[(user_activity >= 2) & (user_activity < 5)].head(10).index.tolist()

# Combine and ensure we have 30
selected_users = highly_active + moderately_active + less_active

# If we don't have enough in any category, fill from top users
if len(selected_users) < 30:
    remaining_needed = 30 - len(selected_users)
    additional = user_activity[~user_activity.index.isin(selected_users)].head(remaining_needed).index.tolist()
    selected_users.extend(additional)

selected_users = selected_users[:30]  # Ensure exactly 30

print(f"  Selected {len(selected_users)} users:")
print(f"    - Highly active (>15 views): {len(highly_active)}")
print(f"    - Moderately active (5-15 views): {len(moderately_active)}")
print(f"    - Less active (2-5 views): {len(less_active)}")

# Generate recommendations
print("\n[8] Generating recommendations...")
final_recs = []
success_count = 0
fallback_count = 0

for i, user_id in enumerate(selected_users, 1):
    try:
        recs = recommend_hybrid(user_id, n_recs=3)
        
        if len(recs) >= 3:
            final_recs.append({
                'adventurer_id': user_id,
                'rec1': recs[0],
                'rec2': recs[1],
                'rec3': recs[2]
            })
            success_count += 1
            print(f"  [{i:2d}/30] {user_id}: ✓ Generated 3 recommendations")
        else:
            # Use fallback if not enough recommendations
            fallback = recommend_fallback(3)
            final_recs.append({
                'adventurer_id': user_id,
                'rec1': fallback[0] if len(fallback) > 0 else '',
                'rec2': fallback[1] if len(fallback) > 1 else '',
                'rec3': fallback[2] if len(fallback) > 2 else ''
            })
            fallback_count += 1
            print(f"  [{i:2d}/30] {user_id}: ⚠ Used fallback (only {len(recs)} recs)")
            
    except Exception as e:
        # Emergency fallback
        fallback = recommend_fallback(3)
        final_recs.append({
            'adventurer_id': user_id,
            'rec1': fallback[0] if len(fallback) > 0 else '',
            'rec2': fallback[1] if len(fallback) > 1 else '',
            'rec3': fallback[2] if len(fallback) > 2 else ''
        })
        fallback_count += 1
        print(f"  [{i:2d}/30] {user_id}: ✗ Error - {e}")

# Save to eval.csv
df_final = pd.DataFrame(final_recs)
df_final.to_csv(P('eval.csv'), index=False)

print("\n[9] Results Summary")
print("="*60)
print(f"✅ SUCCESS! Generated eval.csv")
print(f"   Total users: {len(df_final)}")
print(f"   Successful hybrid recs: {success_count}")
print(f"   Fallback used: {fallback_count}")

# Analyze diversity
all_recs = []
for col in ['rec1', 'rec2', 'rec3']:
    all_recs.extend(df_final[col].dropna().tolist())

unique_items = len(set(all_recs))
total_recs = len(all_recs)
print(f"\nDiversity Analysis:")
print(f"   Unique items recommended: {unique_items}")
print(f"   Total recommendations: {total_recs}")
print(f"   Coverage: {unique_items/views_pub['content_id'].nunique()*100:.1f}% of available content")

# Show sample
print(f"\nSample of recommendations (first 5 users):")
print(df_final.head().to_string(index=False))

print("\n" + "="*60)
print("SUBMISSION READY!")
print("File: eval.csv")
print("Format: adventurer_id, rec1, rec2, rec3")
print(f"Rows: {len(df_final)} adventurers")
print("="*60)

# Validate the submission
print("\n[10] Validating submission...")
issues = []

# Check for missing values
for col in ['adventurer_id', 'rec1', 'rec2', 'rec3']:
    missing = df_final[col].isna().sum()
    if missing > 0:
        issues.append(f"  ⚠ {missing} missing values in {col}")

# Check for duplicates
duplicates = df_final['adventurer_id'].duplicated().sum()
if duplicates > 0:
    issues.append(f"  ⚠ {duplicates} duplicate adventurer_ids")

# Check that recommendations are valid content IDs
valid_content = set(views_pub['content_id'].unique())
for col in ['rec1', 'rec2', 'rec3']:
    invalid = df_final[~df_final[col].isin(valid_content) & df_final[col].notna()][col].nunique()
    if invalid > 0:
        issues.append(f"  ⚠ {invalid} invalid content IDs in {col}")

if issues:
    print("Issues found:")
    for issue in issues:
        print(issue)
else:
    print("✓ All validation checks passed!")
    
print("\n🎯 Ready to submit eval.csv for competition!")