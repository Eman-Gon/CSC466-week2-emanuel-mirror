import pandas as pd
import numpy as np
from pathlib import Path

# We're already in week5 directory
ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("DEBUG: Why is precision 0%?")
print("="*60)

# Load data
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

# Get publisher scope
pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())
views_pub = df_views[df_views["adventurer_id"].isin(sub_ids)]
publisher_content_scope = set(views_pub['content_id'].unique())

# Calculate watch_pct
df_merged = df_views.merge(df_metadata[['content_id', 'minutes']], on='content_id', how='left')
denom = (df_merged['minutes'] * 60).replace(0, np.nan)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / denom).clip(0, 1)

# Load one of your recommendation files
collab_recs = pd.read_csv(P('collaborative_eval.csv'))
test_user = collab_recs['adventurer_id'].iloc[0]

print(f"\n1. Testing user: {test_user}")
print(f"   Recommendations: {collab_recs.iloc[0]['rec1']}, {collab_recs.iloc[0]['rec2']}")

# What did this user actually watch?
user_views = df_merged[df_merged['adventurer_id'] == test_user]
print(f"\n2. User's viewing history ({len(user_views)} total views):")
print(user_views[['content_id', 'seconds_viewed', 'watch_pct']].head(10))

# What did they "like" (>50% watch)?
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"\n3. Content user 'liked' at different thresholds:")
for thresh in THRESHOLDS:
    liked = user_views[user_views['watch_pct'] >= thresh]['content_id'].unique()
    # Filter to publisher scope
    liked_in_scope = [c for c in liked if c in publisher_content_scope]
    print(f"   {thresh*100:.0f}% threshold: {len(liked_in_scope)} items in scope")
    if len(liked_in_scope) > 0 and len(liked_in_scope) <= 5:
        print(f"      Items: {liked_in_scope}")

# Check if recommendations are in the liked set
print(f"\n4. Are recommendations hitting likes?")
rec1, rec2 = collab_recs.iloc[0]['rec1'], collab_recs.iloc[0]['rec2']
for thresh in THRESHOLDS:
    liked = set(user_views[user_views['watch_pct'] >= thresh]['content_id'].unique())
    liked = liked & publisher_content_scope
    hit1 = rec1 in liked
    hit2 = rec2 in liked
    print(f"   {thresh*100:.0f}% threshold: rec1={hit1}, rec2={hit2}")

# Check ALL test users
print(f"\n5. Checking ALL {len(collab_recs)} test users:")
for thresh in THRESHOLDS:
    total_likes = 0
    total_recs = 0
    users_with_likes = 0
    
    for _, row in collab_recs.iterrows():
        user_id = row['adventurer_id']
        user_views = df_merged[df_merged['adventurer_id'] == user_id]
        user_views = user_views[user_views['content_id'].isin(publisher_content_scope)]
        
        liked = set(user_views[user_views['watch_pct'] >= thresh]['content_id'].unique())
        
        if len(liked) > 0:
            users_with_likes += 1
            total_likes += len(liked)
        
        recs = [row['rec1'], row['rec2']]
        total_recs += len(recs)
    
    avg_likes = total_likes / len(collab_recs) if len(collab_recs) > 0 else 0
    print(f"   {thresh*100:.0f}% threshold:")
    print(f"      - {users_with_likes}/{len(collab_recs)} users have liked content")
    print(f"      - Average {avg_likes:.1f} liked items per user")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("If users have 0 liked items, precision will always be 0!")
print("Try lowering the threshold or checking if watch_pct is calculated correctly.")
print("="*60)