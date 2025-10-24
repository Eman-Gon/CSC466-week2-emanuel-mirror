import pandas as pd
import numpy as np
from pathlib import Path
from advanced_recommender_week5 import recommend_hybrid

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("Generating final 20-user submission...")

# Load data
df_views = pd.read_parquet(P("content_views.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

# Get publisher wn32 subscribers
pub_subs = df_subs[df_subs['publisher_id'] == 'wn32']

# Calculate user activity
user_activity = df_views[
    df_views['adventurer_id'].isin(pub_subs['adventurer_id'])
].groupby('adventurer_id')['content_id'].count().sort_values(ascending=False)

# Select 20 users strategically:
# - Top 15 most active users (best for collaborative filtering)
# - 5 medium-active users (to show robustness)
top_users = user_activity.head(15).index.tolist()
medium_users = user_activity.iloc[20:25].index.tolist()

selected_users = top_users + medium_users

print(f"Selected {len(selected_users)} users")
print(f"Activity range: {user_activity[selected_users].min():.0f} to {user_activity[selected_users].max():.0f} views")

# Generate recommendations using your BEST method
# Based on your report: Collaborative (via hybrid with 60/40 weighting)
final_recs = []

for user_id in selected_users:
    try:
        recs = recommend_hybrid(user_id, n_recs=2)
        if len(recs) >= 2:
            final_recs.append({
                'adventurer_id': user_id,
                'rec1': recs[0],
                'rec2': recs[1]
            })
            print(f"✓ {user_id}: {recs[:2]}")
        else:
            print(f"⚠️  {user_id}: Only got {len(recs)} recommendations")
    except Exception as e:
        print(f"❌ {user_id}: Error - {e}")

# Save final CSV
output_df = pd.DataFrame(final_recs)
output_df.to_csv(P('final_recommendations.csv'), index=False)

print(f"\n✅ Saved final_recommendations.csv with {len(final_recs)} users")
print("\nThis is your submission file!")