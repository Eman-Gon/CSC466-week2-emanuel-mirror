import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("GENERATING FINAL SUBMISSION")
print("="*60)

# Load data
df_views = pd.read_parquet(P("content_views.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

# Get publisher wn32 subscribers
pub_subs = df_subs[df_subs['publisher_id'] == 'wn32']

# Calculate user activity
user_activity = df_views[
    df_views['adventurer_id'].isin(pub_subs['adventurer_id'])
].groupby('adventurer_id')['content_id'].count().sort_values(ascending=False)

# Select 20 users strategically
top_users = user_activity.head(15).index.tolist()
medium_users = user_activity.iloc[20:25].index.tolist()
selected_users = top_users + medium_users

print(f"Selected {len(selected_users)} users")
print(f"Activity range: {user_activity[selected_users].min():.0f} to {user_activity[selected_users].max():.0f} views\n")

# ============================================================================
# CHOOSE YOUR METHOD HERE
# ============================================================================
# Based on test_final_methods.py results, uncomment ONE of these:

# # Option A: Pure Collaborative (if it worked for all 20 users)
# from recommender import recommend_for_user
# METHOD_NAME = "Collaborative Filtering"
# def get_recommendations(user_id, n_recs=2):
#     return recommend_for_user(user_id, n_recs=n_recs)

# Option B: Hybrid (if collaborative alone had issues)
from advanced_recommender_week5 import recommend_hybrid
METHOD_NAME = "Hybrid (60% Collaborative + 40% Content)"
def get_recommendations(user_id, n_recs=2):
    return recommend_hybrid(user_id, n_recs=n_recs)

# Option C: Adaptive (uses heuristic as fallback)
# from recommender import recommend_for_user
# from heuristic_recommender import recommend_trending
# METHOD_NAME = "Adaptive (Collaborative with Heuristic Fallback)"
# def get_recommendations(user_id, n_recs=2):
#     recs = recommend_for_user(user_id, n_recs=n_recs)
#     if len(recs) < 2:
#         recs = recommend_trending(user_id, n_recs=2)
#     return recs

# ============================================================================

print(f"Using method: {METHOD_NAME}\n")

# Generate recommendations
final_recs = []
failed_users = []

for user_id in selected_users:
    try:
        recs = get_recommendations(user_id, n_recs=2)
        
        if len(recs) >= 2:
            final_recs.append({
                'adventurer_id': user_id,
                'rec1': recs[0],
                'rec2': recs[1]
            })
            print(f"✓ {user_id}: {recs[0]}, {recs[1]}")
        else:
            failed_users.append(user_id)
            print(f"⚠️  {user_id}: Only got {len(recs)} recommendations")
            
    except Exception as e:
        failed_users.append(user_id)
        print(f"❌ {user_id}: Error - {e}")

# Save final CSV
output_df = pd.DataFrame(final_recs)
output_df.to_csv(P('final_recommendations.csv'), index=False)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Method: {METHOD_NAME}")
print(f"Successful: {len(final_recs)}/20 users")
print(f"Failed: {len(failed_users)} users")

if failed_users:
    print(f"Failed users: {failed_users}")
    print("\n⚠️  WARNING: You don't have 20 users! Consider using fallback method.")
else:
    print("\n✅ SUCCESS! All 20 users have recommendations")

# Validate recommendations
all_recs = pd.concat([output_df['rec1'], output_df['rec2']])
unique_items = len(all_recs.unique())
print(f"\nUnique items recommended: {unique_items}")
print(f"Coverage: {unique_items/38*100:.1f}% of publisher catalog")

# Check if recommendations are in scope
publisher_content = df_views[
    df_views['adventurer_id'].isin(pub_subs['adventurer_id'])
]['content_id'].unique()

invalid_recs = all_recs[~all_recs.isin(publisher_content)]
if len(invalid_recs) > 0:
    print(f"\n⚠️  WARNING: {len(invalid_recs)} recommendations outside publisher scope!")
else:
    print("\n✅ All recommendations are valid publisher content")

print("\n" + "="*60)
print("NEXT STEP: Update writeup.md to match your chosen method")
print("="*60)