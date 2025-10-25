import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name


df_views = pd.read_parquet(P("content_views.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

pub_subs = df_subs[df_subs['publisher_id'] == 'wn32']

user_activity = df_views[
    df_views['adventurer_id'].isin(pub_subs['adventurer_id'])
].groupby('adventurer_id')['content_id'].count().sort_values(ascending=False)

top_users = user_activity.head(15).index.tolist()
medium_users = user_activity.iloc[20:25].index.tolist()
selected_users = top_users + medium_users

print(f"Selected {len(selected_users)} users")
print(f"Activity range: {user_activity[selected_users].min():.0f} to {user_activity[selected_users].max():.0f} views\n")

from advanced_recommender_week5 import recommend_hybrid
METHOD_NAME = "Hybrid (60% Collaborative + 40% Content)"
def get_recommendations(user_id, n_recs=2):
    return recommend_hybrid(user_id, n_recs=n_recs)


print(f"Using method: {METHOD_NAME}\n")

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
        print(f" {user_id}: Error - {e}")

output_df = pd.DataFrame(final_recs)
output_df.to_csv(P('final_recommendations.csv'), index=False)


print("SUMMARY")

print(f"Method: {METHOD_NAME}")
print(f"Successful: {len(final_recs)}/20 users")
print(f"Failed: {len(failed_users)} users")

if failed_users:
    print(f"Failed users: {failed_users}")
    print("\n  WARNING: You don't have 20 users! Consider using fallback method.")
else:
    print("\nSUCCESS! All 20 users have recommendations")

all_recs = pd.concat([output_df['rec1'], output_df['rec2']])
unique_items = len(all_recs.unique())
print(f"\nUnique items recommended: {unique_items}")
print(f"Coverage: {unique_items/38*100:.1f}% of publisher catalog")

publisher_content = df_views[
    df_views['adventurer_id'].isin(pub_subs['adventurer_id'])
]['content_id'].unique()

invalid_recs = all_recs[~all_recs.isin(publisher_content)]
if len(invalid_recs) > 0:
    print(f"\n WARNING: {len(invalid_recs)} recommendations outside publisher scope!")
else:
    print("\nAll recommendations are valid publisher content")
