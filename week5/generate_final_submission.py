import pandas as pd
import numpy as np
from pathlib import Path
from advanced_recommender_week4 import recommend_hybrid, user_item

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("GENERATING FINAL SUBMISSION")
print("="*60)

# Strategy: Use a mix of methods for different user types
# - For users with lots of history: Use hybrid (best for personalization)
# - For users with little history: Use heuristic (safer)

# Calculate user activity
user_activity = user_item.sum(axis=1).sort_values(ascending=False)

print(f"\nTotal users in system: {len(user_activity)}")
print(f"User activity range: {user_activity.min():.0f} to {user_activity.max():.0f} views")

# Select 20 diverse users
# Strategy: 
# - 10 highly active users (>10 views) - use hybrid
# - 5 moderately active (5-10 views) - use hybrid
# - 5 less active (3-5 views) - use heuristic + hybrid blend

highly_active = user_activity[user_activity >= 10].head(10).index.tolist()
moderately_active = user_activity[(user_activity >= 5) & (user_activity < 10)].head(5).index.tolist()
less_active = user_activity[(user_activity >= 3) & (user_activity < 5)].head(5).index.tolist()

# If we don't have enough, fill from top users
selected_users = highly_active + moderately_active + less_active
if len(selected_users) < 20:
    additional = user_activity.head(20 - len(selected_users)).index.tolist()
    selected_users = list(set(selected_users + additional))[:20]

selected_users = selected_users[:20]

print(f"\nSelected {len(selected_users)} users:")
print(f"  - Highly active (≥10 views): {len([u for u in selected_users if user_activity[u] >= 10])}")
print(f"  - Moderately active (5-9 views): {len([u for u in selected_users if 5 <= user_activity[u] < 10])}")
print(f"  - Less active (3-4 views): {len([u for u in selected_users if 3 <= user_activity[u] < 5])}")

# Generate recommendations
final_recs = []
success_count = 0

for user_id in selected_users:
    try:
        # Use hybrid for all users (it balances personalization and popularity)
        recs = recommend_hybrid(user_id, n_recs=2)
        
        if len(recs) >= 2:
            final_recs.append({
                'adventurer_id': user_id,
                'rec1': recs[0],
                'rec2': recs[1]
            })
            success_count += 1
    except Exception as e:
        print(f"  ⚠️  Error for user {user_id}: {e}")

# Save
df_final = pd.DataFrame(final_recs)
df_final.to_csv(P('final_recommendations.csv'), index=False)

print(f"\n✅ SUCCESS!")
print(f"   Saved {len(df_final)} users to final_recommendations.csv")
print(f"   Success rate: {success_count}/{len(selected_users)} ({success_count/len(selected_users)*100:.1f}%)")

# Show sample
print(f"\nSample recommendations:")
print(df_final.head(5).to_string(index=False))

# Analyze diversity
all_recs = set(df_final['rec1'].tolist() + df_final['rec2'].tolist())
print(f"\nDiversity:")
print(f"  Unique items recommended: {len(all_recs)}")
print(f"  Total recommendations: {len(df_final) * 2}")
print(f"  Diversity ratio: {len(all_recs) / (len(df_final) * 2) * 100:.1f}%")

print("\n" + "="*60)
print("READY FOR SUBMISSION!")
print("="*60)
print("Files to submit:")
print("  1. final_recommendations.csv")
print("  2. All Python scripts in week5/")
print("  3. Your 2-page report")
print("="*60)