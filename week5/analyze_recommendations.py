import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("WHY ARE RECOMMENDATIONS MISSING?")
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
df_merged = views_pub.merge(df_metadata[['content_id', 'minutes']], on='content_id', how='left')
denom = (df_merged['minutes'] * 60).replace(0, np.nan)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / denom).clip(0, 1)

# Load recommendations
collab_recs = pd.read_csv(P('collaborative_eval.csv'))

# Analyze user 52st who got 2/2 hits with heuristic
test_user = '52st'

print(f"\n1. Analyzing user: {test_user}")
print("="*60)

# What did this user watch and like?
user_views = df_merged[
    (df_merged['adventurer_id'] == test_user) &
    (df_merged['content_id'].isin(publisher_content_scope))
]

liked_items = user_views[user_views['watch_pct'] >= 0.5]['content_id'].tolist()
print(f"\nUser liked {len(liked_items)} items:")
print(liked_items)

# What did different methods recommend?
collab_rec = collab_recs[collab_recs['adventurer_id'] == test_user].iloc[0]
print(f"\nCollaborative recommended: [{collab_rec['rec1']}, {collab_rec['rec2']}]")

heuristic_recs = pd.read_csv(P('heuristic_eval.csv'))
heur_rec = heuristic_recs[heuristic_recs['adventurer_id'] == test_user].iloc[0]
print(f"Heuristic recommended:     [{heur_rec['rec1']}, {heur_rec['rec2']}]")

# Check if collaborative recs are in liked set
print(f"\n2. Are collaborative recs in the liked set?")
print(f"   {collab_rec['rec1']} in liked: {collab_rec['rec1'] in liked_items}")
print(f"   {collab_rec['rec2']} in liked: {collab_rec['rec2'] in liked_items}")

print(f"\n3. Are heuristic recs in the liked set?")
print(f"   {heur_rec['rec1']} in liked: {heur_rec['rec1'] in liked_items}")
print(f"   {heur_rec['rec2']} in liked: {heur_rec['rec2'] in liked_items}")

# Now let's check: did this user even WATCH the collaborative recs?
print(f"\n4. Did user watch the collaborative recommendations at all?")
for rec in [collab_rec['rec1'], collab_rec['rec2']]:
    watched = rec in user_views['content_id'].values
    if watched:
        watch_pct = user_views[user_views['content_id'] == rec]['watch_pct'].iloc[0]
        print(f"   {rec}: YES (watched {watch_pct*100:.1f}%)")
    else:
        print(f"   {rec}: NO (never watched)")

print(f"\n5. Did user watch the heuristic recommendations at all?")
for rec in [heur_rec['rec1'], heur_rec['rec2']]:
    watched = rec in user_views['content_id'].values
    if watched:
        watch_pct = user_views[user_views['content_id'] == rec]['watch_pct'].iloc[0]
        print(f"   {rec}: YES (watched {watch_pct*100:.1f}%)")
    else:
        print(f"   {rec}: NO (never watched)")

# Check overall popularity of these items
print(f"\n6. How popular are these items overall?")
for item_id in [collab_rec['rec1'], collab_rec['rec2'], heur_rec['rec1'], heur_rec['rec2']]:
    view_count = len(views_pub[views_pub['content_id'] == item_id])
    total_views = len(views_pub)
    print(f"   {item_id}: {view_count} views ({view_count/total_views*100:.2f}% of all views)")

# Most popular items
print(f"\n7. What are the MOST popular items?")
popularity = views_pub['content_id'].value_counts().head(10)
for item_id, count in popularity.items():
    in_liked = "✓" if item_id in liked_items else " "
    print(f"   {in_liked} {item_id}: {count} views")

print("\n" + "="*60)
print("HYPOTHESIS")
print("="*60)
print("Your collaborative filter is recommending items that:")
print("1. Are similar to what the user liked")
print("2. But the user hasn't watched yet")
print("")
print("The heuristic recommends the 2 most popular items,")
print("which many users have already watched.")
print("")
print("For your report: This shows collaborative is better for")
print("DISCOVERY, while heuristic is safer but less personalized.")
print("="*60)