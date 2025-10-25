import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax() 

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())

views_pub = df_views[df_views["adventurer_id"].isin(sub_ids)]
PUBLISHER_CONTENT_SCOPE = set(views_pub['content_id'].unique())

print(f"Publisher: {publisher_id}")
print(f"Content scope: {len(PUBLISHER_CONTENT_SCOPE)} items")
print(f"Heuristic will ONLY recommend from these {len(PUBLISHER_CONTENT_SCOPE)} items\n")

MONTH_ORDER = ["Frostmere", "Emberfall", "Lunaris", "Verdantia", "Solstice",
               "Duskveil", "Starshade", "Aurorath", "Mysthaven", "Eclipsion"]
MONTH_TO_INDEX = {m: i for i, m in enumerate(MONTH_ORDER)}

def to_ordinal(row):
    month_idx = MONTH_TO_INDEX.get(row['month'], 0)
    day = row.get('day_of_month', 1) if 'day_of_month' in row.index else 1
    return row['year'] * 240 + month_idx * 24 + (day - 1)

print("Temporal ")
views_pub_df = views_pub.copy()
views_pub_df['view_ordinal'] = views_pub_df.apply(to_ordinal, axis=1)
print(f"View ordinal range: {views_pub_df['view_ordinal'].min()} to {views_pub_df['view_ordinal'].max()}")

def recommend_trending(user_id, n_recs=2):
    """
    Global heuristic: time-decayed popularity by language
    SCOPED TO PUBLISHER'S CONTENT ONLY
    """
    user_lang = None
    if user_id in set(df_adventurers['adventurer_id'].values):
        user_lang = df_adventurers.loc[
            df_adventurers['adventurer_id'] == user_id, 'primary_language'
        ].iloc[0]

    recent_cutoff = views_pub_df['view_ordinal'].max() - 60
    recent = views_pub_df[
        (views_pub_df['view_ordinal'] > recent_cutoff) &
        (views_pub_df['content_id'].isin(PUBLISHER_CONTENT_SCOPE))
    ]
    if len(recent) < 100:
        recent_cutoff = views_pub_df['view_ordinal'].max() - 120
        recent = views_pub_df[
            (views_pub_df['view_ordinal'] > recent_cutoff) &
            (views_pub_df['content_id'].isin(PUBLISHER_CONTENT_SCOPE))
        ]

    if user_lang is not None:
        recent_meta = recent.merge(
            df_metadata[['content_id', 'language_code']],
            on='content_id',
            how='left'
        )
        recent_lang = recent_meta[recent_meta['language_code'] == user_lang]
        if len(recent_lang) > 0:
            recent = recent_lang

    trending = recent['content_id'].value_counts()

    if len(trending) >= n_recs:
        return trending.head(n_recs).index.tolist()
    else:
        overall_popular = views_pub_df[
            views_pub_df['content_id'].isin(PUBLISHER_CONTENT_SCOPE)
        ]['content_id'].value_counts()

        if len(overall_popular) >= n_recs:
            return overall_popular.head(n_recs).index.tolist()
        else:
            return overall_popular.index.tolist()

if __name__ == "__main__":
    print("\nTesting trending recommender")
    test_users = ['4uds', '4jyy', '52st', 'tegt', 'do8o']

    for uid in test_users:
        try:
            recs = recommend_trending(uid, n_recs=2)
            print(f"{uid}: {recs}")
            for rec in recs:
                if rec not in PUBLISHER_CONTENT_SCOPE:
                    print(f"WARNING: {rec} is NOT in publisher scope!")
        except Exception as e:
            print(f"{uid}: Error - {e}")

    print("\nHeuristic recommender.")
    print(f"All recommendations scoped to {len(PUBLISHER_CONTENT_SCOPE)} items.")
