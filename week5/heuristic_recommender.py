import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

# Load data
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))

# Create ordinal dates for temporal calculations
MONTH_ORDER = ["Frostmere", "Emberfall", "Lunaris", "Verdantia", "Solstice",
                "Duskveil", "Starshade", "Aurorath", "Mysthaven", "Eclipsion"]
MONTH_TO_INDEX = {m: i for i, m in enumerate(MONTH_ORDER)}

def to_ordinal(row):
    month_idx = MONTH_TO_INDEX.get(row['month'], 0)
    day = row.get('day_of_month', 1) if 'day_of_month' in row.index else 1
    return row['year'] * 240 + month_idx * 24 + (day - 1)

# Calculate view ordinals
print("Calculating temporal ordinals...")
df_views['view_ordinal'] = df_views.apply(to_ordinal, axis=1)
print(f"View ordinal range: {df_views['view_ordinal'].min()} to {df_views['view_ordinal'].max()}")

# Simple trending recommender
def recommend_trending(user_id, n_recs=2):
    """Global heuristic: time-decayed popularity by language"""
    
    # Get user's language (default to Reptilian if not found)
    user_lang = 'RP'  # Default for publisher wn32
    if user_id in df_adventurers['adventurer_id'].values:
        user_lang = df_adventurers.loc[
            df_adventurers['adventurer_id']==user_id, 
            'primary_language'
        ].iloc[0]
    
    # Recent views (last 60 days)
    recent_cutoff = df_views['view_ordinal'].max() - 60
    recent = df_views[df_views['view_ordinal'] > recent_cutoff]
    
    # If too few recent views, expand window
    if len(recent) < 100:
        recent_cutoff = df_views['view_ordinal'].max() - 120
        recent = df_views[df_views['view_ordinal'] > recent_cutoff]
    
    # Filter by language
    recent_meta = recent.merge(
        df_metadata[['content_id', 'language_code']], 
        on='content_id'
    )
    recent_meta = recent_meta[recent_meta['language_code'] == user_lang]
    
    # If no content in user's language, use all languages
    if len(recent_meta) == 0:
        recent_meta = recent.merge(
            df_metadata[['content_id', 'language_code']], 
            on='content_id'
        )
    
    # Count views as popularity score
    trending = recent_meta['content_id'].value_counts()
    
    # Return top N
    if len(trending) >= n_recs:
        return trending.head(n_recs).index.tolist()
    else:
        # Fallback to overall most popular
        overall_popular = df_views['content_id'].value_counts()
        return overall_popular.head(n_recs).index.tolist()

# Test it
if __name__ == "__main__":
    print("\nTesting trending recommender...")
    test_users = ['4uds', '4jyy', '52st', 'tegt', 'do8o']
    
    for uid in test_users:
        try:
            recs = recommend_trending(uid, n_recs=2)
            print(f"{uid}: {recs}")
        except Exception as e:
            print(f"{uid}: Error - {e}")
    
    print("\n✓ Heuristic recommender working!")