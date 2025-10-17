import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("DEEP INVESTIGATION - PROFESSOR'S HINTS")
print("="*60)

# Load data
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))

print("\n1. THE EGG MYSTERY - Checking for patterns...")
print(f"Unique adventurers: {df_adventurers['adventurer_id'].nunique():,}")
print(f"Unique content: {df_metadata['content_id'].nunique():,}")
print(f"Unique publishers: {df_views['publisher_id'].nunique():,}")

print("\nFirst 10 content IDs (sorted):")
first_10 = sorted(df_metadata['content_id'].unique())[:10]
for cid in first_10:
    print(f"  {cid}")

print("\nLast 10 content IDs (sorted):")
last_10 = sorted(df_metadata['content_id'].unique())[-10:]
for cid in last_10:
    print(f"  {cid}")

# Check for round numbers or special counts
total_content = df_metadata['content_id'].nunique()
total_adventurers = df_adventurers['adventurer_id'].nunique()
total_publishers = df_views['publisher_id'].nunique()

print(f"\n🥚 EGG CLUES:")
print(f"Total content items: {total_content}")
print(f"Total adventurers: {total_adventurers}")
print(f"Total publishers: {total_publishers}")
print(f"\nContent / Publishers = {total_content / total_publishers:.1f}")
print(f"Adventurers / Publishers = {total_adventurers / total_publishers:.1f}")

# Check for patterns in counts
print(f"\n🔍 PATTERN ANALYSIS:")
print(f"982 content items... interesting number!")
print(f"26 publishers... 26 letters in alphabet?")
print(f"25,770 adventurers...")

# Check content metadata columns
print(f"\nContent metadata columns: {list(df_metadata.columns)}")
print(f"Adventurer metadata columns: {list(df_adventurers.columns)}")

print("\n2. TEMPORAL BIMODALITY - Day/Month patterns...")

# Calculate watch percentage for ALL views
df_merged = df_views.merge(df_metadata[['content_id', 'minutes']], on='content_id', how='left')
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / (df_merged['minutes'] * 60)).clip(0, 1)

# Since no day_of_week, check by month
if 'month' in df_merged.columns:
    month_stats = df_merged.groupby('month')['watch_pct'].agg(['mean', 'median', 'count'])
    print("\nWatch % by Month:")
    print(month_stats)

# Check by day_of_month
if 'day_of_month' in df_merged.columns:
    day_stats = df_merged.groupby('day_of_month')['watch_pct'].agg(['mean', 'count']).sort_index()
    print("\nWatch % by Day of Month (first 10 days):")
    print(day_stats.head(10))

print("\n3. TIMESTAMP VALIDATION - Checking for impossible dates...")

# Create ordinal dates
MONTH_ORDER = ["Frostmere", "Emberfall", "Lunaris", "Verdantia", "Solstice",
                "Duskveil", "Starshade", "Aurorath", "Mysthaven", "Eclipsion"]
MONTH_TO_INDEX = {m: i for i, m in enumerate(MONTH_ORDER)}

def to_ordinal(row):
    month_idx = MONTH_TO_INDEX.get(row['month'], 0)
    return row['year'] * 240 + month_idx * 24 + (row.get('day_of_month', 1) - 1)

# Views dates
df_views['view_ordinal'] = df_views.apply(to_ordinal, axis=1)

# Content release dates
df_metadata['release_ordinal'] = df_metadata.apply(to_ordinal, axis=1)

# Merge and check
df_check = df_views.merge(
    df_metadata[['content_id', 'release_ordinal']], 
    on='content_id', 
    how='left'
)

impossible = df_check[df_check['view_ordinal'] < df_check['release_ordinal']]
print(f"\nViews BEFORE content was created: {len(impossible):,}")
if len(impossible) > 0:
    print("⚠️  WARNING: Time travel detected!")
    print(impossible[['adventurer_id', 'content_id', 'view_ordinal', 'release_ordinal']].head(10))
else:
    print("✓ All timestamps are valid")

# Check for date anomalies
print("\nChecking for date anomalies...")
date_range = df_views['view_ordinal'].max() - df_views['view_ordinal'].min()
print(f"View date range: {date_range} days")
print(f"Min view date: Year {df_views['year'].min()}, {df_views['month'].min()}")
print(f"Max view date: Year {df_views['year'].max()}, {df_views['month'].max()}")

print("\n4. WATCH PERCENTAGE DISTRIBUTION - Looking for bimodality...")

watch_pct_clean = df_merged['watch_pct'].dropna()

# Bin the watch percentages
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
watch_dist = pd.cut(watch_pct_clean, bins=bins).value_counts().sort_index()

print("\nWatch % Distribution:")
for interval, count in watch_dist.items():
    pct = count / len(watch_pct_clean) * 100
    bar = "█" * int(pct / 2)
    print(f"{interval}: {count:6,} ({pct:5.1f}%) {bar}")

# Check for bimodality
low_watch = (watch_pct_clean < 0.3).sum()
high_watch = (watch_pct_clean > 0.7).sum()
mid_watch = ((watch_pct_clean >= 0.3) & (watch_pct_clean <= 0.7)).sum()

print(f"\n📈 BIMODALITY CHECK:")
print(f"Low engagement (<30%): {low_watch:,} ({low_watch/len(watch_pct_clean)*100:.1f}%)")
print(f"Mid engagement (30-70%): {mid_watch:,} ({mid_watch/len(watch_pct_clean)*100:.1f}%)")
print(f"High engagement (>70%): {high_watch:,} ({high_watch/len(watch_pct_clean)*100:.1f}%)")

if low_watch > high_watch * 1.5 or high_watch > low_watch * 1.5:
    print("⚠️  BIMODAL PATTERN DETECTED! Like the Snapchat flash/no-flash example!")

print("\n5. GENRE & LANGUAGE PATTERNS...")

# Need to merge to get genre_id
meta_cols_needed = ['content_id']
if 'genre_id' in df_metadata.columns:
    meta_cols_needed.append('genre_id')
if 'language_code' in df_metadata.columns:
    meta_cols_needed.append('language_code')

df_with_meta = df_views.merge(df_metadata[meta_cols_needed], on='content_id', how='left')

# Genre analysis
if 'genre_id' in df_with_meta.columns:
    print("\nTop 10 Genres by view count:")
    genre_views = df_with_meta['genre_id'].value_counts().head(10)
    for genre, count in genre_views.items():
        print(f"  {genre}: {count:,}")
else:
    print("\nNo genre_id column found")

# Language analysis
if 'language_code' in df_with_meta.columns:
    print("\nLanguage distribution in views:")
    lang_views = df_with_meta['language_code'].value_counts()
    for lang, count in lang_views.items():
        print(f"  {lang}: {count:,}")
else:
    print("\nNo language_code found")

print("\n6. MODEL LEARNING CHECK - What patterns correlate?")

# If you have your recommendations, load them
import os
if os.path.exists(P("recommendations.csv")):
    recs = pd.read_csv(P("recommendations.csv"))
    rec_content_ids = []
    for col in recs.columns:
        if col.startswith('rec'):
            rec_content_ids.extend(recs[col].dropna().unique())
    
    print(f"\nTotal recommended items: {len(set(rec_content_ids))}")
    
    # What are the characteristics of recommended content?
    rec_meta = df_metadata[df_metadata['content_id'].isin(rec_content_ids)]
    
    if 'genre_id' in rec_meta.columns:
        print("\nTop genres in recommendations:")
        print(rec_meta['genre_id'].value_counts().head())
    
    if 'language_code' in rec_meta.columns:
        print("\nLanguages in recommendations:")
        print(rec_meta['language_code'].value_counts())
else:
    print("\n⚠️  No recommendations.csv found - run recommender.py first")

print("\n" + "="*60)
print("🥚 EGG MYSTERY SUMMARY")
print("="*60)
print(f"Key numbers found:")
print(f"  - 982 content items (not a round number)")
print(f"  - 26 publishers (= 26 letters!)")
print(f"  - 25,770 adventurers")
print(f"  - ~991 adventurers per publisher")
print(f"  - ~38 content items per publisher")
print("\nProfessor said: 'How many eggs did we start with?'")
print("Could the answer be hidden in these ratios? 🤔")
print("="*60)