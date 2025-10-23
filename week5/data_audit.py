import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("WEEK 3: DATA QUALITY AUDIT")

print("\nLoading datasets...")
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))
df_cancels = pd.read_parquet(P("cancellations.parquet"))

datasets = {
    "content_views": df_views,
    "content_metadata": df_metadata,
    "adventurer_metadata": df_adventurers,
    "subscriptions": df_subs,
    "cancellations": df_cancels
}

print("\n1. MISSING VALUES")
for name, df in datasets.items():
    print(f"\n{name}:")
    print(f"  Total rows: {len(df):,}")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col in missing[missing > 0].index:
            pct = missing[col] / len(df) * 100
            print(f"  - {col}: {missing[col]:,} ({pct:.1f}%)")
    else:
        print("  No missing values")

print("\n2. DUPLICATE RECORDS")
for name, df in datasets.items():
    total_dups = df.duplicated().sum()
    print(f"\n{name}: {total_dups:,} duplicates ({total_dups/len(df)*100:.2f}%)")
    if name == "content_views":
        user_content_dups = df.duplicated(subset=['adventurer_id', 'content_id']).sum()
        print(f"  Duplicate (user, content) pairs: {user_content_dups:,}")

print("\n3. DATA RANGES AND ANOMALIES")
if 'age' in df_adventurers.columns:
    print(f"\nAge:")
    print(f"  Min: {df_adventurers['age'].min()}")
    print(f"  Max: {df_adventurers['age'].max()}")
    print(f"  Mean: {df_adventurers['age'].mean():.1f}")

df_merged = df_views.merge(df_metadata[['content_id', 'minutes']], on='content_id', how='left')
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / (df_merged['minutes'] * 60)).clip(0, None)

print(f"\nWatch Percentage:")
print(f"  Min: {df_merged['watch_pct'].min():.2f}")
print(f"  Max: {df_merged['watch_pct'].max():.2f}")
print(f"  Mean: {df_merged['watch_pct'].mean():.2f}")
over_100 = (df_merged['watch_pct'] > 1.0).sum()
if over_100 > 0:
    print(f"  {over_100:,} views with watch_pct greater than 100 percent")

if 'rating' in df_views.columns:
    print(f"\nRatings:")
    print(f"  Coverage: {df_views['rating'].notna().sum():,} / {len(df_views):,} ({df_views['rating'].notna().sum()/len(df_views)*100:.1f}%)")

print("\n4. CATEGORICAL DISTRIBUTIONS")
if 'gender' in df_adventurers.columns:
    print(f"\nGender: {df_adventurers['gender'].value_counts().to_dict()}")

if 'genre_id' in df_metadata.columns:
    print(f"\nTop 10 Genres:")
    for genre, count in df_metadata['genre_id'].value_counts().head(10).items():
        print(f"  {genre}: {count}")

print("\n5. CONSTRAINT VIOLATIONS AND SEMANTIC ISSUES")
over_length = (df_merged['seconds_viewed'] > df_merged['minutes'] * 60).sum()
print(f"\nViews longer than content: {over_length:,}")
if over_length > 0:
    print("  Users watched more than content duration")

MONTH_ORDER = ["Frostmere", "Emberfall", "Lunaris", "Verdantia", "Solstice",
                "Duskveil", "Starshade", "Aurorath", "Mysthaven", "Eclipsion"]
MONTH_TO_INDEX = {m: i for i, m in enumerate(MONTH_ORDER)}

def to_ordinal(row):
    month_idx = MONTH_TO_INDEX.get(row['month'], 0)
    return row['year'] * 240 + month_idx * 24 + (row.get('day_of_month', 1) - 1)

df_views['view_ordinal'] = df_views.apply(to_ordinal, axis=1)
df_metadata['release_ordinal'] = df_metadata.apply(to_ordinal, axis=1)
df_check = df_views.merge(df_metadata[['content_id', 'release_ordinal']], on='content_id', how='left')
time_violations = (df_check['view_ordinal'] < df_check['release_ordinal']).sum()
print(f"\nViews before content created: {time_violations:,}")
if time_violations == 0:
    print("  All temporal constraints satisfied")

if 'age' in df_adventurers.columns:
    extreme_ages = (df_adventurers['age'] > 1000).sum()
    print(f"\nAdventurers over 1000 years old: {extreme_ages:,}")
    if extreme_ages > 0:
        by_region = df_adventurers[df_adventurers['age'] > 1000].groupby('region')['age'].agg(['count', 'mean'])
        print("  Distribution by region:")
        print(by_region)

print(f"\nSemantic type checks:")
print(f"  adventurer_id type: {df_adventurers['adventurer_id'].dtype}")
print(f"  content_id type: {df_metadata['content_id'].dtype}")
if df_adventurers['adventurer_id'].dtype == 'object':
    print("  IDs are strings, correct for categorical data")
