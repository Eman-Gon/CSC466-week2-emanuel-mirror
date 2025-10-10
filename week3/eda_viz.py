import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("Generating visualizations...")

# Load data
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))

# Calculate watch percentage
meta_cols = ['content_id', 'minutes']
for col in ['genre_id', 'language_code']:
    if col in df_metadata.columns:
        meta_cols.append(col)

df_merged = df_views.merge(df_metadata[meta_cols], on='content_id', how='left')

if 'minutes' in df_merged.columns:
    denom = (df_merged['minutes'] * 60).replace(0, np.nan)
    df_merged['watch_pct'] = (df_merged['seconds_viewed'] / denom).clip(0, 1)
else:
    df_merged['watch_pct'] = np.nan

# Add age (for Age vs Watch %) if available
if 'adventurer_id' in df_views.columns and 'adventurer_id' in df_adventurers.columns:
    age_cols = ['adventurer_id']
    if 'age' in df_adventurers.columns:
        age_cols.append('age')
    df_merged = df_merged.merge(df_adventurers[age_cols], on='adventurer_id', how='left')

# FIGURE 1: User Activity
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('User Activity Analysis', fontsize=16, fontweight='bold')

user_views = df_views.groupby('adventurer_id').size()
axes[0, 0].hist(user_views, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Views per User')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title(f'Views Distribution (Mean: {user_views.mean():.1f})')
axes[0, 0].axvline(user_views.mean(), linestyle='--', label='Mean')
axes[0, 0].legend()

user_content = df_views.groupby('adventurer_id')['content_id'].nunique()
axes[0, 1].hist(user_content, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Unique Content per User')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title(f'Content Diversity (Mean: {user_content.mean():.1f})')

axes[1, 0].hist(df_merged['watch_pct'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Watch Percentage')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title(f'Watch % (Mean: {df_merged["watch_pct"].mean():.2f})')
axes[1, 0].axvline(0.5, linestyle='--', label='50%')
axes[1, 0].legend()

if 'rating' in df_views.columns:
    rating_counts = df_views['rating'].dropna().value_counts().sort_index()
    axes[1, 1].bar(rating_counts.index, rating_counts.values, alpha=0.7)
    axes[1, 1].set_xlabel('Rating')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Rating Distribution')
else:
    axes[1, 1].text(0.5, 0.5, 'No ratings', ha='center', va='center')

plt.tight_layout()
plt.savefig(P('eda_users.png'), dpi=300)
print("Saved: eda_users.png")

# FIGURE 2: Content Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Content Analysis', fontsize=16, fontweight='bold')

content_views = df_views.groupby('content_id').size()
axes[0, 0].hist(content_views, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Views per Content')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title(f'Content Popularity (Mean: {content_views.mean():.1f})')
axes[0, 0].set_yscale('log')

if 'genre_id' in df_merged.columns:
    genre_counts = df_merged['genre_id'].dropna().value_counts().head(15)
    axes[0, 1].barh(range(len(genre_counts)), genre_counts.values, alpha=0.7)
    axes[0, 1].set_yticks(range(len(genre_counts)))
    axes[0, 1].set_yticklabels(genre_counts.index)
    axes[0, 1].set_xlabel('Views')
    axes[0, 1].set_title('Top 15 Genres')
    axes[0, 1].invert_yaxis()
else:
    axes[0, 1].text(0.5, 0.5, 'No genre_id', ha='center', va='center')

axes[1, 0].hist(df_metadata['minutes'].dropna(), bins=50, alpha=0.7)
axes[1, 0].set_xlabel('Duration (minutes)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title(f'Content Duration (Mean: {df_metadata["minutes"].mean():.1f})')

if 'language_code' in df_merged.columns:
    lang_counts = df_merged['language_code'].dropna().value_counts().head(10)
    axes[1, 1].bar(range(len(lang_counts)), lang_counts.values, alpha=0.7)
    axes[1, 1].set_xticks(range(len(lang_counts)))
    axes[1, 1].set_xticklabels(lang_counts.index, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Views')
    axes[1, 1].set_title('Top 10 Languages')
else:
    axes[1, 1].text(0.5, 0.5, 'No language_code', ha='center', va='center')

plt.tight_layout()
plt.savefig(P('eda_content.png'), dpi=300)
print("Saved: eda_content.png")

# FIGURE 3: Demographics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Demographics', fontsize=16, fontweight='bold')

if 'age' in df_adventurers.columns:
    axes[0, 0].hist(df_adventurers['age'].dropna(), bins=40, alpha=0.7)
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Age Distribution (Mean: {df_adventurers["age"].mean():.1f})')
else:
    axes[0, 0].text(0.5, 0.5, 'No age', ha='center', va='center')

if 'gender' in df_adventurers.columns:
    gender_counts = df_adventurers['gender'].dropna().value_counts()
    axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Gender Distribution')
else:
    axes[0, 1].text(0.5, 0.5, 'No gender', ha='center', va='center')

if 'region' in df_adventurers.columns:
    region_counts = df_adventurers['region'].dropna().value_counts().head(15)
    axes[1, 0].barh(range(len(region_counts)), region_counts.values, alpha=0.7)
    axes[1, 0].set_yticks(range(len(region_counts)))
    axes[1, 0].set_yticklabels(region_counts.index)
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_title('Top 15 Regions')
    axes[1, 0].invert_yaxis()
else:
    axes[1, 0].text(0.5, 0.5, 'No region', ha='center', va='center')

if 'age' in df_merged.columns and 'watch_pct' in df_merged.columns:
    age_watch = df_merged[['age', 'watch_pct']].dropna().groupby('age')['watch_pct'].mean()
    if not age_watch.empty:
        axes[1, 1].scatter(age_watch.index, age_watch.values, alpha=0.6)
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Avg Watch %')
        axes[1, 1].set_title('Age vs Watch Percentage')
    else:
        axes[1, 1].text(0.5, 0.5, 'No age/watch data', ha='center', va='center')
else:
    axes[1, 1].text(0.5, 0.5, 'No age/watch data', ha='center', va='center')

plt.tight_layout()
plt.savefig(P('eda_demographics.png'), dpi=300)
print("Saved: eda_demographics.png")

print("\nAll done!")
