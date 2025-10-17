import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name


df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

print(f"\nDataset sizes:")
print(f"  Views: {len(df_views):,}")
print(f"  Content: {len(df_metadata):,}")
print(f"  Adventurers: {len(df_adventurers):,}")

df_views = df_views.sort_values('seconds_viewed', ascending=False)\
    .drop_duplicates(subset=['adventurer_id', 'content_id'], keep='first')


df_merged = df_views.merge(
    df_metadata[['content_id', 'minutes']], 
    on='content_id', 
    how='left'
)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / (df_merged['minutes'] * 60)).clip(0, 1)


df_views_clean = df_merged[
    (df_merged['watch_pct'].fillna(0) >= 0.05) | 
    (df_merged['seconds_viewed'] >= 30)
].copy()

print(f"After cleaning: {len(df_views_clean):,} views")

pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"Selected publisher: {publisher_id} ({pub_counts.max():,} subscribers)")

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())

publisher_content = df_subs[df_subs['publisher_id'] == publisher_id]['content_id'].unique() \
    if 'content_id' in df_subs.columns else df_views_clean['content_id'].unique()

views_pub = df_views_clean[
    (df_views_clean["adventurer_id"].isin(sub_ids))
].copy()

print(f"\nPublisher data:")
print(f"  Views: {len(views_pub):,}")
print(f"  Users: {views_pub['adventurer_id'].nunique():,}")
print(f"  Content: {views_pub['content_id'].nunique():,}")

content_items = df_metadata[
    df_metadata['content_id'].isin(views_pub['content_id'].unique())
].copy()

print(f"\nContent items: {len(content_items):,}")
print(f"Available columns: {content_items.columns.tolist()}")


scaler = StandardScaler()
content_items['duration_scaled'] = scaler.fit_transform(content_items[['minutes']])
print(f"Duration scaled: mean={content_items['duration_scaled'].mean():.2f}, "
      f"std={content_items['duration_scaled'].std():.2f}")


genre_dummies = pd.DataFrame()
if 'genre_id' in content_items.columns:
    genre_dummies = pd.get_dummies(content_items['genre_id'], prefix='genre')
    print(f"Genre encoding: {len(genre_dummies.columns)} categories")

lang_dummies = pd.DataFrame()
if 'language_code' in content_items.columns:
    lang_dummies = pd.get_dummies(content_items['language_code'], prefix='lang')
    print(f"Language encoding: {len(lang_dummies.columns)} categories")


tfidf_features = pd.DataFrame()
if 'title' in content_items.columns and 'description' in content_items.columns:
    content_items['text_corpus'] = (
        content_items['title'].fillna('') + ' ' + 
        content_items['description'].fillna('')
    )
    
    tfidf = TfidfVectorizer(
        max_features=100,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    tfidf_matrix = tfidf.fit_transform(content_items['text_corpus'])
    tfidf_features = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=content_items.index,
        columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    )
    print(f"TF-IDF: {tfidf_matrix.shape[1]} features")
else:
    print("No text columns available - skipping TF-IDF")

feature_matrix = content_items[['content_id', 'duration_scaled']].set_index('content_id')

if not genre_dummies.empty:
    genre_dummies.index = content_items['content_id'].values
    feature_matrix = feature_matrix.join(genre_dummies, how='left')

if not lang_dummies.empty:
    lang_dummies.index = content_items['content_id'].values
    feature_matrix = feature_matrix.join(lang_dummies, how='left')

if not tfidf_features.empty:
    tfidf_features.index = content_items['content_id'].values
    feature_matrix = feature_matrix.join(tfidf_features, how='left')

feature_matrix = feature_matrix.fillna(0)
print(f"Final feature matrix: {feature_matrix.shape}")


user_item = views_pub.groupby(['adventurer_id', 'content_id'])['watch_pct']\
    .max().unstack(fill_value=0).astype(np.float32)

common_items = user_item.columns.intersection(feature_matrix.index)
feature_matrix = feature_matrix.loc[common_items]
user_item = user_item[common_items]

print(f"Aligned items: {len(common_items)}")

print("Computing similarity matrices...")
item_collab_sim = cosine_similarity(user_item.T.values)
item_content_sim = cosine_similarity(feature_matrix.values)

ALPHA = 0.6
BETA = 0.4
item_hybrid_sim = ALPHA * item_collab_sim + BETA * item_content_sim

print(f"Hybrid weights: {ALPHA} collaborative, {BETA} content-based")

def recommend_hybrid(user_id, n_recs=10):
    """Generate recommendations using hybrid similarity"""
    if user_id not in user_item.index:
        return []
    
    user_profile = user_item.loc[user_id].values
    seen_idx = np.where(user_profile > 0)[0]
    
    if len(seen_idx) == 0:
        return []
    
    user_weights = user_profile[seen_idx]
    scores = (item_hybrid_sim[seen_idx].T * user_weights).sum(axis=1)
    scores[seen_idx] = -np.inf
    
    top_idx = np.argsort(-scores)[:n_recs]
    return [user_item.columns[i] for i in top_idx if np.isfinite(scores[i])]

def recommend_baseline(user_id, n_recs=2):
    """Baseline collaborative filtering"""
    if user_id not in user_item.index:
        return []
    user_profile = user_item.loc[user_id].values
    seen_idx = np.where(user_profile > 0)[0]
    if len(seen_idx) == 0:
        return []
    scores = (item_collab_sim[seen_idx].T * user_profile[seen_idx]).sum(axis=1)
    scores[seen_idx] = -np.inf
    top_idx = np.argsort(-scores)[:n_recs]
    return [user_item.columns[i] for i in top_idx if np.isfinite(scores[i])]

user_activity = user_item.sum(axis=1).sort_values(ascending=False)
eval_users = user_activity.index[:9].tolist()

print(f"\nSelected {len(eval_users)} users")

pre_eval = []
for uid in eval_users:
    recs = recommend_baseline(uid, n_recs=2)
    if len(recs) >= 2:
        pre_eval.append({'adventurer_id': uid, 'rec1': recs[0], 'rec2': recs[1]})

pd.DataFrame(pre_eval).to_csv(P('pre_eval.csv'), index=False)
print(f"✓ Saved pre_eval.csv ({len(pre_eval)} users)")

post_eval = []
for uid in eval_users:
    recs = recommend_hybrid(uid, n_recs=2)
    if len(recs) >= 2:
        post_eval.append({'adventurer_id': uid, 'rec1': recs[0], 'rec2': recs[1]})

pd.DataFrame(post_eval).to_csv(P('post_eval.csv'), index=False)
print(f"✓ Saved post_eval.csv ({len(post_eval)} users)")

print("COMPLETE! Files generated:")
print("  - pre_eval.csv")
print("  - post_eval.csv")