"""
Week 7: User Personas via Behavioral Clustering
Steven Gonzalez
CSC-466 Fall 2025

APPROACH - AVOIDING SELF-REFERENTIAL LOOPS:
1. Cluster on BEHAVIORAL PATTERNS (HOW users engage)
   - Completion consistency, genre diversity, loyalty patterns
2. Describe with CONTEXT FEATURES (WHAT they consume, volume)
   - Number of views, content preferences, demographics

This separates clustering features from description features.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name
OUTPUT = lambda name: ROOT / name

def calculate_gini(values):
    """Calculate Gini coefficient for concentration measurement"""
    if len(values) == 0 or values.sum() == 0:
        return 0
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

print("WEEK 7: USER PERSONAS VIA BEHAVIORAL CLUSTERING")
print("=" * 70)
print("\nAPPROACH - AVOIDING SELF-REFERENTIAL LOOPS:")
print("  1. Cluster on BEHAVIORAL PATTERNS (HOW users engage)")
print("     - Completion consistency, genre diversity, loyalty patterns")
print("  2. Describe with CONTEXT FEATURES (WHAT they consume, volume)")
print("     - Number of views, content preferences, demographics")
print("=" * 70)

# ============================================================
# PART 1: LOAD AND CLEAN DATA
# ============================================================
print("\n[1] Loading Data")

df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))
df_cancels = pd.read_parquet(P("cancellations.parquet"))

print(f"    Views: {len(df_views):,}")
print(f"    Adventurers: {len(df_adventurers):,}")
print(f"    Content: {len(df_metadata):,}")

# Clean data
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
][['adventurer_id', 'content_id', 'seconds_viewed', 'watch_pct', 'minutes']].copy()

print(f"    Clean views: {len(df_views_clean):,}")

# ============================================================
# PART 2: BUILD ALL FEATURES (CLUSTERING + DESCRIPTION)
# ============================================================
print("\n[2] Building User Features")

# Start with adventurer base data
user_profiles = df_adventurers[['adventurer_id', 'age']].copy()

# Enrich views with metadata
df_views_enriched = df_views_clean.merge(
    df_metadata[['content_id', 'genre_id', 'language_code', 'minutes']],  # <-- no publisher_id
    on='content_id',
    how='left'
)

# === BEHAVIORAL FEATURES (for clustering) ===
print("\n    Building BEHAVIORAL features (for clustering):")

# 1. Completion behavior patterns
print("      ✓ Completion consistency")
completion_stats = df_views_enriched.groupby('adventurer_id')['watch_pct'].agg(['mean', 'std', 'count']).reset_index()
completion_stats.columns = ['adventurer_id', 'avg_completion_rate', 'completion_std', 'num_views']
completion_stats['completion_consistency'] = 1 / (1 + completion_stats['completion_std'].fillna(0))

# Finisher ratio (% of content watched >80%)
finisher_counts = df_views_enriched[df_views_enriched['watch_pct'] >= 0.8].groupby('adventurer_id').size()
completion_stats['finisher_ratio'] = (finisher_counts / completion_stats['num_views']).fillna(0)

# 2. Genre diversity (concentration)
print("      ✓ Genre concentration")
genre_concentration = []
for adv_id in df_views_enriched['adventurer_id'].unique():
    user_views = df_views_enriched[df_views_enriched['adventurer_id'] == adv_id]
    genre_counts = user_views['genre_id'].value_counts().values
    gini = calculate_gini(genre_counts)
    genre_concentration.append({'adventurer_id': adv_id, 'genre_concentration': gini})
genre_df = pd.DataFrame(genre_concentration)

# 3. Publisher loyalty
print("      ✓ Publisher concentration (from subscriptions)")
publisher_diversity = df_subs.groupby('adventurer_id')['publisher_id'].nunique().reset_index()
publisher_diversity.columns = ['adventurer_id', 'num_unique_publishers']
publisher_diversity['publisher_concentration'] = 1 / (1 + publisher_diversity['num_unique_publishers'])

# 4. Viewing pattern consistency
print("      ✓ Viewing time consistency")
viewing_stats = df_views_enriched.groupby('adventurer_id')['seconds_viewed'].agg(['mean', 'std']).reset_index()
viewing_stats.columns = ['adventurer_id', 'avg_viewing_time', 'viewing_time_std']
viewing_stats['viewing_consistency'] = viewing_stats['viewing_time_std'] / (viewing_stats['avg_viewing_time'] + 1)

# 5. Subscription churn behavior
print("      ✓ Churn rate")
sub_counts = df_subs.groupby('adventurer_id').size().reset_index(name='num_subs')
churn_counts = df_cancels.groupby('adventurer_id').size().reset_index(name='num_churns')
churn_data = sub_counts.merge(churn_counts, on='adventurer_id', how='left').fillna(0)
churn_data['churn_rate'] = (churn_data['num_churns'] / churn_data['num_subs']).clip(0, 1).fillna(0)

# 6. Content length preference
print("      ✓ Content length preference")
content_length = df_views_enriched.groupby('adventurer_id')['minutes'].mean().reset_index()
content_length.columns = ['adventurer_id', 'avg_content_length']

# 7. Language diversity
print("      ✓ Language diversity")
lang_diversity = df_views_enriched.groupby('adventurer_id')['language_code'].nunique().reset_index()
lang_diversity.columns = ['adventurer_id', 'num_languages']
lang_diversity['language_concentration'] = 1 / (1 + lang_diversity['num_languages'])

# === CONTEXT FEATURES (for description only) ===
print("\n    Building CONTEXT features (for description):")

# View volume
print("      ✓ View counts")
view_counts = df_views_enriched.groupby('adventurer_id').size().reset_index(name='total_views')
unique_content = df_views_enriched.groupby('adventurer_id')['content_id'].nunique().reset_index(name='unique_content_viewed')

# Most watched genre
print("      ✓ Favorite genre")
favorite_genre = df_views_enriched.groupby('adventurer_id')['genre_id'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else -1).reset_index()
favorite_genre.columns = ['adventurer_id', 'favorite_genre_id']

# Total watch time
print("      ✓ Total watch time")
total_time = df_views_enriched.groupby('adventurer_id')['seconds_viewed'].sum().reset_index(name='total_watch_seconds')

# Subscription counts
print("      ✓ Subscription counts")
sub_publisher_counts = df_subs.groupby('adventurer_id')['publisher_id'].nunique().reset_index(name='num_sub_publishers')

# ============================================================
# PART 3: MERGE FEATURES
# ============================================================
print("\n[3] Merging All Features")

# Merge behavioral features
for df in [completion_stats, genre_df, publisher_diversity, viewing_stats, 
           churn_data[['adventurer_id', 'churn_rate', 'num_subs', 'num_churns']], 
           content_length, lang_diversity]:
    user_profiles = user_profiles.merge(df, on='adventurer_id', how='left')

# Merge context features
for df in [view_counts, unique_content, favorite_genre, total_time, sub_publisher_counts]:
    user_profiles = user_profiles.merge(df, on='adventurer_id', how='left')

# Fill missing values
user_profiles = user_profiles.fillna(0)

# Add engagement flag (for filtering)
user_profiles['is_active'] = (user_profiles['num_views'] > 0).astype(int)

print(f"    Total features: {user_profiles.shape[1]}")
print(f"    Total users: {len(user_profiles):,}")

# Filter to active users only (optional but recommended)
user_profiles_active = user_profiles[user_profiles['is_active'] == 1].copy()
print(f"    Active users (for clustering): {len(user_profiles_active):,}")

# ============================================================
# PART 4: SELECT CLUSTERING FEATURES
# ============================================================
print("\n[4] Selecting Features for Clustering")
print("    Using BEHAVIORAL features only (HOW users engage):\n")

# CRITICAL: These features describe HOW users behave, not WHAT/HOW MUCH they consume
clustering_features = [
    'completion_consistency',      # HOW predictable their completion is
    'finisher_ratio',             # HOW often they finish content
    'genre_concentration',        # HOW focused their genre preferences are
    'publisher_concentration',    # HOW loyal to specific publishers
    'viewing_consistency',        # HOW consistent their viewing duration is
    'churn_rate',                # HOW often they cancel subscriptions
    'avg_content_length',        # WHAT length content they prefer
    'language_concentration',    # HOW focused on specific languages
]

for feat in clustering_features:
    print(f"      - {feat}")

X = user_profiles_active[clustering_features].copy()
print(f"\n    Feature matrix: {X.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"    ✓ Features scaled (mean=0, std=1)")

# ============================================================
# PART 5: FIND OPTIMAL K
# ============================================================
print("\n[5] Finding Optimal Number of Clusters")

results = []
K_range = range(3, 11)

print("\nTesting k values:")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    inertia = kmeans.inertia_
    
    results.append({
        'k': k,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski': calinski,
        'inertia': inertia
    })
    
    print(f"  k={k}: Silhouette={silhouette:.3f}, DB={davies_bouldin:.3f}, CH={calinski:.0f}")

results_df = pd.DataFrame(results)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(results_df['k'], results_df['inertia'], 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Inertia', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Elbow Plot', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(results_df['k'], results_df['silhouette'], 'ro-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Silhouette Score (Higher = Better)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(results_df['k'], results_df['davies_bouldin'], 'go-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Davies-Bouldin Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Davies-Bouldin Score (Lower = Better)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(results_df['k'], results_df['calinski'], 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Calinski-Harabasz Score (Higher = Better)', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT('cluster_evaluation_behavioral.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved cluster_evaluation_behavioral.png")

# Select optimal k
optimal_k = int(results_df.loc[results_df['silhouette'].idxmax(), 'k'])
best_silhouette = results_df['silhouette'].max()
print(f"\n{'='*70}")
print(f"OPTIMAL K = {optimal_k} (Silhouette: {best_silhouette:.3f})")
print(f"{'='*70}")

# ============================================================
# PART 6: FINAL CLUSTERING
# ============================================================
print(f"\n[6] Clustering with k={optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=300)
user_profiles_active['cluster'] = kmeans_final.fit_predict(X_scaled)

print("\nCluster distribution:")
for cluster_id in sorted(user_profiles_active['cluster'].unique()):
    count = (user_profiles_active['cluster'] == cluster_id).sum()
    pct = count / len(user_profiles_active) * 100
    print(f"  Cluster {cluster_id}: {count:,} users ({pct:.1f}%)")

# ============================================================
# PART 7: DESCRIBE CLUSTERS
# ============================================================
print("\n[7] Describing Clusters")

# Description features: behavioral patterns + context
description_features = [
    # Behavioral (used in clustering)
    'completion_consistency', 'finisher_ratio', 'genre_concentration',
    'publisher_concentration', 'viewing_consistency', 'churn_rate',
    'avg_content_length', 'language_concentration',
    # Context (NOT used in clustering)
    'total_views', 'unique_content_viewed', 'total_watch_seconds',
    'num_subs', 'num_sub_publishers', 'age'
]

cluster_summary = user_profiles_active.groupby('cluster')[description_features].mean()

print("\nCluster Summary (mean values):")
print(cluster_summary.round(3))

cluster_summary.to_csv(OUTPUT('cluster_summary_behavioral.csv'))
print(f"\n✓ Saved cluster_summary_behavioral.csv")

# ============================================================
# PART 8: VISUALIZATION
# ============================================================
print("\n[8] Creating Visualizations")

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=user_profiles_active['cluster'], 
                     cmap='tab10', 
                     alpha=0.6, 
                     s=50,
                     edgecolors='black',
                     linewidth=0.5)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
          fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
          fontsize=12, fontweight='bold')
plt.title(f'User Personas - Behavioral Clustering (k={optimal_k})', 
         fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT('cluster_visualization_behavioral.png'), dpi=300, bbox_inches='tight')
print("✓ Saved cluster_visualization_behavioral.png")

# ============================================================
# PART 9: CREATE PERSONA INSIGHTS
# ============================================================
print("\n[9] Creating Persona Insights")

persona_insights = []

for cluster_id in sorted(user_profiles_active['cluster'].unique()):
    cluster_data = user_profiles_active[user_profiles_active['cluster'] == cluster_id]
    
    insight = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        # Behavioral characteristics (used in clustering)
        'completion_consistency': cluster_data['completion_consistency'].mean(),
        'finisher_ratio': cluster_data['finisher_ratio'].mean(),
        'genre_concentration': cluster_data['genre_concentration'].mean(),
        'churn_rate': cluster_data['churn_rate'].mean(),
        'viewing_consistency': cluster_data['viewing_consistency'].mean(),
        'publisher_concentration': cluster_data['publisher_concentration'].mean(),
        'avg_content_length': cluster_data['avg_content_length'].mean(),
        # Context (for interpretation)
        'avg_views': cluster_data['total_views'].mean(),
        'avg_unique_content': cluster_data['unique_content_viewed'].mean(),
        'avg_watch_hours': cluster_data['total_watch_seconds'].mean() / 3600,
        'avg_age': cluster_data['age'].mean()
    }
    persona_insights.append(insight)

persona_df = pd.DataFrame(persona_insights)
print("\nPersona Insights:")
print(persona_df.round(3))

persona_df.to_csv(OUTPUT('persona_insights_behavioral.csv'), index=False)
print(f"\n✓ Saved persona_insights_behavioral.csv")

# ============================================================
# PART 10: SAVE RESULTS
# ============================================================
print("\n[10] Saving Results")

user_profiles_active.to_csv(OUTPUT('user_profiles_with_clusters_behavioral.csv'), index=False)
print("✓ Saved user_profiles_with_clusters_behavioral.csv")

print("\n" + "=" * 70)
print("COMPLETE - Behavioral Clustering")
print("=" * 70)
print(f"✓ Optimal clusters: {optimal_k}")
print(f"✓ Silhouette score: {best_silhouette:.3f}")
print(f"✓ Total users clustered: {len(user_profiles_active):,}")
print("\nKey approach:")
print("  - CLUSTERED on: Behavioral patterns (HOW users engage)")
print("  - DESCRIBED with: Context features (WHAT they consume, volume)")
print("  - This avoids self-referential loops!")
print("\nGenerated files:")
print("  - cluster_evaluation_behavioral.png")
print("  - cluster_visualization_behavioral.png")
print("  - cluster_summary_behavioral.csv")
print("  - persona_insights_behavioral.csv")
print("  - user_profiles_with_clusters_behavioral.csv")
print("\nNext: Analyze persona_insights_behavioral.csv and create compelling personas!")
print("=" * 70)