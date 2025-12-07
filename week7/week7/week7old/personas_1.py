# personas.py
"""
Week 7: User Personas via Clustering - FIXED VERSION
Steven Gonzalez
CSC-466 Fall 2025

IMPROVEMENTS:
- Uses behavioral ratios instead of volume metrics
- Avoids self-referential loops
- Features capture HOW users behave, not HOW MUCH
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name


print("WEEK 7: USER PERSONA DISCOVERY (IMPROVED)")
print("\n[1] Loading data")
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))
df_cancels = pd.read_parquet(P("cancellations.parquet"))

print(f"    Views: {len(df_views):,}")
print(f"    Adventurers: {len(df_adventurers):,}")
print(f"    Subscriptions: {len(df_subs):,}")
print(f"    Cancellations: {len(df_cancels):,}")

print("\n[2] Cleaning data")

# Remove duplicates
df_views = df_views.sort_values('seconds_viewed', ascending=False)\
    .drop_duplicates(subset=['adventurer_id', 'content_id'], keep='first')

# Calculate watch percentage
df_merged = df_views.merge(
    df_metadata[['content_id', 'minutes']], 
    on='content_id', 
    how='left'
)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / (df_merged['minutes'] * 60)).clip(0, 1)

# Filter low engagement
df_views_clean = df_merged[
    (df_merged['watch_pct'].fillna(0) >= 0.05) | 
    (df_merged['seconds_viewed'] >= 30)
][['adventurer_id', 'content_id', 'seconds_viewed', 'watch_pct', 'minutes']].copy()

print(f"    Clean views: {len(df_views_clean):,}")

print("\n[3] Building user profiles with BEHAVIORAL features")

# ===================================================================
# VIEWING BEHAVIOR - Include std dev for consistency calculations
# ===================================================================
viewing_features = df_views_clean.groupby('adventurer_id').agg({
    'seconds_viewed': ['sum', 'mean', 'count', 'std'],
    'watch_pct': ['mean', 'median', 'std'],
    'content_id': 'nunique'
}).reset_index()

viewing_features.columns = ['adventurer_id', 
                            'total_watch_time', 
                            'avg_watch_time',
                            'num_views',
                            'viewing_time_std',
                            'avg_completion_rate',
                            'median_completion_rate',
                            'completion_std',
                            'unique_content']

# Behavioral feature 1: Completion consistency (are they predictable?)
viewing_features['completion_consistency'] = 1 / (1 + viewing_features['completion_std'].fillna(0))

# Behavioral feature 2: Viewing consistency (binger vs. steady viewer)
viewing_features['viewing_consistency'] = (
    viewing_features['viewing_time_std'] / (viewing_features['avg_watch_time'] + 1)
).fillna(0)

print("    ✓ Viewing features with consistency metrics")

# ===================================================================
# SUBSCRIPTION BEHAVIOR - Calculate ratios for loyalty
# ===================================================================
sub_features = df_subs.groupby('adventurer_id').agg({
    'publisher_id': ['count', 'nunique']
}).reset_index()
sub_features.columns = ['adventurer_id', 'num_subscriptions', 'num_publishers']

# Behavioral feature 3: Publisher concentration (loyal vs. variety-seeking)
sub_features['publisher_concentration'] = 1 / (1 + sub_features['num_publishers'])

print("    ✓ Subscription features with loyalty metrics")

# ===================================================================
# CHURN BEHAVIOR - Calculate rate, not count
# ===================================================================
churn_features = df_cancels.groupby('adventurer_id').size().reset_index(name='num_churns')

# Merge to calculate churn rate
sub_churn = sub_features[['adventurer_id', 'num_subscriptions']].merge(
    churn_features, on='adventurer_id', how='left'
).fillna(0)

# Behavioral feature 4: Churn rate (not raw count)
sub_churn['churn_rate'] = (sub_churn['num_churns'] / sub_churn['num_subscriptions']).clip(0, 1).fillna(0)

print("    ✓ Churn features with rate calculation")

# ===================================================================
# CONTENT PREFERENCES - Concentration, not just diversity
# ===================================================================
# First merge with metadata to get genre and language
df_views_enriched = df_views_clean.merge(
    df_metadata[['content_id', 'genre_id', 'language_code']], 
    on='content_id', 
    how='left'
)

# Behavioral feature 5: Average content length (short-form vs. long-form)
# Use the minutes column from df_views_clean (already has it from earlier merge)
content_length = df_views_clean.groupby('adventurer_id')['minutes'].mean().reset_index()
content_length.columns = ['adventurer_id', 'avg_content_length']

# Behavioral feature 6: Genre concentration using Gini coefficient
def calculate_gini(values):
    """Gini coefficient: 0 = perfect equality, 1 = perfect inequality"""
    if len(values) == 0 or values.sum() == 0:
        return 0
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

genre_gini = []
for adv_id in df_views_enriched['adventurer_id'].unique():
    user_views = df_views_enriched[df_views_enriched['adventurer_id'] == adv_id]
    genre_counts = user_views['genre_id'].value_counts().values
    gini_coef = calculate_gini(genre_counts)
    genre_gini.append({'adventurer_id': adv_id, 'genre_concentration': gini_coef})

genre_concentration_df = pd.DataFrame(genre_gini)

print("    ✓ Content preference features with Gini concentration")

# Behavioral feature 7: Finisher ratio (% of content watched >80%)
finisher_counts = df_views_enriched[df_views_enriched['watch_pct'] >= 0.8].groupby('adventurer_id').size()
total_counts = df_views_enriched.groupby('adventurer_id').size()
finisher_ratio = (finisher_counts / total_counts).reset_index(name='finisher_ratio')
finisher_ratio['finisher_ratio'] = finisher_ratio['finisher_ratio'].fillna(0)

print("    ✓ Finisher ratio calculated")

# ===================================================================
# MERGE ALL FEATURES
# ===================================================================
user_profiles = df_adventurers[['adventurer_id', 'age']].copy()

for df in [viewing_features, sub_features, sub_churn[['adventurer_id', 'churn_rate']], 
           content_length, genre_concentration_df, finisher_ratio]:
    user_profiles = user_profiles.merge(df, on='adventurer_id', how='left')

user_profiles = user_profiles.fillna(0)

# Behavioral feature 8: Is active flag (basic engagement indicator)
user_profiles['is_active'] = (user_profiles['num_views'] > 0).astype(int)

print(f"\n    User profiles: {user_profiles.shape}")
print(f"    All columns: {user_profiles.columns.tolist()}")

print("\n[4] Selecting BEHAVIORAL features for clustering")
print("    (No volume metrics - using ratios and patterns only)")

# Select features - BEHAVIORAL ONLY, no volume
clustering_features = [
    # Completion behavior (HOW they watch)
    'avg_completion_rate',           # Do they finish content?
    'completion_consistency',        # Are they predictable?
    'finisher_ratio',               # % watched >80%
    
    # Content preferences (WHAT they prefer)
    'genre_concentration',          # Specialist vs. generalist
    'avg_content_length',           # Short-form vs. long-form
    
    # Subscription strategy (LOYALTY patterns)
    'churn_rate',                   # Stable vs. churner (RATIO)
    'publisher_concentration',      # Loyal vs. variety-seeking
    
    # Engagement pattern (CONSISTENCY)
    'viewing_consistency',          # Binger vs. steady viewer
    
    # Basic flag
    'is_active'                     # Engaged vs. dormant
]

print(f"\n    Clustering on {len(clustering_features)} behavioral features:")
for feat in clustering_features:
    print(f"      - {feat}")

X = user_profiles[clustering_features].copy()
print(f"\n    Feature matrix: {X.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"    ✓ Features scaled (mean=0, std=1)")

print("\n[5] Finding optimal number of clusters")

results = []
K_range = range(3, 9)

print("\nTesting k values:")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, labels)
    inertia = kmeans.inertia_
    
    results.append({
        'k': k,
        'silhouette': silhouette,
        'inertia': inertia
    })
    
    print(f"  k={k}: Silhouette={silhouette:.3f}, Inertia={inertia:.0f}")

results_df = pd.DataFrame(results)

# Plot elbow + silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(results_df['k'], results_df['inertia'], 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Inertia', fontsize=12, fontweight='bold')
ax1.set_title('Elbow Plot', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(results_df['k'], results_df['silhouette'], 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax2.set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(P('cluster_evaluation_improved.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved cluster_evaluation_improved.png")

# Pick optimal k
optimal_k = int(results_df.loc[results_df['silhouette'].idxmax(), 'k'])
best_silhouette = results_df['silhouette'].max()
print(f"\n{'='*60}")
print(f"OPTIMAL K = {optimal_k} (Silhouette: {best_silhouette:.3f})")
print(f"{'='*60}")

print(f"\n[6] Clustering with k={optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=300)
user_profiles['cluster'] = kmeans_final.fit_predict(X_scaled)

print("\nCluster distribution:")
for cluster_id in sorted(user_profiles['cluster'].unique()):
    count = (user_profiles['cluster'] == cluster_id).sum()
    pct = count / len(user_profiles) * 100
    print(f"  Cluster {cluster_id}: {count:,} users ({pct:.1f}%)")

print("\n[7] Describing clusters")

# Include behavioral features + context features for description
# (Note: num_churns and num_subscriptions are in the original merge but may not be in user_profiles)
# Use what we actually have
description_features = [
    # Behavioral features used in clustering
    'avg_completion_rate',
    'completion_consistency',
    'finisher_ratio',
    'genre_concentration',
    'avg_content_length',
    'churn_rate',
    'publisher_concentration',
    'viewing_consistency',
    'is_active',
    # Context features for interpretation
    'age',
    'num_views',
    'num_subscriptions'
]

cluster_summary = user_profiles.groupby('cluster')[description_features].agg(['mean', 'median'])
print("\nCluster Summary (mean values):")
print(cluster_summary.xs('mean', level=1, axis=1).round(3))

cluster_summary.to_csv(P('cluster_summary_improved.csv'))
print(f"\n✓ Saved cluster_summary_improved.csv")

print("\n[8] Creating visualizations")

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=user_profiles['cluster'], 
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
plt.savefig(P('cluster_visualization_improved.png'), dpi=300, bbox_inches='tight')
print("✓ Saved cluster_visualization_improved.png")

print("\n[9] Creating persona insights")

persona_insights = []

for cluster_id in sorted(user_profiles['cluster'].unique()):
    cluster_data = user_profiles[user_profiles['cluster'] == cluster_id]
    
    insight = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        # Behavioral characteristics
        'avg_completion': cluster_data['avg_completion_rate'].mean(),
        'completion_consistency': cluster_data['completion_consistency'].mean(),
        'finisher_ratio': cluster_data['finisher_ratio'].mean(),
        'genre_concentration': cluster_data['genre_concentration'].mean(),
        'avg_content_length': cluster_data['avg_content_length'].mean(),
        'churn_rate': cluster_data['churn_rate'].mean(),
        'publisher_concentration': cluster_data['publisher_concentration'].mean(),
        'viewing_consistency': cluster_data['viewing_consistency'].mean(),
        # Context (for interpretation, not clustering)
        'avg_views': cluster_data['num_views'].mean(),
        'avg_subs': cluster_data['num_subscriptions'].mean(),
        'avg_age': cluster_data['age'].mean()
    }
    persona_insights.append(insight)

persona_df = pd.DataFrame(persona_insights)
print("\nPersona Insights:")
print(persona_df.round(3))

persona_df.to_csv(P('persona_insights_improved.csv'), index=False)
print(f"\n✓ Saved persona_insights_improved.csv")

print("\n[10] Saving results")

user_profiles.to_csv(P('user_profiles_with_clusters_improved.csv'), index=False)
print("✓ Saved user_profiles_with_clusters_improved.csv")

print("\n" + "=" * 60)
print("COMPLETE - Improved Behavioral Clustering")
print("=" * 60)
print(f"✓ Optimal clusters: {optimal_k}")
print(f"✓ Silhouette score: {best_silhouette:.3f}")
print(f"✓ Total users: {len(user_profiles):,}")
print(f"✓ Behavioral features used: {len(clustering_features)}")
print("\nKey improvements over original:")
print("  - Used behavioral ratios instead of volume metrics")
print("  - Avoided self-referential loops")
print("  - Captures HOW users behave, not HOW MUCH they consume")
print("\nGenerated files:")
print("  - cluster_evaluation_improved.png")
print("  - cluster_visualization_improved.png")
print("  - cluster_summary_improved.csv")
print("  - persona_insights_improved.csv")
print("  - user_profiles_with_clusters_improved.csv")
print("\nNext: Analyze persona_insights_improved.csv and name your personas!")
print("=" * 60)