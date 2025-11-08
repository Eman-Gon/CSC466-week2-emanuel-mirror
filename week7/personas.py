# personas.py
"""
Week 7: User Personas via Clustering
Emanuel Gonzalez
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

print("="*60)
print("WEEK 7: USER PERSONA DISCOVERY")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1] Loading data...")
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))
df_cancels = pd.read_parquet(P("cancellations.parquet"))

print(f"    Views: {len(df_views):,}")
print(f"    Adventurers: {len(df_adventurers):,}")
print(f"    Subscriptions: {len(df_subs):,}")
print(f"    Cancellations: {len(df_cancels):,}")

# Check what columns cancellations actually has
print(f"    Cancellation columns: {df_cancels.columns.tolist()}")

# ============================================
# 2. CLEAN DATA
# ============================================
print("\n[2] Cleaning data...")

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
].copy()

print(f"    Clean views: {len(df_views_clean):,}")

# ============================================
# 3. BUILD USER PROFILES
# ============================================
print("\n[3] Building user profiles...")

# VIEWING BEHAVIOR
viewing_features = df_views_clean.groupby('adventurer_id').agg({
    'seconds_viewed': ['sum', 'mean', 'count'],
    'watch_pct': ['mean', 'median'],
    'content_id': 'nunique'
}).reset_index()

viewing_features.columns = ['adventurer_id', 
                            'total_watch_time', 
                            'avg_watch_time',
                            'num_views',
                            'avg_completion_rate',
                            'median_completion_rate',
                            'unique_content']

# SUBSCRIPTION BEHAVIOR
sub_features = df_subs.groupby('adventurer_id').agg({
    'publisher_id': ['count', 'nunique']
}).reset_index()
sub_features.columns = ['adventurer_id', 'num_subscriptions', 'num_publishers']

# CHURN BEHAVIOR - just count how many times they churned
churn_features = df_cancels.groupby('adventurer_id').size().reset_index(name='num_churns')

# CONTENT PREFERENCES
df_views_enriched = df_views_clean.merge(
    df_metadata[['content_id', 'genre_id', 'language_code']], 
    on='content_id', 
    how='left'
)

genre_diversity = df_views_enriched.groupby('adventurer_id')['genre_id'].nunique().reset_index()
genre_diversity.columns = ['adventurer_id', 'genre_diversity']

lang_diversity = df_views_enriched.groupby('adventurer_id')['language_code'].nunique().reset_index()
lang_diversity.columns = ['adventurer_id', 'lang_diversity']

# MERGE EVERYTHING
user_profiles = df_adventurers[['adventurer_id', 'age']].copy()

for df in [viewing_features, sub_features, churn_features, 
           genre_diversity, lang_diversity]:
    user_profiles = user_profiles.merge(df, on='adventurer_id', how='left')

user_profiles = user_profiles.fillna(0)

print(f"    User profiles: {user_profiles.shape}")
print(f"    Features: {user_profiles.columns.tolist()}")

# ============================================
# 4. PREPARE FOR CLUSTERING
# ============================================
print("\n[4] Preparing features for clustering...")

# Select features (behavioral only for clustering)
clustering_features = [
    'total_watch_time',
    'avg_watch_time',
    'num_views',
    'unique_content',
    'avg_completion_rate',
    'num_subscriptions',
    'num_publishers',
    'num_churns',
    'genre_diversity',
    'lang_diversity'
]

X = user_profiles[clustering_features].copy()
print(f"    Feature matrix: {X.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"    Scaled features: mean=0, std=1")

# ============================================
# 5. FIND OPTIMAL K
# ============================================
print("\n[5] Finding optimal number of clusters...")

results = []
K_range = range(3, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, labels)
    inertia = kmeans.inertia_
    
    results.append({
        'k': k,
        'silhouette': silhouette,
        'inertia': inertia
    })
    
    print(f"    k={k}: Silhouette={silhouette:.3f}, Inertia={inertia:.0f}")

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
plt.savefig(P('cluster_evaluation.png'), dpi=300, bbox_inches='tight')
print(f"\n    ✓ Saved cluster_evaluation.png")

# Pick optimal k
optimal_k = int(results_df.loc[results_df['silhouette'].idxmax(), 'k'])
best_silhouette = results_df['silhouette'].max()
print(f"\n    🎯 Optimal k: {optimal_k} (Silhouette: {best_silhouette:.3f})")

# ============================================
# 6. FINAL CLUSTERING
# ============================================
print(f"\n[6] Clustering with k={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
user_profiles['cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"    Cluster distribution:")
for cluster_id in sorted(user_profiles['cluster'].unique()):
    count = (user_profiles['cluster'] == cluster_id).sum()
    pct = count / len(user_profiles) * 100
    print(f"      Cluster {cluster_id}: {count:,} users ({pct:.1f}%)")

# ============================================
# 7. DESCRIBE CLUSTERS
# ============================================
print("\n[7] Describing clusters...")

# Add age back for description
description_features = clustering_features + ['age']

cluster_summary = user_profiles.groupby('cluster')[description_features].agg(['mean', 'median'])
print("\n    Cluster Summary (mean values):")
print(cluster_summary.xs('mean', level=1, axis=1).round(2))

# Save detailed summary
cluster_summary.to_csv(P('cluster_summary.csv'))
print(f"\n    ✓ Saved cluster_summary.csv")

# ============================================
# 8. VISUALIZE WITH PCA
# ============================================
print("\n[8] Creating visualizations...")

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=user_profiles['cluster'], 
                     cmap='viridis', 
                     alpha=0.6, 
                     s=50,
                     edgecolors='black',
                     linewidth=0.5)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
          fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
          fontsize=12, fontweight='bold')
plt.title(f'User Personas (k={optimal_k} clusters)', 
         fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(P('cluster_visualization.png'), dpi=300, bbox_inches='tight')
print(f"    ✓ Saved cluster_visualization.png")

# ============================================
# 9. PERSONA PROFILES
# ============================================
print("\n[9] Creating persona profiles...")

# For each cluster, get typical characteristics
persona_insights = []

for cluster_id in sorted(user_profiles['cluster'].unique()):
    cluster_data = user_profiles[user_profiles['cluster'] == cluster_id]
    
    insight = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        'avg_views': cluster_data['num_views'].mean(),
        'avg_completion': cluster_data['avg_completion_rate'].mean(),
        'avg_churns': cluster_data['num_churns'].mean(),
        'avg_subs': cluster_data['num_subscriptions'].mean(),
        'avg_age': cluster_data['age'].mean()
    }
    persona_insights.append(insight)

persona_df = pd.DataFrame(persona_insights)
print("\n" + "="*60)
print("PERSONA INSIGHTS")
print("="*60)
print(persona_df.round(2))

persona_df.to_csv(P('persona_insights.csv'), index=False)
print(f"\n✓ Saved persona_insights.csv")

# ============================================
# 10. SAVE RESULTS
# ============================================
print("\n[10] Saving results...")

user_profiles.to_csv(P('user_profiles_with_clusters.csv'), index=False)
print(f"    ✓ Saved user_profiles_with_clusters.csv")

print("\n" + "="*60)
print("COMPLETE! 🎉")
print("="*60)
print("\nGenerated files:")
print("  - cluster_evaluation.png")
print("  - cluster_visualization.png")
print("  - cluster_summary.csv")
print("  - persona_insights.csv")
print("  - user_profiles_with_clusters.csv")
print("\nNext steps:")
print("  1. Analyze cluster_summary.csv to name your personas")
print("  2. Create compelling visualizations")
print("  3. Write writeup.md connecting to churn/recommendations")