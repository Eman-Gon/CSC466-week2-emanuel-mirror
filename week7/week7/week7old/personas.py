"""
Week 7: User Personas via SVD-Based Clustering
Steven Gonzalez
CSC-466 Fall 2025

APPROACH:
1. Cluster on LATENT FEATURES (SVD embeddings from user-content matrix)
2. Describe with BEHAVIORAL FEATURES (completion patterns, preferences)

This avoids self-referential loops while providing interpretability.
Based on Lucas's lecture: "cluster embeddings, but then use features that 
didn't go into those embeddings to describe the clusters"
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

def calculate_gini(values):
    """Calculate Gini coefficient for concentration measurement"""
    if len(values) == 0 or values.sum() == 0:
        return 0
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

print("WEEK 7: USER PERSONAS VIA SVD-BASED CLUSTERING")
print("=" * 70)
print("\nAPPROACH:")
print("  1. Extract latent features via SVD (avoid self-referential loops)")
print("  2. Cluster on latent features")
print("  3. Describe clusters using behavioral features (interpretability)")
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

# Keep important columns including minutes
df_views_clean = df_merged[
    (df_merged['watch_pct'].fillna(0) >= 0.05) | 
    (df_merged['seconds_viewed'] >= 30)
][['adventurer_id', 'content_id', 'seconds_viewed', 'watch_pct', 'minutes']].copy()

print(f"    Clean views: {len(df_views_clean):,}")

# ============================================================
# PART 2: CREATE USER-CONTENT INTERACTION MATRIX
# ============================================================
print("\n[2] Building User-Content Interaction Matrix")

# Create pivot table: rows=users, columns=content, values=watch_pct
interaction_matrix = df_views_clean.pivot_table(
    index='adventurer_id',
    columns='content_id',
    values='watch_pct',
    fill_value=0
)

print(f"    Matrix shape: {interaction_matrix.shape}")
print(f"    Users: {interaction_matrix.shape[0]:,}")
print(f"    Content items: {interaction_matrix.shape[1]:,}")
print(f"    Sparsity: {(interaction_matrix == 0).sum().sum() / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.1%}")

# ============================================================
# PART 3: EXTRACT LATENT FEATURES VIA SVD
# ============================================================
print("\n[3] Extracting Latent Features via SVD")

# Determine number of components
# Rule of thumb: capture 80-90% of variance, but also keep it manageable
n_components = min(50, interaction_matrix.shape[1] - 1)

svd = TruncatedSVD(n_components=n_components, random_state=42)
latent_features = svd.fit_transform(interaction_matrix)

# Analyze explained variance
cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
components_80 = np.argmax(cumulative_variance >= 0.80) + 1
components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print(f"    SVD components: {n_components}")
print(f"    Total variance explained: {svd.explained_variance_ratio_.sum():.1%}")
print(f"    Components for 80% variance: {components_80}")
print(f"    Components for 90% variance: {components_90}")

# Visualize variance
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, n_components + 1), svd.explained_variance_ratio_, 'b-', linewidth=2)
plt.xlabel('Component', fontsize=12, fontweight='bold')
plt.ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
plt.title('SVD Scree Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, n_components + 1), cumulative_variance, 'r-', linewidth=2)
plt.axhline(y=0.80, color='g', linestyle='--', label='80% variance')
plt.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
plt.xlabel('Component', fontsize=12, fontweight='bold')
plt.ylabel('Cumulative Variance', fontsize=12, fontweight='bold')
plt.title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(P('svd_variance_explained.png'), dpi=300, bbox_inches='tight')
print(f"\n    ✓ Saved svd_variance_explained.png")

# Use components that explain 80% variance for clustering
latent_features_reduced = latent_features[:, :components_80]
print(f"\n    Using {components_80} components for clustering (80% variance)")

# ============================================================
# PART 4: FIND OPTIMAL K
# ============================================================
print("\n[4] Finding Optimal Number of Clusters")

results = []
K_range = range(3, 11)

print("\nTesting k values:")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
    labels = kmeans.fit_predict(latent_features_reduced)
    
    silhouette = silhouette_score(latent_features_reduced, labels)
    davies_bouldin = davies_bouldin_score(latent_features_reduced, labels)
    calinski = calinski_harabasz_score(latent_features_reduced, labels)
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
plt.savefig(P('cluster_evaluation_svd.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved cluster_evaluation_svd.png")

# Select optimal k
optimal_k = int(results_df.loc[results_df['silhouette'].idxmax(), 'k'])
best_silhouette = results_df['silhouette'].max()
print(f"\n{'='*70}")
print(f"OPTIMAL K = {optimal_k} (Silhouette: {best_silhouette:.3f})")
print(f"{'='*70}")

# ============================================================
# PART 5: CLUSTER ON LATENT FEATURES
# ============================================================
print(f"\n[5] Clustering on Latent Features (k={optimal_k})")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=300)
cluster_labels = kmeans_final.fit_predict(latent_features_reduced)

# Create user profiles dataframe with cluster assignments
user_profiles = pd.DataFrame({
    'adventurer_id': interaction_matrix.index,
    'cluster': cluster_labels
})

print("\nCluster distribution:")
for cluster_id in sorted(user_profiles['cluster'].unique()):
    count = (user_profiles['cluster'] == cluster_id).sum()
    pct = count / len(user_profiles) * 100
    print(f"  Cluster {cluster_id}: {count:,} users ({pct:.1f}%)")

# ============================================================
# PART 6: BUILD BEHAVIORAL FEATURES FOR INTERPRETATION
# ============================================================
print("\n[6] Building Behavioral Features for Interpretation")
print("    (NOT used for clustering, only for describing clusters)")

# Merge with adventurer data
user_profiles = user_profiles.merge(
    df_adventurers[['adventurer_id', 'age']], 
    on='adventurer_id', 
    how='left'
)

# Enrich views with metadata
df_views_enriched = df_views_clean.merge(
    df_metadata[['content_id', 'genre_id', 'language_code', 'minutes']], 
    on='content_id', 
    how='left'
)

# 1. Completion behavior
print("    ✓ Completion behavior")
completion_features = df_views_enriched.groupby('adventurer_id').agg({
    'watch_pct': ['mean', 'std', 'median', 'count']
}).reset_index()
completion_features.columns = ['adventurer_id', 'avg_completion', 'completion_std', 
                                'median_completion', 'num_views']
completion_features['completion_consistency'] = 1 / (1 + completion_features['completion_std'])

finisher_counts = df_views_enriched[df_views_enriched['watch_pct'] >= 0.8]\
    .groupby('adventurer_id').size()
finisher_ratio_df = (finisher_counts / completion_features.set_index('adventurer_id')['num_views'])\
    .reset_index(name='finisher_ratio')
completion_features = completion_features.merge(finisher_ratio_df, on='adventurer_id', how='left')
completion_features['finisher_ratio'] = completion_features['finisher_ratio'].fillna(0)

# 2. Genre concentration
print("    ✓ Genre concentration")
genre_concentration = []
for adv_id in df_views_enriched['adventurer_id'].unique():
    user_views = df_views_enriched[df_views_enriched['adventurer_id'] == adv_id]
    genre_counts = user_views['genre_id'].value_counts().values
    gini_coef = calculate_gini(genre_counts)
    genre_concentration.append({'adventurer_id': adv_id, 'genre_concentration': gini_coef})
genre_df = pd.DataFrame(genre_concentration)

# 3. Content length preference
print("    ✓ Content length preference")
content_length_pref = df_views_clean.groupby('adventurer_id')['minutes'].mean()\
    .reset_index(name='avg_content_length')

# 4. Subscription strategy
print("    ✓ Subscription strategy")
sub_counts = df_subs.groupby('adventurer_id').size().reset_index(name='num_subs')
churn_counts = df_cancels.groupby('adventurer_id').size().reset_index(name='num_churns')
sub_strategy = sub_counts.merge(churn_counts, on='adventurer_id', how='left').fillna(0)
sub_strategy['churn_rate'] = (sub_strategy['num_churns'] / sub_strategy['num_subs']).clip(0, 1).fillna(0)

publisher_loyalty = df_subs.groupby('adventurer_id')['publisher_id'].nunique()\
    .reset_index(name='num_publishers')
sub_strategy = sub_strategy.merge(publisher_loyalty, on='adventurer_id', how='left')
sub_strategy['publisher_concentration'] = 1 / (1 + sub_strategy['num_publishers'])

# 5. Viewing consistency
print("    ✓ Viewing consistency")
viewing_var = df_views_enriched.groupby('adventurer_id')['seconds_viewed'].std()\
    .reset_index(name='viewing_time_std')
viewing_mean = df_views_enriched.groupby('adventurer_id')['seconds_viewed'].mean()\
    .reset_index(name='viewing_time_mean')
viewing_consistency = viewing_var.merge(viewing_mean, on='adventurer_id')
viewing_consistency['viewing_consistency'] = (
    viewing_consistency['viewing_time_std'] / (viewing_consistency['viewing_time_mean'] + 1)
)

# Merge all behavioral features
for df in [completion_features, genre_df, content_length_pref, 
           sub_strategy, viewing_consistency]:
    user_profiles = user_profiles.merge(df, on='adventurer_id', how='left')

user_profiles = user_profiles.fillna(0)

print(f"\n    ✓ User profiles with behavioral features: {user_profiles.shape}")

# ============================================================
# PART 7: DESCRIBE CLUSTERS USING BEHAVIORAL FEATURES
# ============================================================
print("\n[7] Describing Clusters with Behavioral Features")

behavioral_features = [
    'avg_completion', 'completion_consistency', 'finisher_ratio',
    'genre_concentration', 'avg_content_length',
    'churn_rate', 'publisher_concentration',
    'viewing_consistency', 'num_views', 'age'
]

cluster_summary = user_profiles.groupby('cluster')[behavioral_features].agg(['mean', 'median', 'std'])

print("\nCluster Summary (mean values):")
print(cluster_summary.xs('mean', level=1, axis=1).round(3))

cluster_summary.to_csv(P('cluster_summary_svd.csv'))
print(f"\n✓ Saved cluster_summary_svd.csv")

# ============================================================
# PART 8: VISUALIZATION
# ============================================================
print("\n[8] Visualization")

# PCA for 2D visualization of latent features
# Handle case where we only have 1 SVD component
if latent_features_reduced.shape[1] == 1:
    # Use the 1 component for x-axis, create dummy y-axis
    latent_pca = np.column_stack([latent_features_reduced[:, 0], np.zeros(len(latent_features_reduced))])
    var_x, var_y = 100.0, 0.0
else:
    pca = PCA(n_components=2, random_state=42)
    latent_pca = pca.fit_transform(latent_features_reduced)
    var_x, var_y = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

plt.figure(figsize=(12, 8))
scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1],
                     c=cluster_labels,
                     cmap='tab10',
                     alpha=0.6,
                     s=50,
                     edgecolors='black',
                     linewidth=0.5)

plt.xlabel(f'SVD Component 1 ({var_x:.1%} variance)', 
          fontsize=12, fontweight='bold')
plt.ylabel(f'Dummy axis' if var_y == 0 else f'PC2 ({var_y:.1%} variance)', 
          fontsize=12, fontweight='bold')
plt.title(f'User Personas - SVD-Based Clustering (k={optimal_k})', 
         fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(P('cluster_visualization_svd.png'), dpi=300, bbox_inches='tight')
print("✓ Saved cluster_visualization_svd.png")

# ============================================================
# PART 9: CREATE PERSONA INSIGHTS
# ============================================================
print("\n[9] Creating Persona Insights")

persona_insights = []

for cluster_id in sorted(user_profiles['cluster'].unique()):
    cluster_data = user_profiles[user_profiles['cluster'] == cluster_id]
    
    insight = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        'avg_completion': cluster_data['avg_completion'].mean(),
        'completion_consistency': cluster_data['completion_consistency'].mean(),
        'finisher_ratio': cluster_data['finisher_ratio'].mean(),
        'genre_concentration': cluster_data['genre_concentration'].mean(),
        'churn_rate': cluster_data['churn_rate'].mean(),
        'viewing_consistency': cluster_data['viewing_consistency'].mean(),
        'num_views': cluster_data['num_views'].mean(),
        'avg_age': cluster_data['age'].mean()
    }
    persona_insights.append(insight)

persona_df = pd.DataFrame(persona_insights)
print("\nPersona Insights:")
print(persona_df.round(3))

persona_df.to_csv(P('persona_insights_svd.csv'), index=False)
print(f"\n✓ Saved persona_insights_svd.csv")

# ============================================================
# PART 10: SAVE RESULTS
# ============================================================
print("\n[10] Saving Results")

user_profiles.to_csv(P('user_profiles_with_clusters_svd.csv'), index=False)
print("✓ Saved user_profiles_with_clusters_svd.csv")

print("\n" + "=" * 70)
print("COMPLETE - SVD-Based Clustering")
print("=" * 70)
print(f"✓ Optimal clusters: {optimal_k}")
print(f"✓ Silhouette score: {best_silhouette:.3f}")
print(f"✓ Total users: {len(user_profiles):,}")
print(f"✓ SVD components used: {components_80} (80% variance)")
print("\nGenerated files:")
print("  - svd_variance_explained.png")
print("  - cluster_evaluation_svd.png")
print("  - cluster_visualization_svd.png")
print("  - cluster_summary_svd.csv")
print("  - persona_insights_svd.csv")
print("  - user_profiles_with_clusters_svd.csv")
print("\nNext: Analyze persona_insights_svd.csv and write compelling personas!")
print("=" * 70)