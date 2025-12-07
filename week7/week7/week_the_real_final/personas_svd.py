from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
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




# PART 1: LOAD DATA

print("\n[1] Loading Data")

df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))
df_cancels = pd.read_parquet(P("cancellations.parquet"))

print(f"    Views: {len(df_views):,}")
print(f"    Adventurers: {len(df_adventurers):,}")
print(f"    Content: {len(df_metadata):,}")
print(f"    Adventurer columns: {df_adventurers.columns.tolist()}")
print(f"    Subscription columns: {df_subs.columns.tolist()}")


# PART 2: CLEAN AND PREPARE DATA

print("\n[2] Cleaning Data")

# Remove duplicate views (keep highest watch time)
df_views = df_views.sort_values('seconds_viewed', ascending=False)\
    .drop_duplicates(subset=['adventurer_id', 'content_id'], keep='first')

print(f"    After deduplication: {len(df_views):,} views")

# Calculate watch percentage
df_merged = df_views.merge(
    df_metadata[['content_id', 'minutes']], 
    on='content_id', 
    how='left'
)
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / (df_merged['minutes'] * 60)).clip(0, 1)

# Filter meaningful views (>5% watched OR >30 seconds)
df_views_clean = df_merged[
    (df_merged['watch_pct'].fillna(0) >= 0.05) | 
    (df_merged['seconds_viewed'] >= 30)
].copy()

print(f"    After filtering: {len(df_views_clean):,} meaningful views")

# Filter to active users (at least 5 views for meaningful clustering)
MIN_VIEWS = 10
user_view_counts = df_views_clean['adventurer_id'].value_counts()
active_users = user_view_counts[user_view_counts >= MIN_VIEWS].index

df_views_filtered = df_views_clean[df_views_clean['adventurer_id'].isin(active_users)].copy()
print(f"    Active users (>={MIN_VIEWS} views): {len(active_users):,}")
print(f"    Views from active users: {len(df_views_filtered):,}")


# PART 3: BUILD USER-CONTENT INTERACTION MATRIX

print("\n[3] Building User-Content Interaction Matrix")

# Create mappings
user_ids = df_views_filtered['adventurer_id'].unique()
content_ids = df_views_filtered['content_id'].unique()

user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
content_id_map = {cid: idx for idx, cid in enumerate(content_ids)}

print(f"    Matrix dimensions: {len(user_ids):,} users × {len(content_ids):,} content")

# Map to indices
df_views_filtered['user_idx'] = df_views_filtered['adventurer_id'].map(user_id_map)
df_views_filtered['content_idx'] = df_views_filtered['content_id'].map(content_id_map)

# Create sparse interaction matrix
# Using watch_pct as interaction strength (0-1)
interaction_matrix = csr_matrix(
    (df_views_filtered['watch_pct'].fillna(0).values, 
     (df_views_filtered['user_idx'].values, df_views_filtered['content_idx'].values)),
    shape=(len(user_ids), len(content_ids))
)

print(f"    Matrix shape: {interaction_matrix.shape}")
print(f"    Matrix density: {interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]) * 100:.3f}%")
print(f"    Non-zero entries: {interaction_matrix.nnz:,}")


# PART 4: APPLY SVD TO EXTRACT LATENT FEATURES

print("\n[4] Applying SVD to Extract Latent Features")

# Choose number of latent dimensions
# Rule of thumb: sqrt(min(n_users, n_content)) or 50-100
n_components = 30
# min(100, min(len(user_ids), len(content_ids)) // 2)
print(f"    Using {n_components} latent dimensions")

svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=10)
latent_features = svd.fit_transform(interaction_matrix)

print(f"    Latent feature matrix shape: {latent_features.shape}")
print(f"    Explained variance: {svd.explained_variance_ratio_.sum():.1%}")
print(f"    Top 10 components explain: {svd.explained_variance_ratio_[:10].sum():.1%}")

# Visualize variance explained
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, n_components + 1), svd.explained_variance_ratio_, 'bo-', alpha=0.6)
plt.xlabel('Component', fontweight='bold', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontweight='bold', fontsize=12)
plt.title('SVD Component Variance', fontweight='bold', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cumsum = np.cumsum(svd.explained_variance_ratio_)
plt.plot(range(1, n_components + 1), cumsum, 'ro-', alpha=0.6)
plt.axhline(y=0.8, color='g', linestyle='--', label='80% variance')
plt.xlabel('Component', fontweight='bold', fontsize=12)
plt.ylabel('Cumulative Explained Variance', fontweight='bold', fontsize=12)
plt.title('Cumulative Variance Explained', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT('svd_variance_explained.png'), dpi=300, bbox_inches='tight')
print(f"    ✓ Saved svd_variance_explained.png")


# PART 5: FIND OPTIMAL K

print("\n[5] Finding Optimal Number of Clusters")

# Standardize latent features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(latent_features)
print(f"    ✓ Standardized latent features")

results = []
K_range = range(2, 8)

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
plt.savefig(OUTPUT('cluster_evaluation_svd.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved cluster_evaluation_svd.png")

# Select optimal k
optimal_k = int(results_df.loc[results_df['silhouette'].idxmax(), 'k'])
best_silhouette = results_df['silhouette'].max()

print(f"\n{'='*80}")
print(f"OPTIMAL K = {optimal_k} (Silhouette: {best_silhouette:.3f})")
print(f"{'='*80}")


# PART 6: FINAL CLUSTERING

print(f"\n[6] Final Clustering with k={optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=300)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Create user profiles dataframe
user_profiles = pd.DataFrame({
    'adventurer_id': user_ids,
    'cluster': cluster_labels
})

print("\nCluster distribution:")
for cluster_id in range(optimal_k):
    count = (cluster_labels == cluster_id).sum()
    pct = count / len(cluster_labels) * 100
    print(f"  Cluster {cluster_id}: {count:,} users ({pct:.1f}%)")


# PART 7: INTERPRET CLUSTERS BY CONTENT PREFERENCES

print("\n[7] Interpreting Clusters via Content Analysis")

# Add cluster labels to views
df_views_clustered = df_views_filtered.merge(
    user_profiles[['adventurer_id', 'cluster']], 
    on='adventurer_id', 
    how='left'
)

# Enrich with metadata (publisher_id might not be in content_metadata)
# Check if publisher_id exists in content_metadata
metadata_cols = ['content_id', 'genre_id', 'language_code']
if 'publisher_id' in df_metadata.columns:
    metadata_cols.append('publisher_id')

df_views_enriched = df_views_clustered.merge(
    df_metadata[metadata_cols], 
    on='content_id', 
    how='left'
)

# If publisher_id not in content_metadata, we'll skip publisher-related analysis
has_publisher_in_content = 'publisher_id' in df_views_enriched.columns

cluster_insights = []

for cluster_id in range(optimal_k):
    cluster_views = df_views_enriched[df_views_enriched['cluster'] == cluster_id]
    cluster_users = user_profiles[user_profiles['cluster'] == cluster_id]['adventurer_id'].values
    
    # Content preferences
    top_genres = cluster_views['genre_id'].value_counts().head(5)
    avg_content_length = cluster_views['minutes'].mean()
    avg_completion = cluster_views['watch_pct'].mean()
    
    # Engagement metrics
    avg_views_per_user = len(cluster_views) / len(cluster_users)
    unique_content_per_user = cluster_views.groupby('adventurer_id')['content_id'].nunique().mean()
    
    # Publisher diversity (if available)
    if has_publisher_in_content:
        unique_publishers = cluster_views['publisher_id'].nunique()
    else:
        # Get from subscriptions data instead
        cluster_user_subs = df_subs[df_subs['adventurer_id'].isin(cluster_users)]
        unique_publishers = cluster_user_subs['publisher_id'].nunique() if len(cluster_user_subs) > 0 else 0
    
    # Language diversity
    top_languages = cluster_views['language_code'].value_counts().head(3)
    
    # Churn analysis
    cluster_subs = df_subs[df_subs['adventurer_id'].isin(cluster_users)]
    cluster_churns = df_cancels[df_cancels['adventurer_id'].isin(cluster_users)]
    churn_rate = len(cluster_churns) / len(cluster_subs) if len(cluster_subs) > 0 else 0
    
    insight = {
        'cluster': cluster_id,
        'size': len(cluster_users),
        'avg_views_per_user': avg_views_per_user,
        'unique_content_per_user': unique_content_per_user,
        'avg_completion_rate': avg_completion,
        'avg_content_length_minutes': avg_content_length,
        'unique_publishers': unique_publishers,
        'churn_rate': churn_rate,
        'top_genre': top_genres.index[0] if len(top_genres) > 0 else -1,
        'top_genre_share': top_genres.values[0] / len(cluster_views) if len(top_genres) > 0 else 0,
        'top_language': top_languages.index[0] if len(top_languages) > 0 else 'unknown'
    }
    
    cluster_insights.append(insight)
    
    print(f"\nCluster {cluster_id} ({len(cluster_users):,} users):")
    print(f"  Avg views per user: {avg_views_per_user:.1f}")
    print(f"  Avg completion rate: {avg_completion:.1%}")
    print(f"  Avg content length: {avg_content_length:.1f} min")
    print(f"  Top genres: {dict(top_genres.head(3))}")
    print(f"  Churn rate: {churn_rate:.1%}")

persona_df = pd.DataFrame(cluster_insights)
persona_df.to_csv(OUTPUT('persona_insights_svd.csv'), index=False)
print(f"\n✓ Saved persona_insights_svd.csv")


# PART 8: DETAILED GENRE ANALYSIS PER CLUSTER

print("\n[8] Analyzing Genre Preferences by Cluster")

# Create genre preference matrix
genre_cluster_matrix = df_views_enriched.groupby(['cluster', 'genre_id']).size().unstack(fill_value=0)

# Normalize by cluster size
for cluster_id in range(optimal_k):
    cluster_size = (user_profiles['cluster'] == cluster_id).sum()
    genre_cluster_matrix.loc[cluster_id] = genre_cluster_matrix.loc[cluster_id] / cluster_size

print("\nTop 3 genres per cluster:")
for cluster_id in range(optimal_k):
    top_genres = genre_cluster_matrix.loc[cluster_id].sort_values(ascending=False).head(3)
    print(f"  Cluster {cluster_id}: {dict(top_genres)}")

genre_cluster_matrix.to_csv(OUTPUT('genre_preferences_by_cluster.csv'))
print(f"\n✓ Saved genre_preferences_by_cluster.csv")


# PART 9: VISUALIZATION

print("\n[9] Creating Visualizations")

# PCA for 2D visualization of latent space
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=cluster_labels, 
                     cmap='tab10', 
                     alpha=0.6, 
                     s=50,
                     edgecolors='black',
                     linewidth=0.5)

# Add cluster centroids
centroids_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='red', 
           marker='X', 
           s=300, 
           edgecolors='black', 
           linewidth=2,
           label='Centroids')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
          fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
          fontsize=12, fontweight='bold')
plt.title(f'User Personas in Latent Feature Space (k={optimal_k}, SVD-based)', 
         fontsize=14, fontweight='bold', pad=15)
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT('cluster_visualization_svd.png'), dpi=300, bbox_inches='tight')
print("✓ Saved cluster_visualization_svd.png")
plt.close()


# UMAP Visualization (better cluster separation)

print("Generating UMAP visualization...")

import umap
import plotly.express as px

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

umap_embedding = reducer.fit_transform(X_scaled)

umap_df = pd.DataFrame({
    'UMAP1': umap_embedding[:, 0],
    'UMAP2': umap_embedding[:, 1],
    'Cluster': cluster_labels.astype(str)
})

fig = px.scatter(
    umap_df, 
    x='UMAP1', 
    y='UMAP2', 
    color='Cluster',
    title='User Personas (Interactive UMAP)',
    width=900,
    height=700
)

fig.write_html(OUTPUT('umap_interactive.html'))
print("✓ Saved umap_interactive.html")

# Genre heatmap
plt.figure(figsize=(14, 8))
top_genres = genre_cluster_matrix.sum().sort_values(ascending=False).head(15).index
genre_subset = genre_cluster_matrix[top_genres]

sns.heatmap(genre_subset, 
            annot=True, 
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Avg Views per User'},
            linewidths=0.5,
            linecolor='black')

plt.xlabel('Genre ID', fontsize=12, fontweight='bold')
plt.ylabel('Cluster', fontsize=12, fontweight='bold')
plt.title('Top Genre Preferences by Cluster (Views per User)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(OUTPUT('genre_heatmap_by_cluster.png'), dpi=300, bbox_inches='tight')
print("✓ Saved genre_heatmap_by_cluster.png")
plt.close()


# PART 10: ADD DEMOGRAPHIC CONTEXT

print("\n[10] Adding Demographic Context")

# Merge with adventurer metadata
user_profiles_full = user_profiles.merge(
    df_adventurers[['adventurer_id', 'age', 'gender', 'region']],
    on='adventurer_id', 
    how='left'
)

# Add engagement metrics from views
engagement_metrics = df_views_filtered.groupby('adventurer_id').agg({
    'content_id': 'count',
    'seconds_viewed': 'sum',
    'watch_pct': 'mean'
}).reset_index()
engagement_metrics.columns = ['adventurer_id', 'total_views', 'total_seconds', 'avg_completion']

user_profiles_full = user_profiles_full.merge(engagement_metrics, on='adventurer_id', how='left')

# Add subscription data
sub_counts = df_subs.groupby('adventurer_id').agg(
    unique_publishers=('publisher_id', 'nunique'),
    total_subscriptions=('publisher_id', 'count')
).reset_index()

user_profiles_full = user_profiles_full.merge(sub_counts, on='adventurer_id', how='left')
user_profiles_full = user_profiles_full.fillna(0)

print(f"    Full user profiles: {len(user_profiles_full):,} users")
print(f"    Features: {user_profiles_full.shape[1]}")

user_profiles_full.to_csv(OUTPUT('user_profiles_with_clusters_svd.csv'), index=False)
print(f"✓ Saved user_profiles_with_clusters_svd.csv")

# Demographic summary by cluster
demographic_summary = user_profiles_full.groupby('cluster')[['age', 'total_views', 'avg_completion', 'total_subscriptions']].mean()
print("\nDemographic Summary by Cluster:")
print(demographic_summary.round(2))

demographic_summary.to_csv(OUTPUT('demographic_summary_by_cluster.csv'))
print(f"✓ Saved demographic_summary_by_cluster.csv")


# PART 11: SAVE EVALUATION METRICS

print("\n[11] Saving Evaluation Metrics")

evaluation_metrics = {
    'optimal_k': optimal_k,
    'silhouette_score': best_silhouette,
    'davies_bouldin_score': results_df.loc[results_df['k'] == optimal_k, 'davies_bouldin'].values[0],
    'calinski_harabasz_score': results_df.loc[results_df['k'] == optimal_k, 'calinski'].values[0],
    'n_users': len(user_ids),
    'n_content': len(content_ids),
    'n_latent_features': n_components,
    'variance_explained': svd.explained_variance_ratio_.sum(),
    'matrix_density': interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1])
}

eval_df = pd.DataFrame([evaluation_metrics])
eval_df.to_csv(OUTPUT('evaluation_metrics.csv'), index=False)
print(f"✓ Saved evaluation_metrics.csv")

print("\n" + "=" * 80)
print("CLUSTERING COMPLETE - SVD LATENT FEATURE APPROACH")
print("=" * 80)
print(f"\nKey Results:")
print(f"  ✓ Optimal clusters: {optimal_k}")
print(f"  ✓ Silhouette score: {best_silhouette:.3f}")
print(f"  ✓ Users clustered: {len(user_ids):,}")
print(f"  ✓ Latent dimensions: {n_components}")
print(f"  ✓ Variance explained: {svd.explained_variance_ratio_.sum():.1%}")

print(f"\nWhy This Approach Works:")
print(f"  ✓ No self-referential loops - we didn't define features manually")
print(f"  ✓ SVD discovered hidden patterns in user-content interactions")
print(f"  ✓ Clusters based on content taste similarity, not arbitrary metrics")
print(f"  ✓ Interpretable via genre preferences and engagement patterns")

print(f"\nGenerated Files:")
files = [
    'svd_variance_explained.png',
    'cluster_evaluation_svd.png',
    'cluster_visualization_svd.png',
    'genre_heatmap_by_cluster.png',
    'persona_insights_svd.csv',
    'genre_preferences_by_cluster.csv',
    'user_profiles_with_clusters_svd.csv',
    'demographic_summary_by_cluster.csv',
    'evaluation_metrics.csv'
]
for f in files:
    print(f"  - {f}")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Analyze persona_insights_svd.csv to understand each cluster")
print("  2. Look at genre_preferences_by_cluster.csv for content patterns")
print("  3. Create compelling persona names based on content preferences")
print("  4. Connect insights to churn prediction and recommendations")
print("=" * 80)