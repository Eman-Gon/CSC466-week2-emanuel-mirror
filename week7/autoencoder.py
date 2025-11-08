# Autoencoder.py

"""
Week 7: Autoencoder-Based Clustering
Emanuel Gonzalez
Inspired by Connor's presentation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    HAS_KERAS = True
except ImportError:
    print("⚠️  TensorFlow not available. Install with: pip install tensorflow")
    HAS_KERAS = False

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("AUTOENCODER-BASED CLUSTERING")
print("="*60)

if not HAS_KERAS:
    print("\n❌ TensorFlow is required for autoencoder. Exiting...")
    exit(1)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1] Loading preprocessed data...")
user_profiles = pd.read_csv(P('user_profiles_with_clusters.csv'))

# Select features
features = [
    'total_watch_time',
    'avg_watch_time',
    'num_views',
    'unique_content',
    'avg_completion_rate',
    'num_subscriptions',
    'num_publishers',
    'num_churns',
    'genre_diversity',
    'lang_diversity',
    'age'
]

X = user_profiles[features].values
print(f"    Feature matrix: {X.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# 2. BUILD AUTOENCODER
# ============================================
print("\n[2] Building autoencoder...")

input_dim = X_scaled.shape[1]
encoding_dim = 8  # Compressed representation size

# Build encoder
encoder_input = keras.Input(shape=(input_dim,))
encoded = layers.Dense(32, activation='relu')(encoder_input)
encoded = layers.Dense(16, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(encoded)

# Build decoder
decoded = layers.Dense(16, activation='relu')(encoded)
decoded = layers.Dense(32, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

# Full autoencoder
autoencoder = keras.Model(encoder_input, decoded)

# Encoder model (for extracting embeddings)
encoder = keras.Model(encoder_input, encoded)

print(f"    Autoencoder architecture:")
print(f"    Input: {input_dim} → 32 → 16 → {encoding_dim} → 16 → 32 → {input_dim}")

# ============================================
# 3. TRAIN AUTOENCODER
# ============================================
print("\n[3] Training autoencoder...")

autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=100,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=0
)

print(f"    Final training loss: {history.history['loss'][-1]:.4f}")
print(f"    Final validation loss: {history.history['val_loss'][-1]:.4f}")

# Plot training history
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
ax.set_ylabel('Loss (MSE)', fontweight='bold', fontsize=12)
ax.set_title('Autoencoder Training History', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(P('autoencoder_training.png'), dpi=300, bbox_inches='tight')
print("    ✓ Saved autoencoder_training.png")
plt.close()

# ============================================
# 4. EXTRACT EMBEDDINGS
# ============================================
print("\n[4] Extracting embeddings...")

embeddings = encoder.predict(X_scaled, verbose=0)
print(f"    Embeddings shape: {embeddings.shape}")

# ============================================
# 5. CLUSTER ON EMBEDDINGS
# ============================================
print("\n[5] Clustering on autoencoder embeddings...")

results = []
K_range = range(3, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(embeddings)
    
    silhouette = silhouette_score(embeddings, labels)
    inertia = kmeans.inertia_
    
    results.append({
        'k': k,
        'silhouette': silhouette,
        'inertia': inertia
    })
    
    print(f"    k={k}: Silhouette={silhouette:.3f}, Inertia={inertia:.0f}")

results_df = pd.DataFrame(results)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(results_df['k'], results_df['inertia'], 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Inertia', fontsize=12, fontweight='bold')
ax1.set_title('Elbow Plot (Autoencoder)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(results_df['k'], results_df['silhouette'], 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax2.set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(P('autoencoder_clustering_eval.png'), dpi=300, bbox_inches='tight')
print("\n    ✓ Saved autoencoder_clustering_eval.png")
plt.close()

# Pick optimal k
optimal_k = int(results_df.loc[results_df['silhouette'].idxmax(), 'k'])
best_silhouette = results_df['silhouette'].max()
print(f"\n    🎯 Optimal k: {optimal_k} (Silhouette: {best_silhouette:.3f})")

# ============================================
# 6. FINAL CLUSTERING
# ============================================
print(f"\n[6] Final clustering with k={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
user_profiles['ae_cluster'] = kmeans_final.fit_predict(embeddings)

print(f"    Cluster distribution:")
for cluster_id in sorted(user_profiles['ae_cluster'].unique()):
    count = (user_profiles['ae_cluster'] == cluster_id).sum()
    pct = count / len(user_profiles) * 100
    print(f"      Cluster {cluster_id}: {count:,} users ({pct:.1f}%)")

# ============================================
# 7. VISUALIZE EMBEDDINGS
# ============================================
print("\n[7] Visualizing embeddings...")

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=user_profiles['ae_cluster'], 
                     cmap='viridis', 
                     alpha=0.6, 
                     s=50,
                     edgecolors='black',
                     linewidth=0.5)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
          fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
          fontsize=12, fontweight='bold')
plt.title(f'Autoencoder Embeddings (k={optimal_k} clusters)', 
         fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(P('autoencoder_visualization.png'), dpi=300, bbox_inches='tight')
print("    ✓ Saved autoencoder_visualization.png")
plt.close()

# ============================================
# 8. COMPARE WITH ORIGINAL CLUSTERING
# ============================================
print("\n[8] Comparing autoencoder vs original clustering...")

# Create confusion matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(user_profiles['cluster'], user_profiles['ae_cluster'])
nmi = normalized_mutual_info_score(user_profiles['cluster'], user_profiles['ae_cluster'])

print(f"    Adjusted Rand Index: {ari:.3f}")
print(f"    Normalized Mutual Information: {nmi:.3f}")

# Cross-tabulation
cross_tab = pd.crosstab(user_profiles['cluster'], user_profiles['ae_cluster'], 
                       margins=True, margins_name='Total')
print(f"\n    Cross-tabulation:")
print(cross_tab)

# ============================================
# 9. SAVE RESULTS
# ============================================
print("\n[9] Saving results...")

user_profiles.to_csv(P('user_profiles_with_ae_clusters.csv'), index=False)
print("    ✓ Saved user_profiles_with_ae_clusters.csv")

# Save autoencoder persona insights
ae_persona_insights = []
for cluster_id in sorted(user_profiles['ae_cluster'].unique()):
    cluster_data = user_profiles[user_profiles['ae_cluster'] == cluster_id]
    
    insight = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        'avg_views': cluster_data['num_views'].mean(),
        'avg_completion': cluster_data['avg_completion_rate'].mean(),
        'avg_churns': cluster_data['num_churns'].mean(),
        'avg_subs': cluster_data['num_subscriptions'].mean(),
        'avg_age': cluster_data['age'].mean()
    }
    ae_persona_insights.append(insight)

ae_persona_df = pd.DataFrame(ae_persona_insights)
ae_persona_df.to_csv(P('ae_persona_insights.csv'), index=False)
print("    ✓ Saved ae_persona_insights.csv")

print("\n" + "="*60)
print("AUTOENCODER CLUSTERING COMPLETE! 🎉")
print("="*60)
print("\nGenerated files:")
print("  - autoencoder_training.png")
print("  - autoencoder_clustering_eval.png")
print("  - autoencoder_visualization.png")
print("  - user_profiles_with_ae_clusters.csv")
print("  - ae_persona_insights.csv")
print("\nKey findings:")
print(f"  - Original K-Means silhouette: 0.267 (k=7)")
print(f"  - Autoencoder silhouette: {best_silhouette:.3f} (k={optimal_k})")
print(f"  - Clustering agreement (ARI): {ari:.3f}")
print(f"  - Clustering agreement (NMI): {nmi:.3f}")