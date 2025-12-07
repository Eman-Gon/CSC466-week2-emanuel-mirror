"""
Week 7: Visualizations for SVD-Based Clustering
Steven Gonzalez
Creating compelling visualizations for persona insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

# Load SVD clustering results
user_profiles = pd.read_csv(P('user_profiles_with_clusters_svd.csv'))
persona_insights = pd.read_csv(P('persona_insights_svd.csv'))
genre_preferences = pd.read_csv(P('genre_preferences_by_cluster.csv'), index_col=0)

print(f"Loaded {len(user_profiles):,} users")
print(f"Found {len(persona_insights)} clusters from SVD approach")

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

# Define colors
num_clusters = len(persona_insights)
colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))


# 1. CLUSTER COMPARISON - KEY METRICS

print("\n[1] Creating cluster comparison dashboard")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Average views per user
ax1 = axes[0, 0]
bars1 = ax1.bar(persona_insights['cluster'], persona_insights['avg_views_per_user'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Cluster', fontweight='bold', fontsize=12)
ax1.set_ylabel('Avg Views per User', fontweight='bold', fontsize=12)
ax1.set_title('Engagement Level', fontsize=13, fontweight='bold', pad=10)
ax1.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Completion rate
ax2 = axes[0, 1]
bars2 = ax2.bar(persona_insights['cluster'], persona_insights['avg_completion_rate'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Cluster', fontweight='bold', fontsize=12)
ax2.set_ylabel('Avg Completion Rate', fontweight='bold', fontsize=12)
ax2.set_title('Content Completion', fontsize=13, fontweight='bold', pad=10)
ax2.grid(axis='y', alpha=0.3)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0%}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Content length preference
ax3 = axes[0, 2]
bars3 = ax3.bar(persona_insights['cluster'], persona_insights['avg_content_length_minutes'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Cluster', fontweight='bold', fontsize=12)
ax3.set_ylabel('Avg Content Length (min)', fontweight='bold', fontsize=12)
ax3.set_title('Content Length Preference', fontsize=13, fontweight='bold', pad=10)
ax3.grid(axis='y', alpha=0.3)
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Unique content diversity
ax4 = axes[1, 0]
bars4 = ax4.bar(persona_insights['cluster'], persona_insights['unique_content_per_user'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Cluster', fontweight='bold', fontsize=12)
ax4.set_ylabel('Unique Content per User', fontweight='bold', fontsize=12)
ax4.set_title('Content Diversity', fontsize=13, fontweight='bold', pad=10)
ax4.grid(axis='y', alpha=0.3)
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Churn rate
ax5 = axes[1, 1]
bars5 = ax5.bar(persona_insights['cluster'], persona_insights['churn_rate'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Cluster', fontweight='bold', fontsize=12)
ax5.set_ylabel('Churn Rate', fontweight='bold', fontsize=12)
ax5.set_title('Subscription Churn', fontsize=13, fontweight='bold', pad=10)
ax5.grid(axis='y', alpha=0.3)
for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0%}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Publisher diversity
ax6 = axes[1, 2]
bars6 = ax6.bar(persona_insights['cluster'], persona_insights['unique_publishers'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax6.set_xlabel('Cluster', fontweight='bold', fontsize=12)
ax6.set_ylabel('Unique Publishers', fontweight='bold', fontsize=12)
ax6.set_title('Publisher Diversity', fontsize=13, fontweight='bold', pad=10)
ax6.grid(axis='y', alpha=0.3)
for bar in bars6:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.suptitle('Persona Comparison Dashboard (SVD-Based Clustering)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(P('persona_dashboard_svd.png'), dpi=300, bbox_inches='tight')
print("✓ Saved persona_dashboard_svd.png")
plt.close()


# 2. CLUSTER SIZE DISTRIBUTION

print("\n[2] Creating cluster size distribution")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
sizes = persona_insights['size']
labels = [f'Cluster {i}\n({size:,} users)' for i, size in enumerate(sizes)]
explode = [0.05 if size < sizes.median() else 0 for size in sizes]

wedges, texts, autotexts = ax1.pie(sizes, 
                                     labels=labels,
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     explode=explode,
                                     startangle=90,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})

ax1.set_title('User Distribution Across Personas', fontsize=14, fontweight='bold', pad=15)

# Bar chart
ax2.bar(persona_insights['cluster'], persona_insights['size'], 
        color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Cluster', fontweight='bold', fontsize=12)
ax2.set_ylabel('Number of Users', fontweight='bold', fontsize=12)
ax2.set_title('Cluster Sizes', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)

for i, size in enumerate(persona_insights['size']):
    ax2.text(i, size, f'{size:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(P('cluster_distribution_svd.png'), dpi=300, bbox_inches='tight')
print("✓ Saved cluster_distribution_svd.png")
plt.close()


# 3. GENRE PREFERENCE HEATMAP (ENHANCED)

print("\n[3] Creating enhanced genre preference visualization")

# Get top 20 genres overall
top_genres = genre_preferences.sum().sort_values(ascending=False).head(20).index
genre_subset = genre_preferences[top_genres]

fig, ax = plt.subplots(figsize=(16, 8))

# Normalize by row (cluster) to show relative preferences
genre_normalized = genre_subset.div(genre_subset.sum(axis=1), axis=0)

sns.heatmap(genre_normalized, 
            annot=True, 
            fmt='.2%',
            cmap='RdYlGn',
            cbar_kws={'label': 'Relative Preference Within Cluster'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax)

ax.set_xlabel('Genre ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
ax.set_title('Genre Preference Patterns by Cluster (Normalized)', 
            fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(P('genre_preferences_normalized.png'), dpi=300, bbox_inches='tight')
print("✓ Saved genre_preferences_normalized.png")
plt.close()


# 4. ENGAGEMENT VS CHURN SCATTER

print("\n[4] Creating engagement vs churn analysis")

fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(persona_insights['avg_views_per_user'], 
                     persona_insights['churn_rate'],
                     s=persona_insights['size'] / 10,  # Size based on cluster size
                     c=persona_insights['cluster'],
                     cmap='tab10',
                     alpha=0.7,
                     edgecolors='black',
                     linewidth=2)

# Add cluster labels
for idx, row in persona_insights.iterrows():
    ax.annotate(f"Cluster {row['cluster']}\n({row['size']:,} users)", 
                (row['avg_views_per_user'], row['churn_rate']),
                fontsize=10, 
                fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

ax.set_xlabel('Average Views per User', fontsize=12, fontweight='bold')
ax.set_ylabel('Churn Rate', fontsize=12, fontweight='bold')
ax.set_title('Engagement vs Churn by Persona', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)

# Add quadrant lines
ax.axvline(persona_insights['avg_views_per_user'].median(), 
          color='red', linestyle='--', alpha=0.5, label='Median Engagement')
ax.axhline(persona_insights['churn_rate'].median(), 
          color='blue', linestyle='--', alpha=0.5, label='Median Churn')
ax.legend()

plt.tight_layout()
plt.savefig(P('engagement_vs_churn.png'), dpi=300, bbox_inches='tight')
print("✓ Saved engagement_vs_churn.png")
plt.close()


# 5. COMPLETION VS CONTENT LENGTH

print("\n[5] Creating completion vs content length analysis")

fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(persona_insights['avg_content_length_minutes'], 
                     persona_insights['avg_completion_rate'],
                     s=persona_insights['size'] / 10,
                     c=persona_insights['cluster'],
                     cmap='tab10',
                     alpha=0.7,
                     edgecolors='black',
                     linewidth=2)

# Add cluster labels
for idx, row in persona_insights.iterrows():
    ax.annotate(f"Cluster {row['cluster']}", 
                (row['avg_content_length_minutes'], row['avg_completion_rate']),
                fontsize=11, 
                fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.set_xlabel('Average Content Length (minutes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Completion Rate', fontsize=12, fontweight='bold')
ax.set_title('Content Length Preference vs Completion Behavior', 
            fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(P('completion_vs_length.png'), dpi=300, bbox_inches='tight')
print("✓ Saved completion_vs_length.png")
plt.close()


# 6. RADAR CHART - PERSONA PROFILES

print("\n[6] Creating radar chart for persona profiles")

from math import pi

# Select key metrics for radar chart
metrics = ['avg_views_per_user', 'avg_completion_rate', 'unique_content_per_user', 
           'avg_content_length_minutes', 'churn_rate']
metric_labels = ['Engagement', 'Completion', 'Diversity', 'Content Length', 'Churn']

# Normalize metrics to 0-1 scale
normalized_data = persona_insights[metrics].copy()
for col in metrics:
    normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                           (normalized_data[col].max() - normalized_data[col].min())

# Invert churn rate (higher is worse)
normalized_data['churn_rate'] = 1 - normalized_data['churn_rate']

# Create radar chart
angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))]
angles += angles[:1]

fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

for idx, cluster_id in enumerate(persona_insights['cluster']):
    ax = axes[idx]
    
    values = normalized_data.iloc[cluster_id].values.tolist()
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[cluster_id], label=f'Cluster {cluster_id}')
    ax.fill(angles, values, alpha=0.25, color=colors[cluster_id])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'Cluster {cluster_id}\n({persona_insights.iloc[cluster_id]["size"]:,} users)', 
                fontsize=12, fontweight='bold', pad=20)
    ax.grid(True)

# Remove extra subplot if odd number of clusters
if len(persona_insights) < len(axes):
    for idx in range(len(persona_insights), len(axes)):
        fig.delaxes(axes[idx])

plt.suptitle('Persona Profile Comparison (Radar Charts)', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(P('persona_radar_charts.png'), dpi=300, bbox_inches='tight')
print("✓ Saved persona_radar_charts.png")
plt.close()


# SUMMARY

print("\n" + "=" * 70)
print("VISUALIZATIONS COMPLETE (SVD Approach)")
print("=" * 70)
print("\nGenerated files:")
print("  - persona_dashboard_svd.png (6 key metrics)")
print("  - cluster_distribution_svd.png (size distribution)")
print("  - genre_preferences_normalized.png (content preferences)")
print("  - engagement_vs_churn.png (risk analysis)")
print("  - completion_vs_length.png (behavior patterns)")
print("  - persona_radar_charts.png (comprehensive profiles)")
print("\nApproach: SVD latent features → clustering → interpretation")
print("Key insight: Clusters based on CONTENT PREFERENCES, not demographics")
print("=" * 70)