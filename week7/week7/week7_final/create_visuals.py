# create_visuals_svd.py
"""
Week 7: Visualizations for SVD-Based Clustering
Steven Gonzalez
Using latent features approach as recommended by Allen
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

print(f"Loaded {len(user_profiles):,} users")
print(f"Found {len(persona_insights)} clusters from SVD approach")

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

# Define colors
num_clusters = len(persona_insights)
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#ee5a6f'][:num_clusters]

print("\n[1] Creating bar chart comparisons")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Average Views by Cluster (context only)
ax1 = axes[0, 0]
bars1 = ax1.bar(persona_insights['cluster'], persona_insights['avg_views'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax1.set_ylabel('Average Views', fontweight='bold', fontsize=12)
ax1.set_title('Average Views by Persona (Context)', fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(range(num_clusters))

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# 2. Completion Consistency
ax2 = axes[0, 1]
bars2 = ax2.bar(persona_insights['cluster'], persona_insights['completion_consistency'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax2.set_ylabel('Completion Consistency', fontweight='bold', fontsize=12)
ax2.set_title('Completion Consistency by Persona', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(range(num_clusters))

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. Churn Rate
ax3 = axes[1, 0]
bars3 = ax3.bar(persona_insights['cluster'], persona_insights['churn_rate'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax3.set_ylabel('Churn Rate', fontweight='bold', fontsize=12)
ax3.set_title('Churn Rate by Persona', fontsize=14, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticks(range(num_clusters))

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# 4. Genre Concentration
ax4 = axes[1, 1]
bars4 = ax4.bar(persona_insights['cluster'], persona_insights['genre_concentration'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax4.set_ylabel('Genre Concentration (Gini)', fontweight='bold', fontsize=12)
ax4.set_title('Genre Concentration by Persona', fontsize=14, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3)
ax4.set_xticks(range(num_clusters))

for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(P('persona_comparison_bars_svd.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_comparison_bars_svd.png")
plt.close()

print("\n[2] Creating cluster distribution pie chart")

fig, ax = plt.subplots(figsize=(10, 8))

sizes = persona_insights['size']
labels = [f'Cluster {i}\n({size:,} users)' for i, size in enumerate(sizes)]
explode = [0.05 if size < 3000 else 0 for size in sizes]

wedges, texts, autotexts = ax.pie(sizes, 
                                    labels=labels,
                                    autopct='%1.1f%%',
                                    colors=colors,
                                    explode=explode,
                                    startangle=90,
                                    textprops={'fontsize': 10, 'fontweight': 'bold'})

ax.set_title('User Distribution Across Personas (SVD Clustering)', fontsize=16, fontweight='bold', pad=20)
plt.savefig(P('persona_distribution_svd.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_distribution_svd.png")
plt.close()

print("\n[3] Creating behavioral feature heatmap")

fig, ax = plt.subplots(figsize=(12, 8))

heatmap_data = persona_insights[[
    'cluster', 
    'completion_consistency',
    'finisher_ratio',
    'genre_concentration',
    'churn_rate',
    'publisher_concentration'
]].set_index('cluster')

heatmap_data_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

sns.heatmap(heatmap_data_norm.T, 
            annot=True, 
            fmt='.2f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Normalized Value'},
            linewidths=0.5,
            linecolor='black',
            ax=ax)

ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Behavioral Feature (for interpretation)', fontsize=12, fontweight='bold')
ax.set_title('Persona Feature Heatmap - Clustered on SVD, Described with Features', fontsize=14, fontweight='bold', pad=15)
ax.set_yticklabels([
    'Completion Consistency',
    'Finisher Ratio', 
    'Genre Concentration',
    'Churn Rate',
    'Publisher Concentration'
], rotation=0)

plt.tight_layout()
plt.savefig(P('persona_heatmap_svd.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_heatmap_svd.png")
plt.close()

print("\n[4] Creating persona cards")

# Generate names based on behavioral patterns
persona_names = {}
persona_descriptions = {}

for i in range(num_clusters):
    row = persona_insights.iloc[i]
    
    if row['avg_views'] == 0:
        name = "Ghost Users"
        desc = "Subscribed but never engaged"
    elif row['finisher_ratio'] > 0.3:
        name = "Committed Subscribers"
        desc = "High completion and consistency"
    elif row['churn_rate'] > 0.7:
        name = "High Churners"
        desc = "Frequent subscribe and cancel"
    elif row['completion_consistency'] > 0.85:
        name = "Consistent Samplers"
        desc = "Predictable but low commitment"
    elif row['genre_concentration'] > 0.5:
        name = "Genre Specialists"
        desc = "Focus on specific content types"
    else:
        name = f"Persona {i}"
        desc = "Mixed behavioral pattern"
    
    persona_names[i] = name
    persona_descriptions[i] = desc

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for idx, (cluster_id, row) in enumerate(persona_insights.iterrows()):
    ax = axes[idx]
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9, persona_names[cluster_id], 
            ha='center', va='top', fontsize=16, fontweight='bold', color=colors[idx])
    
    # Description
    ax.text(5, 8, persona_descriptions[cluster_id],
            ha='center', va='top', fontsize=10, style='italic')
    
    # Stats
    stats_text = f"""
    Size: {row['size']:,} users ({row['size']/len(user_profiles)*100:.1f}%)
    
    BEHAVIORAL CHARACTERISTICS:
    Completion Consistency: {row['completion_consistency']:.2f}
    Finisher Ratio: {row['finisher_ratio']:.0%}
    Genre Concentration: {row['genre_concentration']:.2f}
    Churn Rate: {row['churn_rate']:.0%}
    
    CONTEXT:
    Avg Views: {row['avg_views']:.1f}
    Age: {row['avg_age']:.0f} years
    """
    
    ax.text(5, 5.5, stats_text.strip(),
            ha='center', va='top', fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.2))
    
    # Border
    rect = plt.Rectangle((0.2, 0.2), 9.6, 9.6, fill=False, 
                         edgecolor=colors[idx], linewidth=3)
    ax.add_patch(rect)

# Hide unused subplots
for i in range(num_clusters, len(axes)):
    axes[i].axis('off')

plt.suptitle('Adventurer Personas (SVD-Based Clustering)', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(P('persona_cards_svd.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_cards_svd.png")
plt.close()

print("\n" + "=" * 60)
print("VISUALIZATIONS COMPLETE (SVD Approach)")
print("=" * 60)
print("\nGenerated files:")
print("  - persona_comparison_bars_svd.png")
print("  - persona_distribution_svd.png")
print("  - persona_heatmap_svd.png")
print("  - persona_cards_svd.png")
print("\nApproach: Clustered on SVD latent features,")
print("          Described with behavioral features")
print("=" * 60)