# create_visuals_behavioral.py
"""
Week 7: Visualizations for Behavioral Clustering
Steven Gonzalez
Using behavioral patterns approach (HOW users engage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

# Load behavioral clustering results
user_profiles = pd.read_csv(P('user_profiles_with_clusters_behavioral.csv'))
persona_insights = pd.read_csv(P('persona_insights_behavioral.csv'))

print(f"Loaded {len(user_profiles):,} users")
print(f"Found {len(persona_insights)} clusters from behavioral approach")

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

# Define colors
num_clusters = len(persona_insights)
colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

print("\n[1] Creating behavioral feature comparison")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Completion Consistency
ax1 = axes[0, 0]
bars1 = ax1.bar(persona_insights['cluster'], persona_insights['completion_consistency'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax1.set_ylabel('Completion Consistency', fontweight='bold', fontsize=12)
ax1.set_title('Completion Consistency by Persona', fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(range(num_clusters))
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 2. Finisher Ratio
ax2 = axes[0, 1]
bars2 = ax2.bar(persona_insights['cluster'], persona_insights['finisher_ratio'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax2.set_ylabel('Finisher Ratio', fontweight='bold', fontsize=12)
ax2.set_title('Finisher Ratio by Persona', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(range(num_clusters))
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0%}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. Genre Concentration
ax3 = axes[1, 0]
bars3 = ax3.bar(persona_insights['cluster'], persona_insights['genre_concentration'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax3.set_ylabel('Genre Concentration (Gini)', fontweight='bold', fontsize=12)
ax3.set_title('Genre Concentration by Persona', fontsize=14, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticks(range(num_clusters))
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 4. Churn Rate
ax4 = axes[1, 1]
bars4 = ax4.bar(persona_insights['cluster'], persona_insights['churn_rate'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax4.set_ylabel('Churn Rate', fontweight='bold', fontsize=12)
ax4.set_title('Churn Rate by Persona', fontsize=14, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3)
ax4.set_xticks(range(num_clusters))
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0%}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(P('persona_comparison_behavioral.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_comparison_behavioral.png")
plt.close()

print("\n[2] Creating cluster distribution pie chart")

fig, ax = plt.subplots(figsize=(10, 8))

sizes = persona_insights['size']
labels = [f'Cluster {i}\n({size:,} users)' for i, size in enumerate(sizes)]
explode = [0.05 if size < sizes.median() else 0 for size in sizes]

wedges, texts, autotexts = ax.pie(sizes, 
                                    labels=labels,
                                    autopct='%1.1f%%',
                                    colors=colors,
                                    explode=explode,
                                    startangle=90,
                                    textprops={'fontsize': 10, 'fontweight': 'bold'})

ax.set_title('User Distribution Across Personas (Behavioral Clustering)', fontsize=16, fontweight='bold', pad=20)
plt.savefig(P('persona_distribution_behavioral.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_distribution_behavioral.png")
plt.close()

print("\n[3] Creating behavioral feature heatmap")

fig, ax = plt.subplots(figsize=(12, 8))

heatmap_data = persona_insights[[
    'cluster', 
    'completion_consistency',
    'finisher_ratio',
    'genre_concentration',
    'churn_rate',
    'viewing_consistency',
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
ax.set_ylabel('Behavioral Feature', fontsize=12, fontweight='bold')
ax.set_title('Persona Behavioral Patterns Heatmap', fontsize=14, fontweight='bold', pad=15)
ax.set_yticklabels([
    'Completion Consistency',
    'Finisher Ratio', 
    'Genre Concentration',
    'Churn Rate',
    'Viewing Consistency',
    'Publisher Concentration'
], rotation=0)

plt.tight_layout()
plt.savefig(P('persona_heatmap_behavioral.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_heatmap_behavioral.png")
plt.close()

print("\n[4] Creating context comparison (volume metrics)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Average Views (context)
ax1 = axes[0]
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
            f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Average Watch Hours (context)
ax2 = axes[1]
bars2 = ax2.bar(persona_insights['cluster'], persona_insights['avg_watch_hours'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax2.set_ylabel('Average Watch Hours', fontweight='bold', fontsize=12)
ax2.set_title('Average Watch Hours by Persona (Context)', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(range(num_clusters))
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}h', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(P('persona_context_comparison.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_context_comparison.png")
plt.close()

print("\n" + "=" * 60)
print("VISUALIZATIONS COMPLETE (Behavioral Approach)")
print("=" * 60)
print("\nGenerated files:")
print("  - persona_comparison_behavioral.png")
print("  - persona_distribution_behavioral.png")
print("  - persona_heatmap_behavioral.png")
print("  - persona_context_comparison.png")
print("\nApproach: Clustered on behavioral patterns (HOW),")
print("          Described with context features (WHAT/volume)")
print("=" * 60)