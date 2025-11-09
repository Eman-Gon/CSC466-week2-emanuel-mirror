# create_visuals.py
"""
Week 7: Enhanced Visualizations
Emanuel Gonzalez
Inspired by class presentations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

# Load your results
user_profiles = pd.read_csv(P('user_profiles_with_clusters.csv'))
persona_insights = pd.read_csv(P('persona_insights.csv'))


# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

# Define colors for 7 clusters
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']


print("\n[1] Creating bar chart comparisons")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Average Views by Cluster
ax1 = axes[0, 0]
bars1 = ax1.bar(persona_insights['cluster'], persona_insights['avg_views'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax1.set_ylabel('Average Views', fontweight='bold', fontsize=12)
ax1.set_title('Average Content Views by Persona', fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(range(7))

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# 2. Average Age by Cluster
ax2 = axes[0, 1]
bars2 = ax2.bar(persona_insights['cluster'], persona_insights['avg_age'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax2.set_ylabel('Average Age (years)', fontweight='bold', fontsize=12)
ax2.set_title('Average Age by Persona', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(range(7))

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. Average Churns by Cluster
ax3 = axes[1, 0]
bars3 = ax3.bar(persona_insights['cluster'], persona_insights['avg_churns'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax3.set_ylabel('Average Churns', fontweight='bold', fontsize=12)
ax3.set_title('Churn Behavior by Persona', fontsize=14, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticks(range(7))

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# 4. Completion Rate by Cluster
ax4 = axes[1, 1]
bars4 = ax4.bar(persona_insights['cluster'], persona_insights['avg_completion'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Persona Cluster', fontweight='bold', fontsize=12)
ax4.set_ylabel('Completion Rate', fontweight='bold', fontsize=12)
ax4.set_title('Content Completion by Persona', fontsize=14, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3)
ax4.set_xticks(range(7))

for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(P('persona_comparison_bars.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_comparison_bars.png")
plt.close()

print("\n[2] Creating cluster distribution pie chart")

fig, ax = plt.subplots(figsize=(10, 8))

sizes = persona_insights['size']
labels = [f'Cluster {i}\n({size:,} users)' for i, size in enumerate(sizes)]
explode = [0.05 if i in [4, 5] else 0 for i in range(len(sizes))]

wedges, texts, autotexts = ax.pie(sizes, 
                                    labels=labels,
                                    autopct='%1.1f%%',
                                    colors=colors,
                                    explode=explode,
                                    startangle=90,
                                    textprops={'fontsize': 10, 'fontweight': 'bold'})

ax.set_title('User Distribution Across Personas', fontsize=16, fontweight='bold', pad=20)
plt.savefig(P('persona_distribution.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_distribution.png")
plt.close()

print("\n[3] Creating feature heatmap")

fig, ax = plt.subplots(figsize=(10, 8))

heatmap_data = persona_insights[['cluster', 'avg_views', 'avg_completion', 'avg_churns', 'avg_subs', 'avg_age']].set_index('cluster')
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
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Persona Feature Heatmap (Normalized)', fontsize=14, fontweight='bold', pad=15)
ax.set_yticklabels(['Avg Views', 'Completion', 'Churns', 'Subs', 'Age'], rotation=0)

plt.tight_layout()
plt.savefig(P('persona_heatmap.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_heatmap.png")
plt.close()

print("\n[4] Creating individual persona cards")

# Define persona names based on analysis
persona_names = {
    0: "Ghost Users",
    1: "Active Explorers",
    2: "Subscription Hoppers",
    3: "Committed Finishers",
    4: "Power Users",
    5: "Serial Churners",
    6: "Casual Samplers"
}

persona_descriptions = {
    0: "Signed up but never engaged",
    1: "Engaged viewers exploring content",
    2: "Subscribe often, cancel frequently",
    3: "Watch less but finish content",
    4: "Heavy consumers, younger audience",
    5: "Constant subscribe/cancel cycle",
    6: "Try content but don't commit"
}

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for idx, (cluster_id, row) in enumerate(persona_insights.iterrows()):
    ax = axes[idx]
    
    # Create card background
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9, persona_names[cluster_id], 
            ha='center', va='top', fontsize=16, fontweight='bold', color=colors[idx])
    
    # Description
    ax.text(5, 8, persona_descriptions[cluster_id],
            ha='center', va='top', fontsize=10, style='italic', wrap=True)
    
    # Stats
    stats_text = f"""
    Size: {row['size']:,} users ({row['size']/25770*100:.1f}%)
    
    Avg Views: {row['avg_views']:.1f}
    Completion: {row['avg_completion']:.0%}
    Churns: {row['avg_churns']:.1f}
    Subscriptions: {row['avg_subs']:.1f}
    Age: {row['avg_age']:.0f} years
    """
    
    ax.text(5, 5.5, stats_text.strip(),
            ha='center', va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.2))
    
    # Border
    rect = plt.Rectangle((0.2, 0.2), 9.6, 9.6, fill=False, 
                         edgecolor=colors[idx], linewidth=3)
    ax.add_patch(rect)

# Hide last empty subplot
axes[7].axis('off')

plt.suptitle('Adventurer Personas', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(P('persona_cards.png'), dpi=300, bbox_inches='tight')
print("     Saved persona_cards.png")
plt.close()

print("\nGenerated files:")
print("  - persona_comparison_bars.png")
print("  - persona_distribution.png")
print("  - persona_heatmap.png")
print("  - persona_cards.png")