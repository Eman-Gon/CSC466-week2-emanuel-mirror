import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

# This will generate the charts for your 2-page report

print("Generating visualizations for report...")

# Load evaluation results (you'll need to run evaluate_all_methods.py first)
# For now, let's use placeholder data that you'll update

# Chart 1: Method Performance Comparison
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Random\nBaseline', 'Global\nHeuristic', 'Collaborative\nFiltering', 
           'Content-\nBased', 'Hybrid\n(60/40)']
precision_values = [0.02, 0.30, 0.35, 0.32, 0.42]  # UPDATE THESE WITH REAL VALUES

colors = ['#ff4444', '#ffa500', '#4169e1', '#32cd32', '#9370db']
bars = ax.bar(methods, precision_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.axhline(y=0.02, color='red', linestyle='--', linewidth=2, label='Random Baseline', alpha=0.5)
ax.set_ylabel('Precision@2', fontsize=13, fontweight='bold')
ax.set_title('Recommender System Performance Comparison', fontsize=15, fontweight='bold', pad=20)
ax.set_ylim(0, max(precision_values) * 1.2)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(P('method_comparison.png'), dpi=300, bbox_inches='tight')
print("✓ Saved method_comparison.png")

# Chart 2: User Segmentation Analysis
fig, ax = plt.subplots(figsize=(12, 6))

user_segments = ['Cold Start\n(≤3 views)', 'Warm\n(4-10 views)', 'Hot\n(>10 views)']
x = np.arange(len(user_segments))
width = 0.2

# Sample data - UPDATE THESE with real values from your analysis
heuristic_perf = [0.35, 0.28, 0.22]  # Heuristic works best for cold start
collaborative_perf = [0.08, 0.32, 0.48]  # Collaborative works best for hot users
content_perf = [0.25, 0.30, 0.35]  # Content-based is middle ground
hybrid_perf = [0.30, 0.38, 0.45]  # Hybrid balances all

bars1 = ax.bar(x - 1.5*width, heuristic_perf, width, label='Heuristic', color='#ffa500', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, collaborative_perf, width, label='Collaborative', color='#4169e1', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, content_perf, width, label='Content-Based', color='#32cd32', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, hybrid_perf, width, label='Hybrid', color='#9370db', alpha=0.8)

ax.set_ylabel('Precision@2', fontsize=13, fontweight='bold')
ax.set_title('Performance by User Activity Level', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(user_segments, fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 0.6)

plt.tight_layout()
plt.savefig(P('user_segmentation.png'), dpi=300, bbox_inches='tight')
print("✓ Saved user_segmentation.png")

# Chart 3: Diversity Analysis
fig, ax = plt.subplots(figsize=(10, 6))

methods_short = ['Heuristic', 'Collaborative', 'Content', 'Hybrid']
unique_items = [25, 15, 22, 18]  # UPDATE with real values
total_items = 37  # For publisher wn32

coverage_pct = [u/total_items*100 for u in unique_items]

bars = ax.bar(methods_short, coverage_pct, color=['#ffa500', '#4169e1', '#32cd32', '#9370db'], 
              alpha=0.7, edgecolor='black', linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Catalog Coverage (%)', fontsize=13, fontweight='bold')
ax.set_title('Recommendation Diversity: Unique Items Recommended', fontsize=15, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(P('diversity_analysis.png'), dpi=300, bbox_inches='tight')
print("✓ Saved diversity_analysis.png")

print("\n✅ All visualizations generated!")
print("\nTo update with real data:")
print("1. Run: python evaluate_all_methods.py")
print("2. Copy the actual precision values into this script")
print("3. Run this script again: python create_visualizations.py")