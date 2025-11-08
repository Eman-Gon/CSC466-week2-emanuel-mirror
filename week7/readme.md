# Week 7: User Personas - Complete Setup Guide

## 📦 Files You Have

✅ **personas.py** - Main K-Means clustering (already run - produced your 7 clusters)
✅ **create_visualizations.py** - Creates amazing bar charts, pie charts, heatmaps, persona cards
✅ **autoencoder_clustering.py** - Autoencoder-based clustering for validation
✅ **writeup.md** - Template report (fill in autoencoder results)

## 🚀 Quick Start

### Step 1: Create Enhanced Visualizations (5 minutes)

```bash
cd week7
python3 create_visualizations.py
```

This creates:
- `persona_comparison_bars.png` - **Main visualization like Ryan's presentation**
- `persona_distribution.png` - Pie chart of cluster sizes
- `persona_heatmap.png` - Feature comparison heatmap  
- `persona_cards.png` - Individual persona profile cards

### Step 2: Run Autoencoder Clustering (5-10 minutes)

```bash
# First install TensorFlow if needed
pip install tensorflow

# Then run
python3 autoencoder_clustering.py
```

This creates:
- `autoencoder_training.png` - Training loss curves
- `autoencoder_clustering_eval.png` - Silhouette scores for different k
- `autoencoder_visualization.png` - PCA of embedding space
- `ae_persona_insights.csv` - Autoencoder cluster statistics

### Step 3: Update Writeup (30 minutes)

Open `writeup.md` and:

1. Fill in autoencoder silhouette score (from terminal output)
2. Fill in Adjusted Rand Index (from terminal output)
3. Add any geographic insights if you analyze kingdoms
4. Review and personalize the persona descriptions
5. Add your own insights and observations

### Step 4: Final Polish (15 minutes)

```bash
# Check all files are there
ls -la *.png *.csv *.md *.py

# Expected files:
# - personas.py
# - create_visualizations.py
# - autoencoder_clustering.py
# - writeup.md
# - All .png visualizations
# - All .csv data files
```

## 📊 What Each Persona Means

Based on your clustering results:

| Cluster | Name | Key Trait | Size |
|---------|------|-----------|------|
| 0 | Ghost Users | Never engaged | 8.8% |
| 1 | Active Explorers | Engaged, diverse | 21.0% |
| 2 | Subscription Hoppers | High churn cycle | 22.9% |
| 3 | Committed Finishers | 66% completion | 18.0% |
| 4 | Power Users | 15+ views, young | 4.9% |
| 5 | Serial Churners | 9.3 churns | 4.8% |
| 6 | Casual Samplers | Low completion | 19.6% |

## 🎯 Key Insights for Your Presentation

**Best Clustering:**
- Your silhouette: **0.267** (better than class example at 0.158!)
- 7 clusters (not too many like 77!)
- Balanced sizes (no giant "trash" cluster)

**Churn Connection:**
- Serial Churners (Cluster 5) have 9.27 average churns - highest risk
- Committed Finishers (Cluster 3) have 1.67 churns - most loyal
- Can target interventions by persona

**Recommendation Connection:**
- Power Users need discovery → recommend new/niche content
- Casual Samplers need safety → recommend popular content
- Active Explorers need variety → cross-genre recommendations

## 📸 Visualization Tips (From Class Examples)

**What Lucas praised in Ryan's presentation:**
- ✅ Named personas ("Horror Fans Living in Fear")
- ✅ Connected to geography (Wastelands Kingdom)
- ✅ Clear bar charts comparing personas
- ✅ Mix of rigor + storytelling

**What to avoid:**
- ❌ Too many clusters (77 was too many)
- ❌ Just showing methodology without interpretation
- ❌ No persona names or descriptions
- ❌ Statistics without story

## 🎬 Optional: Geographic Analysis

If you want to replicate Connor's "Wastelands Kingdom" finding:

```python
# Add to analysis
import pandas as pd

user_profiles = pd.read_csv('user_profiles_with_clusters.csv')

# Load adventurer metadata with region info
df_adventurers = pd.read_parquet('adventurer_metadata.parquet')

# If there's a kingdom/region column:
merged = user_profiles.merge(df_adventurers[['adventurer_id', 'kingdom']], on='adventurer_id')

# Analyze cluster-kingdom relationship
for cluster in range(7):
    cluster_data = merged[merged['cluster'] == cluster]
    print(f"\nCluster {cluster} top kingdoms:")
    print(cluster_data['kingdom'].value_counts().head(3))
```

## 🏆 Submission Checklist

Before submitting to GitHub:

- [ ] `personas.py` - Main clustering code (already done ✅)
- [ ] `writeup.md` - Complete 2-3 page report
- [ ] All visualization PNGs
- [ ] `cluster_summary.csv` and `persona_insights.csv`
- [ ] (Optional) `autoencoder_clustering.py` for extra credit
- [ ] (Optional) Geographic analysis

## 💡 Pro Tips

1. **Lead with the story** - Your personas have great names, use them!
2. **Show the best visualizations** - persona_comparison_bars.png is your star
3. **Connect to business value** - Churn prediction + recommendations
4. **Validate your approach** - Silhouette score + autoencoder agreement
5. **Be honest about limitations** - "Age as primary driver didn't work as expected"

## 🆘 Troubleshooting

**TensorFlow installation issues:**
```bash
# If pip install tensorflow fails, try:
conda install tensorflow
# or
pip install tensorflow-macos  # on Mac M1/M2
```

**Visualization doesn't show:**
```bash
# Install missing packages
pip install matplotlib seaborn pandas
```

**Out of memory:**
```bash
# In autoencoder_clustering.py, reduce batch size:
# Change line: batch_size=256 → batch_size=128
```

---

## 🎉 You're All Set!

Your clustering work is already solid (0.267 silhouette!). Now just:

1. Run `create_visualizations.py` (5 min)
2. Run `autoencoder_clustering.py` (10 min)  
3. Update `writeup.md` (30 min)
4. Submit to GitHub!

Good luck! 🚀