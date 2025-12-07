# Week 7: User Personas Project

## Quick Start

### Running the Analysis

1. **Run K-Means clustering locally**:
```bash
python3 personas.py
```

2. **Run autoencoder in Kaggle** (due to TensorFlow requirements):
   - Upload `autoencoder.py` to Kaggle notebook
   - Run the notebook
   - Download results (`user_profiles_with_ae_clusters.csv`)
   - Copy back to local project

3. **Generate visualizations**:
```bash
python3 create_visuals.py
```

## Project Structure
```
week7/
├── personas.py                # Main clustering (run locally)
├── autoencoder.py            # Validation (run in Kaggle)
├── create_visuals.py         # Generate charts (run locally)
├── writeup.md                # Final report
└── *.parquet                 # Data files
```

## Key Results

- **7 user personas** discovered
- **Silhouette score: 0.267** (K-Means)
- **Validation: 0.365** (Autoencoder)
- **25,770 users** clustered

## Why Kaggle?

Used Kaggle for `autoencoder.py` because:
- Free GPU/TPU access
- Pre-installed TensorFlow
- No local setup required
- Easy data upload/download

## Files Generated

After running all scripts:
- `user_profiles_with_clusters.csv` - User cluster assignments
- `persona_insights.csv` - Cluster statistics
- `cluster_evaluation.png` - Silhouette scores
- `persona_comparison_bars.png` - Main visualization
- `persona_distribution.png` - Cluster sizes
- `persona_heatmap.png` - Feature comparison

## Submission
```bash
git add .
git commit -m "Week 7: User Personas"
git push origin main
```
