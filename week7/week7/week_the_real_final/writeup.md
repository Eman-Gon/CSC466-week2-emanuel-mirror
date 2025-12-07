# Week 7: User Personas via SVD-Based Clustering

**Emanuel Gonzalez**

## Summary

I discovered 7 distinct adventurer personas using K-Means clustering on latent features extracted via Singular Value Decomposition, achieving a silhouette score of 0.075 with 3,083 active users. Main takeaway: clustering on SVD embeddings avoids self-referential loops while capturing natural taste-based segmentation.

## The Idea

Starting with 237,667 content views from 25,770 adventurers, I cleaned the data by removing duplicates and filtering low-engagement views (watch percentage < 5% or seconds viewed < 30), resulting in 125,199 meaningful views. I filtered to active users with at least 10 views, leaving 3,083 users with 39,662 views across 314 unique content pieces.

Rather than clustering on hand-crafted behavioral features like completion rate or churn (which creates circular reasoning when both clustering and describing with the same metrics), I applied Truncated SVD to the user-content interaction matrix. This extracted 30 latent dimensions capturing 64.1% of viewing variance. I then standardized these features and ran K-Means for k=2 through k=7, finding k=7 optimal with silhouette score 0.075.

The lower silhouette compared to behavioral clustering (0.267 in typical approaches) is expected when clustering in high-dimensional latent space on complex taste patterns rather than simple engagement metrics. This approach separates clustering (based on what users watch) from interpretation (described by genre preferences not used in clustering).

![User Personas UMAP](./Screenshot%202025-11-28%20at%2012.13.39%20PM.png)

The UMAP visualization shows clear spatial separation between clusters, with Cluster 2 (Horror & Fantasy Niche) forming a distinct isolated group in the middle, while Clusters 0 and 5 group together in the lower left, and Clusters 1, 3, 4, and 6 overlap in the right region, reflecting their shared preferences for Kids and Romance content.

## Results

The seven personas showed distinct content preferences:

**Cluster 4 (Mainstream Majority, 30.2%)**: Largest segment consuming Kids (3.85 views/user) and Romance (2.50) content with the highest churn rate at 84.9%. This represents the biggest retention opportunity.

**Cluster 1 (Romance Enthusiasts, 19.1%)**: Heavy Romance consumers (3.68 views/user) with 74.4% churn and moderate engagement (11.7 views).

**Cluster 5 (Action & Kids Balancers, 21.9%)**: Second-largest group with balanced preferences across Kids (2.61), Comedy (2.58), and Action (2.40), showing the best completion rate at 59.4%.

**Cluster 3 (Kids Content Specialists, 4.2%)**: Strongest Kids preference (5.18 views/user) with highest engagement (15.0 views) but concerning 80% churn, likely representing family accounts.

**Cluster 6 (Romance Devotees, 8.4%)**: Strongest Romance concentration (5.02 views/user) with moderate engagement and 69.0% churn.

**Cluster 2 (Horror & Fantasy Niche, 8.7%)**: Specialized viewers favoring Comedy (3.15), Fantasy (2.18), and Horror (1.89) with high churn (74.3%). The UMAP shows this cluster's isolation reflects unique genre preferences.

**Cluster 0 (Comedy & Kids Enjoyers, 7.6%)**: Balanced viewers with youngest demographic (age 15) and moderate churn (63.9%).

## Reflection

The critical business insight: Cluster 4 represents 30% of active users but suffers 84.9% churn despite solid engagement (13.8 views). This single segment offers the highest-leverage retention opportunity. Future work should investigate content quality gaps in Kids and Romance genres and test persona-specific churn models.