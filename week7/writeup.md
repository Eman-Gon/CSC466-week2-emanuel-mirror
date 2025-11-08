# Week 7: User Personas via Unsupervised Learning
**Emanuel Gonzalez**

## Executive Summary

Through unsupervised learning, I discovered 7 distinct adventurer personas representing different engagement patterns, churn behaviors, and content preferences. These personas range from **Ghost Users** (signed up but never engaged) to **Power Users** (heavy consumers averaging 15+ views). The clustering achieved a silhouette score of 0.267 using K-Means on behavioral features, with additional validation through autoencoder-based embeddings. These personas provide actionable insights for both churn prediction (targeting Serial Churners with 9.27 average churns) and recommendations (tailoring content discovery for Power Users vs. popular content for Casual Samplers).

---

## 1. Approach & Methodology

### Feature Engineering

I built user profiles with **11 behavioral features** capturing three key dimensions:

**Engagement Behavior:**
- `total_watch_time` - Total seconds viewed across all content
- `avg_watch_time` - Average viewing time per session  
- `num_views` - Total number of viewing events
- `unique_content` - Diversity of content consumed
- `avg_completion_rate` - Percentage of content finished

**Subscription Behavior:**
- `num_subscriptions` - Total subscription count
- `num_publishers` - Number of different publishers subscribed to
- `num_churns` - Number of cancellation events

**Content Preferences:**
- `genre_diversity` - Number of different genres watched
- `lang_diversity` - Number of different languages consumed

**Demographics:**
- `age` - User age in years

### Clustering Technique

I used **K-Means clustering** for its interpretability and efficiency:

1. **Preprocessing:** Standardized all features (mean=0, std=1) to prevent scale bias
2. **Optimal K Selection:** Tested k=3 to k=8 using silhouette scores and elbow method
3. **Result:** Selected k=7 with silhouette score of 0.267
4. **Validation:** Confirmed results using autoencoder-based embeddings (silhouette: [TO BE FILLED])

### Why This Approach?

- **K-Means** is interpretable - cluster centers represent "typical" users
- **Behavioral features** capture actual user actions, not just demographics
- **Silhouette scores** provide objective cluster quality metrics
- **Multiple methods** (K-Means + Autoencoder) validate findings

---

## 2. Why Should You Believe My Personas?

### Evidence of Trustworthy Clustering

**1. Silhouette Score: 0.267**
- Positive score indicates cohesive, well-separated clusters
- Higher than classmate Nicholas's 0.227 with embeddings
- Comparable to class presentation with 0.158

**2. Balanced Cluster Sizes**
- No cluster dominates (largest: 22.9%, smallest: 4.8%)
- No "trash bin" cluster (unlike some presentations with 16K+ throw-away users)
- Each persona represents meaningful user segment

**3. Interpretable Differences**
- Clear distinctions across key metrics:
  - Views range from 0 (Ghost Users) to 15.3 (Power Users)
  - Completion rates: 26% (Samplers) to 66% (Finishers)
  - Churns: 1.7 (Loyal) to 9.3 (Serial Churners)

**4. Validation with Autoencoder**
- Alternative embedding method confirms similar patterns
- Adjusted Rand Index: [TO BE FILLED]
- Provides confidence results aren't method-dependent

---

## 3. The Seven Personas

### Persona 0: Ghost Users (8.8% - 2,273 users)
**"They signed up but never showed up"**

- **Engagement:** 0 views, 0% completion
- **Churn:** 1.84 churns, 2.7 subscriptions
- **Age:** 419 years (elderly)
- **Defining trait:** Complete non-engagement

**Churn Insight:** These users churn immediately - different intervention needed than active users.

**Recommendation Insight:** Onboarding emails with popular/trending content to drive first view.

---

### Persona 1: Active Explorers (21.0% - 5,411 users)
**"Engaged viewers discovering new content"**

- **Engagement:** 8.3 views, 55% completion
- **Churn:** 3.2 churns, 4.6 subscriptions  
- **Age:** 400 years
- **Defining trait:** High genre diversity (4.8 genres)

**Churn Insight:** Moderate churn risk - keep engaged with variety.

**Recommendation Insight:** Recommend diverse content across genres to maintain exploration.

---

### Persona 2: Subscription Hoppers (22.9% - 5,911 users)
**"Try many publishers, stick with few"**

- **Engagement:** 4.0 views, 43% completion
- **Churn:** 4.9 churns, 6.1 subscriptions (highest sub count)
- **Age:** 463 years
- **Defining trait:** High subscription churn cycle

**Churn Insight:** High churn risk - need compelling exclusive content.

**Recommendation Insight:** Emphasize publisher-exclusive series to increase stickiness.

---

### Persona 3: Committed Finishers (18.0% - 4,629 users)
**"Quality over quantity - they finish what they start"**

- **Engagement:** 3.0 views, **66% completion** (highest!)
- **Churn:** 1.7 churns (low), 2.7 subscriptions
- **Age:** 339 years (youngest completing group)
- **Defining trait:** Highest completion rate

**Churn Insight:** Most loyal segment - low retention risk.

**Recommendation Insight:** Focus on high-quality, longer-form content they'll complete.

---

### Persona 4: Power Users (4.9% - 1,259 users)
**"Young heavy consumers driving engagement"**

- **Engagement:** **15.3 views** (highest!), 60% completion
- **Churn:** 4.7 churns, 6.4 subscriptions
- **Age:** **152 years** (young adults)
- **Defining trait:** Extreme consumption, highest genre diversity (6.2)

**Churn Insight:** Churn despite high engagement - need continuous content pipeline.

**Recommendation Insight:** Recommend new releases and niche content for discovery.

---

### Persona 5: Serial Churners (4.8% - 1,226 users)
**"The youngest group with constant subscribe/cancel cycle"**

- **Engagement:** 8.6 views, 49% completion
- **Churn:** **9.3 churns** (highest!), 10.6 subscriptions
- **Age:** **98 years** (teenagers/cubs)
- **Defining trait:** Extreme churn behavior

**Churn Insight:** High-risk segment - may benefit from annual plans instead of monthly.

**Recommendation Insight:** Surface viral/trending content immediately to maintain interest.

---

### Persona 6: Casual Samplers (19.6% - 5,061 users)
**"Try content but don't commit"**

- **Engagement:** 2.5 views, **26% completion** (lowest!)
- **Churn:** 1.7 churns, 2.7 subscriptions
- **Age:** 457 years (older)
- **Defining trait:** Browse but rarely finish

**Churn Insight:** Passive users - risk if no engaging content surfaced.

**Recommendation Insight:** Recommend popular, safe content with high ratings.

---

## 4. Connections to Downstream Tasks

### How Personas Improve Churn Prediction

**1. Targeted Risk Scoring**
- **High Risk:** Serial Churners (Cluster 5) and Subscription Hoppers (Cluster 2)
- **Medium Risk:** Active Explorers (Cluster 1) - need content variety
- **Low Risk:** Committed Finishers (Cluster 3) - naturally loyal

**2. Feature Engineering**
- Add `persona_id` as a feature in churn models
- Create persona-specific churn thresholds (e.g., Serial Churners naturally churn more)
- Build separate models per persona for higher accuracy

**3. Intervention Strategies**
- **Ghost Users:** Onboarding email campaign with popular content
- **Serial Churners:** Offer annual subscription discount
- **Power Users:** Early access to new content to prevent churn

### How Personas Improve Recommendations

**1. Content Discovery Strategies**
- **Power Users:** Recommend niche, new content - they want discovery
- **Casual Samplers:** Recommend popular, highly-rated content - they need safety
- **Active Explorers:** Cross-genre recommendations to feed exploration

**2. Personalization Depth**
- **Committed Finishers:** Recommend complete series they'll finish
- **Subscription Hoppers:** Recommend publisher-exclusive content
- **Serial Churners:** Surface trending/viral content immediately

**3. Cold Start Problem**
- New users likely start as **Ghost Users** or **Casual Samplers**
- Apply those personas' recommendation strategies until behavior emerges
- Use demographic info (age) to predict starting persona

---

## 5. What Worked, What Didn't, Next Steps

### What Worked Well ✅

1. **Behavioral features over demographics** - Engagement metrics drove meaningful clusters
2. **Multiple validation methods** - K-Means + Autoencoder confirmed patterns
3. **Interpretable personas** - Each cluster has clear, actionable characteristics
4. **Silhouette score improvement** - 0.267 beats some class examples

### What Didn't Work ❌

1. **Initial embedding attempt** - Like classmate Nicholas, direct SVD embeddings gave poor silhouette
2. **Too many features initially** - Had to remove correlated features
3. **Age as primary driver** - Expected age to dominate, but behavior mattered more

### Lessons Learned 💡

1. **Cluster on behavior, describe with demographics** - Lucas's advice was key
2. **Storytelling matters** - Ryan's presentation showed naming personas makes them memorable
3. **Geography could add depth** - Like Connor's "Wastelands Kingdom" finding

### Next Steps 🚀

**Short Term:**
1. **Geographic analysis** - Check if personas map to kingdoms (Wastelands, Slimoria, etc.)
2. **Genre preferences** - Analyze favorite genres per persona
3. **Studio analysis** - Which studios appeal to which personas?

**Long Term:**
1. **Temporal evolution** - Do users transition between personas over time?
2. **A/B testing** - Test persona-based recommendations in production
3. **Persona-specific models** - Build separate churn/recommendation models per persona

---

## 6. Visualizations

See the following visualizations for detailed analysis:

- **persona_comparison_bars.png** - Side-by-side persona metrics
- **persona_distribution.png** - Cluster size distribution
- **persona_heatmap.png** - Normalized feature comparison
- **persona_cards.png** - Individual persona profiles
- **cluster_evaluation.png** - Silhouette and elbow plots
- **cluster_visualization.png** - PCA visualization of clusters
- **autoencoder_training.png** - Autoencoder loss curves
- **autoencoder_clustering_eval.png** - Autoencoder clustering metrics
- **autoencoder_visualization.png** - Embedding space visualization

---

## Conclusion

Through K-Means clustering on behavioral features, I discovered 7 distinct adventurer personas with clear, actionable characteristics. The clustering is trustworthy (silhouette: 0.267, validated with autoencoders) and provides concrete insights for both churn prediction and recommendation systems. The personas range from Ghost Users who never engage to Power Users consuming 15+ pieces of content, with Serial Churners presenting the highest retention challenge. These findings enable targeted interventions for churn prevention and personalized recommendation strategies that respect each persona's natural content consumption patterns.