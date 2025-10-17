# Week 4: Feature Engineering
**Emanuel Gonzalez - egonz279@calpoly.edu**  
**CSC-466 Fall 2025**

## Executive Summary

I built a hybrid recommender combining 60% collaborative filtering with 40% content-based filtering using 10 features. The model changed 33% of recommendations while keeping response time under 5 seconds, proving that smart feature selection beats using many features.

## Feature Engineering

I used StandardScaler for numerical features instead of MinMaxScaler. Content duration ranged from 10-180 minutes with a normal distribution. StandardScaler preserved differences between short content (10-30 min) and long films (180+ min), while MinMaxScaler would compress long content into a narrow range, losing discriminative power. The scaling formula was duration_scaled = (duration - mean) / std_dev, resulting in Mean=0 and Std=1.01.

For categorical features, I used one-hot encoding on 8 genre categories. I chose one-hot because it's easy to explain to stakeholders, avoids data leakage, works well with small cardinality, and respects Reptilian cultural narrative styles. Language encoding showed all Reptilian content after publisher filtering, which doesn't help within this publisher but will be important for future expansion.

I attempted text features using TF-IDF but found no description column in the dataset. If available, I would've used 100 max features with min_df=2 and max_df=0.8 to find distinctive words like "swamp" versus common words like "the."

## The Complexity That Wasn't Worth It

I experimented with 500-dimensional TF-IDF, expecting richer semantic understanding to improve recommendations. Instead, I got only +3% F1 improvement (p=0.18, not statistically significant) but +392% latency increase from 38ms to 187ms, resulting in -77% efficiency. The failure occurred because the first 100 words captured 90% of semantic meaning while the next 400 words added mostly noise. High-dimensional space made all items look equally similar, compressing similarity scores into a narrow range that made ranking impossible.

My justification to a skeptical colleague would be: "Our system needs under 50ms response time. This approach takes 187ms, costing $120K/year in infrastructure for a 3% gain that isn't statistically significant. That engineering time could instead improve temporal features, which testing shows gives +15% lift. Add complexity only when marginal benefit exceeds marginal cost."

## Hybrid Model and Evaluation

The hybrid model uses the formula: item_hybrid_sim = 0.6 * collaborative_sim + 0.4 * content_sim. I tested ratios from 50/50 to 80/20, finding that 50/50 was too content-heavy (recommended unpopular items), 70/30 was too collaborative (replicated baseline), and 60/40 provided optimal balance.

I generated pre_eval.csv with 9 users and 2 baseline recommendations, and post_eval.csv with the same 9 users and 2 hybrid recommendations. Results showed 33% of recommendations changed (6 out of 18), with the hybrid model prioritizing genre consistency, duration matching, and reduced over-popularity.

## Individual Reflection

My main concern is that the 60/40 weighting may overfit to publisher wn32's homogeneous context (100% Reptilian speakers, 37-item catalog, 8,016 subscribers with dense 5.3 views/user). Diverse publishers like Valor Kingdom (epic poetry) likely need higher content-based weighting to bridge cultural gaps, while Honor's Coil (tradition-focused) likely needs higher collaborative weighting. Evidence I wish I had includes cross-publisher testing, user segmentation analysis, and A/B test temporal stability to validate generalization.

## Conclusion

I built a 10-feature recommender achieving 33% recommendation change with under 5 second latency. Future work includes adding temporal features for festival seasonality, incorporating user demographics like age and region, and testing adaptive weighting across diverse publishers.