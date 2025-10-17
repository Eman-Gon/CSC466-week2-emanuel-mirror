## Week 4: Feature Engineering Results

### Approach
I implemented a hybrid recommender combining:
- **Collaborative filtering** (60% weight): Item-item similarity from user interactions
- **Content-based filtering** (40% weight): Features from content metadata

### Features Engineered
1. **Numerical**: StandardScaler on duration (mean=0, std=1)
2. **Categorical**: One-hot encoding for genre and language
3. **Text**: TF-IDF on title + description (100 features, bigrams)

### Results
Generated two evaluation files:
- `pre_eval.csv`: Baseline collaborative filtering (Week 3)
- `post_eval.csv`: Hybrid with engineered features (Week 4)

**Hypothesis**: The hybrid model will achieve higher acceptance rates because:
1. TF-IDF captures semantic similarity in content descriptions
2. Genre encoding ensures recommendations match user preferences
3. Duration scaling normalizes the feature space

### Next Steps
Awaiting online evaluation results to validate improvements.