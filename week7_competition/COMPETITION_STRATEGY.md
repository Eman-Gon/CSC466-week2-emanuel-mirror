# Competition Strategy Guide

## Overview
Based on your Week 5 analysis, here's how to maximize your competition score:

## Key Insights from Your Analysis

1. **Hybrid Model Performance (Best Option)**
   - Your 60/40 (collaborative/content) hybrid achieved best diversity
   - 19 unique items recommended vs 10 for pure collaborative
   - Slightly lower precision but much better discovery

2. **User Segmentation Matters**
   - Hot users (>10 views): Collaborative works best
   - Warm users (4-10 views): Hybrid optimal
   - Cold users (≤3 views): Heuristic/popular items

## Recommended Strategy

### Option 1: Hybrid Approach (Recommended)
Use `generate_final_eval.py` which implements:
- 60% collaborative + 40% content-based
- Selects diverse users (high/medium/low activity)
- Fallback to popular items if needed
- Best for "overall interest score" metric

### Option 2: Simple Collaborative (Backup)
Use `generate_simple_eval.py` if you have issues:
- Pure collaborative filtering
- Simpler, more stable
- Less diverse but reliable

## Competition Scoring Hints

The professor mentioned scoring is based on "overall interest score across all recommendations". This likely means:

1. **Diversity matters** - Recommending the same 2-3 items to everyone will score poorly
2. **Relevance matters** - Items should match user preferences
3. **Coverage matters** - Using more of the catalog shows better understanding

## Optimization Tips

### 1. User Selection Strategy
```python
# Instead of random, select:
- 10 power users (>15 views) - they have rich profiles
- 10 medium users (5-15 views) - good balance
- 10 newer users (2-5 views) - test cold start handling
```

### 2. Recommendation Tuning
- For users with <3 views: Use popular items in their language
- For users with 3-10 views: Use 70/30 hybrid
- For users with >10 views: Use 50/50 or pure collaborative

### 3. Diversity Boosting
```python
# Add small random factor to break ties
scores = scores * (1 + np.random.normal(0, 0.01, len(scores)))
```

### 4. Filter by User Language
```python
# Match content language to user preference
user_lang = df_adventurers[df_adventurers['adventurer_id']==user_id]['primary_language'].iloc[0]
# Filter recommendations to matching language
```

## Quick Validation Checks

Before submitting, verify:

1. **Format**: Exactly 30 rows, 4 columns
2. **No duplicates**: Each user appears once
3. **Valid content**: All recommendations are real content_ids
4. **Diversity**: At least 20+ unique items across all recommendations
5. **No missing values**: All cells filled

## Running the Scripts

```bash
# Generate main submission
python generate_final_eval.py

# If issues, use simpler version
python generate_simple_eval.py

# Check the output
head eval.csv
wc -l eval.csv  # Should be 31 lines (30 + header)
```

## Expected Output Format

```
adventurer_id,rec1,rec2,rec3
4uds,c123,c456,c789
52st,c234,c567,c890
...
```

## Final Tips

1. **Trust your hybrid model** - It showed best results in your analysis
2. **Ensure 30 users** - Competition requires exactly 30
3. **Check for errors** - Some users might not have enough data
4. **Validate content IDs** - Make sure all recommendations exist

## Emergency Fallback

If all else fails, use this simple approach:
1. Get 30 most active users
2. For each user, recommend 3 most popular unwatched items
3. This guarantees valid output even if not optimal

Good luck! Your hybrid approach should perform well given the "overall interest score" likely rewards diversity.