# Week 6: Churn Prediction for Adventurers

**Emanuel Gonzalez**

---

## 1. Labeling Logic and Avoiding Leakage

### How I Defined Churn

I defined churn as an adventurer who **cancels their subscription within the next 24 days** (one game-month) from the current date.

**Current date:** Verdantia 1, 10235  
**Prediction window:** Next 24 days (Verdantia 10235)

### Avoiding Temporal Leakage

**Critical rule:** Only use data available BEFORE the prediction point.

**My approach:**
- **Training cutoff:** Used all data up to 2 months before current date (ordinal: 2456424)
- **Labels:** For training, I checked if subscriptions active at the training cutoff churned in the FOLLOWING month
- **Features:** Only computed features using data up to the respective cutoff dates

**Why this matters:**  
If I used current data to predict past churns, I'd get artificially high accuracy but fail on real predictions. The model would be "cheating" by seeing the future.

**Temporal split strategy:**
```
[---- Historical Data ----][-- Train Cutoff --][-- Validation --][-- Current --][-- Predict -->]
     (Learn patterns)       (2 months ago)      (1 month ago)      (Now)        (Next 24 days)
```

**Key insight:** I initially tried using only 1 month of historical data, but discovered the data only goes to Lunaris 10235 (not 10236 as the assignment suggested). I adjusted the current date to Verdantia 1, 10235, which is right after the last available data.

---

## 2. Features and Models

### Features Used

Based on lecture findings that `days_subbed` and engagement metrics were most important, I engineered 11 features:

**Subscription Features (most important):**
- `days_subbed` - How long currently subscribed (at cutoff)
- `num_subscriptions` - Total times this user subscribed to this publisher
- `avg_sub_length_user` - User's average subscription length across all publishers

**Engagement Features:**
- `num_content_viewed` - Content watched during this subscription
- `total_seconds_viewed` - Total watch time during subscription
- `median_seconds_viewed` - Typical watch duration per content
- `days_since_last_view` - Days since user last watched content

**Demographics:**
- `age` - User age

**Publisher Features:**
- `pub_churn_rate` - Publisher's historical churn rate (what % of subs cancel)
- `pub_avg_sub_length` - Typical subscription length for this publisher

**Features I deliberately AVOIDED (based on lectures):**
-  Recency alone (proven to be noise in multiple student presentations)
-  Raw watch percentage without context (inconsistent signal)

### Models Evaluated

**Baseline: Predict all churn**
- Accuracy: 0.443
- F1: 0.614
- *Purpose: Establish floor performance and understand class distribution*

**Model 1: Logistic Regression**
- Accuracy: 0.654
- F1: 0.650
- ROC-AUC: 0.721
- *Why: Interpretable baseline, good for linear patterns, fast to train*

**Model 2: Random Forest** ⭐
- Accuracy: 0.741
- F1: 0.739
- ROC-AUC: 0.829
- *Why: Handles non-linear patterns, provides feature importance, best performance*

---

## 3. Validation and Evaluation

### Evaluation Metrics

I used multiple metrics to get a complete picture:
- **Confusion Matrix:** Shows true/false positives/negatives
- **F1 Score:** Balances precision and recall (critical for imbalanced classes)
- **ROC-AUC:** Overall discriminative ability (0.829 = strong model)
- **Accuracy:** Overall correctness (but less meaningful with class imbalance)

### Class Imbalance Handling

**Churn rate in training data:** 44.26%

This is actually well-balanced! The data naturally had enough churners that I didn't need aggressive sampling techniques.

**Strategies used:**
- Class weights in model (`class_weight='balanced'`)
- Stratified sampling (maintained 44% churn rate in train/val splits)
- Did not need SMOTE or undersampling

### Validation Strategy

- **Train/validation split:** 80/20 (25,015 training / 6,254 validation)
- **Stratified sampling:** Yes (maintained churn rate distribution)
- **Random seed:** 42 (for reproducibility)

**Critical fix:** I discovered duplicate subscriptions (users who subscribed multiple times to the same publisher). I kept only the most recent subscription per user-publisher pair to avoid data leakage and duplicate rows in training.

---

## 4. Results

### Feature Importance (Random Forest)

Top 5 most important features:

1. **avg_sub_length_user** - 33.4% importance
   - *Users with historically short subscriptions are more likely to churn again*
   
2. **days_subbed** - 25.0% importance
   - *Current subscription length is highly predictive*
   
3. **num_subscriptions** - 11.3% importance
   - *Users who've subscribed/unsubscribed multiple times have different churn patterns*
   
4. **days_since_last_view** - 7.4% importance
   - *Disengaged users (haven't watched recently) churn more*
   
5. **total_seconds_viewed** - 5.4% importance
   - *Low total engagement correlates with churn*

**Interpretation:**  
The model learned that **user subscription behavior patterns** (avg_sub_length_user, days_subbed) are far more predictive than demographics (age) or publisher characteristics. This aligns with lecture findings where "days subscribed" was consistently the top feature across student presentations.

Interestingly, `num_subscriptions` has moderate importance - users who subscribe/cancel/resubscribe repeatedly have predictable patterns. The engagement features (viewing behavior) are less important than I expected, suggesting churn is more about subscription habits than content consumption.

### Model Comparison

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Baseline (all churn) | 0.443 | 0.614 | - |
| Logistic Regression | 0.654 | 0.650 | 0.721 |
| Random Forest | **0.741** | **0.739** | **0.829** |

**Best model:** Random Forest

**Why:** Random Forest achieved the highest performance across all metrics (74.1% accuracy, 0.829 AUC). The 0.829 ROC-AUC indicates strong discriminative ability - the model correctly ranks churners vs non-churners 82.9% of the time. Random Forest also handles non-linear relationships better than Logistic Regression, which is important since subscription behavior likely has complex patterns (e.g., users who've churned once might be more likely to churn again, but only if they've been subscribed long enough).

### Final Predictions

- **Total active subscriptions:** 21,123
- **Predicted churners:** 6,929
- **Predicted churn rate:** 32.8%

This is lower than the training churn rate (44.3%), which makes sense - the current active users might be more engaged/stable than the historical average. The model is being conservative, which is appropriate for a production system.

---

## 5. What Worked Well

 **Temporal split prevented leakage**  
Using a 2-month lookback for training with a 1-month prediction window ensured I never used future data. This was critical and required careful ordinal date calculation to get right.

 **Handling duplicate subscriptions**  
Discovering that users could subscribe to the same publisher multiple times was crucial. Keeping only the most recent subscription per user-publisher pair prevented duplicate training examples and made predictions unambiguous.

 **Feature engineering based on lecture insights**  
Following the student presentations' findings that `days_subbed` and subscription history were most important led me to focus on subscription behavior features rather than complex engagement metrics. This proved correct - avg_sub_length_user and days_subbed dominated feature importance.

 **Class balancing with weights**  
Using `class_weight='balanced'` in models helped the algorithms properly weight the minority class without needing to undersample or oversample data.

 **Strong model performance**  
Achieving 0.829 ROC-AUC means the model has good discriminative ability and should perform well on unseen data.

---

## 6. What Didn't Work

 **Initial date assumptions were wrong**  
I initially assumed the current date was Starshade 10236 based on the assignment description, but the data only goes to Lunaris 10235. This caused 0% churn in my training set because I was looking for cancellations that didn't exist yet. I had to verify the actual data range to fix this.

 **First temporal split was too narrow**  
Using only 1 month of historical data gave me 0% churn. I needed to go back 2 months to capture actual churn examples. This taught me to always validate label distributions before training.

 **Engagement features less important than expected**  
Based on lectures emphasizing "watch percentage," I expected viewing metrics to be highly predictive. However, they contributed only ~12% combined importance. Subscription behavior dominated. This suggests churn is driven more by subscription fatigue than content quality/engagement.

 **Couldn't use app_opens data**  
The lectures mentioned app_opens as a strong signal, but I didn't have time to integrate it into the feature engineering pipeline. This could have improved performance.

---

## 7. Next Steps and Improvements

If I had more time, I would:

### Feature Engineering:
-  **Add app_opens** - Mentioned in lectures as a strong engagement signal
-  **Subscription gap patterns** - Analyze time between unsubscribe and resubscribe
-  **Content genre preferences** - Whether users watch content in their preferred language/genre
-  **Playlist completion rates** - How much of each playlist they complete
-  **Holiday/seasonal patterns** - Check if churn varies by game calendar season

### Model Improvements:
-  **Try XGBoost** - Often outperforms Random Forest with proper tuning
-  **Hyperparameter tuning** - Grid search on max_depth, n_estimators, min_samples_split
-  **Ensemble methods** - Stack Random Forest + XGBoost + Logistic Regression
-  **Probability threshold tuning** - Maybe 0.5 isn't optimal; test 0.3-0.7 range

### Analysis:
-  **User segmentation** - Separate models for cold/warm/hot users
-  **Error analysis** - Study false positives/negatives to find patterns
-  **Publisher-specific models** - Some publishers might have unique churn patterns

### Validation:
-  **K-fold cross-validation** - More robust performance estimate
-  **Time-series CV** - Walk-forward validation across multiple time periods
-  **Test on multiple months** - Validate model generalizes across seasons

---

## Reflection

### Key Learnings:

**About churn prediction:**  
Churn is primarily driven by user subscription habits, not content engagement. Users with historically short subscriptions are high-risk regardless of how much they're currently watching. This suggests churn is more about user behavior patterns than publisher content quality.

**What surprised me:**  
The data ended in 10235, not 10236 as I assumed. This taught me to always validate data ranges before building models. Also surprising: engagement features (watching content) were much less predictive than subscription history. I expected "days since last view" to be top 3, but subscription patterns dominated.

**Hardest part:**  
Avoiding temporal leakage while handling duplicate subscriptions. Users can subscribe to the same publisher multiple times, which creates tricky edge cases for labeling and feature engineering. I had to carefully think through which subscription to use for each user-publisher pair.

### Alignment with lectures:

 **Days subscribed was indeed the most important feature** - Confirmed in 3+ student presentations, and my model showed it at 25% importance (2nd place)

 **Class imbalance required special handling** - Using `class_weight='balanced'` was critical

 **Temporal leakage is easy to introduce accidentally** - I had to carefully track cutoff dates throughout the pipeline

 **Simple features often outperform complex ones** - Subscription behavior (simple counts and durations) beat complex engagement metrics

---

## Appendix

### Code Structure

```
churn.py
├── Configuration (dates, paths, calendar)
├── Data Loading (5 parquet files)
├── Preprocessing (ordinal dates, merge cancellations)
├── Label Creation (avoiding leakage, deduplication)
├── Feature Engineering
│   ├── Subscription features (days_subbed, num_subscriptions, avg_sub_length)
│   ├── Engagement features (content viewed, watch time, recency)
│   ├── Demographics (age)
│   └── Publisher features (churn rate, avg sub length)
├── Model Training
│   ├── Baseline (predict all churn)
│   ├── Logistic Regression (scaled features)
│   └── Random Forest (unscaled features)
└── Predictions (churn_pred.csv, detailed probabilities)
```

### Files Submitted

- `churn.py` - Full implementation (preprocessing, training, prediction)
- `churn_pred.csv` - Predicted churners (6,929 adventurer_id, publisher_id pairs)
- `writeup.md` - This report

### Validation Results

All 6,929 predictions are valid:
-  All users exist in subscription data
-  All users are currently subscribed (not already cancelled)
-  No duplicate predictions
-  32.8% predicted churn rate (6,929 / 21,123 active subscriptions)