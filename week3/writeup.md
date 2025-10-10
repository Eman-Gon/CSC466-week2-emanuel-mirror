# Week 3: Exploratory Data Analysis Report
**Emanuel Gon**  
**CSC-466 Fall 2025**

---

## Executive Summary

This report documents my data quality audit, exploratory data analysis, and recommender improvements for Week 3. Key findings include:

- **Bimodal engagement pattern**: 64.5% of views have <30% watch time (noise)
- **Severe language bias**: 100% of recommendations are Reptilian (RP) language
- **Data quality issues**: 9,399 duplicate views, 239 impossible watch percentages, 93.2% missing ratings
- **Model improvements**: Duplicate removal and low-engagement filtering to improve signal quality

---

## 1. Data Quality Audit

### 1.1 Missing Values

**Critical Finding: Sparse Ratings**
- **93.2% of views lack ratings** (221,568 out of 237,667 views)
- Only 16,099 views (6.8%) have rating data
- **Impact**: Cannot use ratings as a feature due to extreme sparsity

**Other Datasets:**
- No missing values in `content_metadata`, `adventurer_metadata`, `subscriptions`, or `cancellations` ✓

### 1.2 Duplicate Records

**9,399 duplicate (adventurer, content) pairs identified**
- Same user watched same content multiple times
- Kept record with highest `seconds_viewed` (indicates most complete viewing)
- **Impact**: Removed 3.9% of view records to avoid double-counting

### 1.3 Data Range Anomalies

**Age Distribution:**
- Range: 10 to 9,975 years
- Mean: 390 years
- **3,188 adventurers over 1,000 years old**

This extreme range is **valid** and reflects the fantasy world's lore:
- Ancient dragon-born from Honor's Coil kingdom (Valoron continent)
- Top regions: Duelburg (851 ancient ones, avg 2,485 years), Shield Barrow (929), Narathil (926)
- These are legitimate outliers, not data errors

**Watch Percentage:**
- **239 views (0.1%) exceed 100%** - users watched more than content duration
- Likely causes: replays counted cumulatively, seeking/scrubbing, or auto-play loops
- **Decision**: Kept these records (capped at 100%) as they indicate high engagement

### 1.4 Constraint Violations

**Temporal Constraints: ✓ PASSED**
- All views occur after content creation dates
- No time-travel violations (unlike the Uber example from lecture)

**Duration Constraints: ⚠️ VIOLATION**
- 239 views have `seconds_viewed > content_minutes * 60`
- Addressed by clipping `watch_pct` at 1.0

### 1.5 Class Imbalance

**Language Distribution:**
- Reptilian (RP): 132,989 views (56%)
- Fey Folk (FF): 57,902 views (24%)
- Plant (PL): 23,488 views (10%)
- Solar (SL): 23,288 views (10%)

**Severe imbalance** favoring Reptilian language - potential source of bias.

**Gender Distribution:**
- Nearly balanced: Female (47.7%), Male (47.4%), Non-binary (4.8%) ✓

---

## 2. Exploratory Data Analysis

### 2.1 Bimodal Engagement Pattern (KEY FINDING)

Following Professor Pierce's Snapchat flash/no-flash example, I discovered a **severe bimodal distribution** in watch percentages:

**Distribution:**
- **0-10% watch**: 88,902 views (37.4%) 👈 Dominant mode
- **10-30% watch**: 64,367 views (27.1%)
- **30-70% watch**: 35,373 views (14.9%)  
- **70-90% watch**: 21,800 views (9.2%)
- **90-100% watch**: 27,225 views (11.5%) 👈 Secondary mode

**Summary:**
- **Low engagement (<30%)**: 64.5% of all views
- **High engagement (>70%)**: 20.6% of all views
- **Mid engagement**: Only 14.9%

**Interpretation:**
This bimodal pattern suggests two distinct user behaviors:
1. **Quick bounces** - Users click content but immediately abandon (accidental clicks, poor recommendations, wrong playlist position)
2. **Full engagement** - Users watch most/all of content (genuine interest)

The sparse middle range indicates users either love or hate content - few are ambivalent.

**Impact on Modeling:**
Training on unfiltered data would teach the model that low-engagement matches are acceptable. The 64.5% bounce rate represents **noise, not signal**.

![Bimodal Distribution](eda_bimodality.png)
*Figure 1: Watch percentage distribution showing clear bimodality*

### 2.2 User Activity Patterns

**View Distribution:**
- Mean views per user: 9.2
- Highly skewed - most users have few views, small fraction are power users
- Power-law distribution typical of engagement metrics

**Content Diversity:**
- Mean unique content per user: 8.5
- Users sample broadly rather than re-watching same content

![User Activity](eda_users.png)
*Figure 2: User engagement metrics*

### 2.3 Content Analysis

**Genre Popularity (by view count):**
1. KID (Kids): 43,491 views
2. COM (Comedy): 42,916 views
3. ROM (Romance): 42,567 views
4. RLG (Religious): 32,507 views
5. ACT (Action): 29,547 views

Nearly balanced across top genres - no single genre dominates.

**Language Dominance:**
- Reptilian (RP) has **56% of all views** despite being only 22% of content catalog
- Suggests Oozon continent (swamp/wasteland kingdoms) has most active users

**Content Duration:**
- Mean: 6.5 minutes
- Range: 5-8 minutes
- Relatively uniform duration across content

![Content Analysis](eda_content.png)
*Figure 3: Content popularity and genre distribution*

### 2.4 Demographics

**Age Distribution:**
- Right-skewed with long tail toward ancient ages
- Most users young (10-200 years)
- Dragon-born outliers create extended tail

**Regional Distribution:**
Top regions (by user count):
1. Duelburg (Honor's Coil)
2. Narathil (Honor's Coil)  
3. Shield Barrow (Honor's Coil)
4. Slurpington (Wastelands)
5. Helmhold (Valor)

Honor's Coil kingdom dominates user base.

![Demographics](eda_demographics.png)
*Figure 4: Adventurer demographics and regional distribution*

---

## 3. Recommender Improvements

### 3.1 Changes from Week 2

**Change 1: Duplicate Removal**
- Removed **9,399 duplicate (adventurer, content) pairs**
- Kept record with highest `seconds_viewed` 
- Prevents double-counting same viewing session

**Change 2: Low-Engagement Filtering**
- Removed **103,069 views (45% of data)** with:
  - `watch_pct < 5%` AND `seconds_viewed < 30`
- Rationale: Based on bimodal analysis, these represent noise (accidental clicks, auto-play)
- Hypothesis: **Signal quality > data quantity**

**Change 3: Same Core Algorithm**
- Item-item collaborative filtering with cosine similarity
- Allows direct comparison to Week 2 baseline
- Isolates impact of data cleaning improvements

### 3.2 Implementation Details

**Model Architecture:**
- K-Nearest Neighbors (k=20) on item-item similarity matrix
- Cosine distance metric
- Binary implicit feedback (watched=1, unwatched=0)

**Publisher Selection:**
- Selected publisher `wn32` (8,405 subscribers - largest)
- 27,212 views from this publisher's subscribers after cleaning

**Recommendation Generation:**
- For each user: find items similar to what they've watched
- Score candidates by aggregated similarity to watched items
- Return top 10 unseen items

### 3.3 Model Analysis

**Content Coverage:**
- Recommending only **69 out of 982 items (7%)**
- High concentration - focusing on popular/similar items
- **Concern**: Potential filter bubble

**Genre Bias:**
| Genre | Recommended | Overall Catalog | Bias |
|-------|------------|-----------------|------|
| HOR (Horror) | 18.5% | 15.3% | **OVER** (1.2x) |
| ACT (Action) | 18.5% | 7.8% | **OVER** (2.4x) |
| COM (Comedy) | 18.5% | 14.9% | **OVER** (1.2x) |
| ROM (Romance) | 14.8% | 14.8% | **BALANCED** ✓ |
| RLG (Religious) | 11.1% | 9.3% | **BALANCED** ✓ |

Action genre **severely over-represented** (2.4x) - model learning publisher's content preferences.

**Language Bias: 🚨 CRITICAL ISSUE**
| Language | Recommended | Overall Catalog | Bias |
|----------|------------|-----------------|------|
| RP (Reptilian) | **100.0%** | 22.2% | **OVER** (4.5x) |

**All recommendations are Reptilian language content** - extreme bias.

**Root Cause:**
- All 10 selected users have `primary_language: RP`
- All from Oozon continent regions (Slurpington, Dripwater Delta, Soggy Hollow)
- Publisher `wn32` serves primarily Oozon continent audience

This reveals a fundamental limitation: model learns publisher-specific patterns, not universal preferences.

---

## 4. Recommendation Strategy

### 4.1 User Selection

I selected the **10 most active users from publisher wn32** (power users with highest view counts).

**Rationale:**
- Based on bimodal analysis, testing on engaged users isolates cleaning effects
- Power users have genuine preferences vs casual browsers
- Can validate signal quality improvements before broader rollout

**User Characteristics:**
- Age range: 10-103 years (diverse)
- All speak Reptilian (RP) language
- All from Wastelands/Slimoria kingdoms (Oozon continent)

### 4.2 What I Hope to Learn

**1. Does noise filtering improve precision?**
- Removed 45% of data (103,069 low-engagement views)
- Will recommendations be more accurate without this noise?
- Testing hypothesis: **signal quality > data quantity**

**2. Is language bias a problem?**
- 100% of recommendations are RP language
- Is this appropriate targeting (users are RP speakers) or over-fitting?
- Will users accept content in their native language regardless of other factors?

**3. Genre concentration effects:**
- Model over-represents Action (2.4x) and Horror
- Do power users actually prefer these genres?
- Or is model learning publisher's catalog bias?

**4. Content diversity:**
- Only recommending 7% of available content
- Is this concentration beneficial (popular items) or harmful (filter bubble)?
- Should I increase diversity in future iterations?

### 4.3 Expected Outcomes

**If hypothesis is correct:**
- Engaged users accept recommendations at higher rates than Week 2
- Validates that removing noise improves precision
- Confirms bimodal pattern represents true signal vs noise

**If hypothesis is wrong:**
- Similar/worse acceptance rates suggest:
  - Language bias overwhelms cleaning improvements
  - Feature engineering needed (language matching, genre diversity)
  - Power user behavior differs from general population

**Key Learning:**
Regardless of outcome, this experiment reveals whether **data cleaning alone** can improve recommendations, or if **feature engineering** (language, demographics, content metadata) is essential.

---

## 5. Individual Reflection

### Alternative Approach: Language Filtering

During analysis, I discovered **severe language bias** - 100% of recommendations are Reptilian (RP) language content, despite RP representing only 22.2% of the catalog.

**My Interpretation:**
This occurred because:
1. I selected the publisher with most subscribers (`wn32`)
2. This publisher's audience is primarily RP speakers from Oozon continent
3. Item-item CF learns patterns within this specific audience
4. Result: language-specific recommendations

**Alternative Approach I Considered:**
- Filter recommendations to match user's `primary_language`
- Ensure language-appropriate content
- Would guarantee basic user satisfaction

**Why I Chose NOT To Implement Language Filtering:**

I wanted to test whether **data cleaning improvements alone** (duplicate removal + low-engagement filtering) would surface better recommendations, even with language mismatch.

By keeping language unrestricted, I can:
- Measure the "pure" effect of cleaning changes
- Separate signal quality improvements from feature engineering
- Establish baseline before adding complexity

**Expected Learning:**
- If rejected due to language → Week 4 must add language filtering (critical feature)
- If accepted despite mismatch → Genre/content quality matter more than language

**Trade-off:**
- **Risk**: Higher chance of poor recommendations due to language mismatch
- **Benefit**: Clearer signal about what features matter most for future iterations
- **Decision**: Prioritize learning over short-term performance

### Alternative Interpretation: The >100% Watch Percentage

**The Issue:**
239 views (0.1%) have `watch_percentage > 100%` - users supposedly watched MORE than content duration.

**My Interpretation:**
These represent highly engaged users who **replayed content multiple times**, but the data pipeline summed total watch time instead of capping at 100%.

Evidence supporting this interpretation:
- Small percentage (0.1%) - not systematic corruption
- High engagement signal (watching beyond duration requires interest)
- Consistent with replay behavior

**Alternative Interpretation I Considered:**
These could be **data corruption artifacts** that should be removed entirely as invalid records.

Evidence supporting this interpretation:
- Impossible values violate physical constraints
- Could indicate timestamp errors or measurement bugs
- Might be random noise rather than meaningful signal

**My Decision:**
I kept these records (capped `watch_pct` at 1.0) because:
1. Only 0.1% of data (minimal impact)
2. Underlying signal (high engagement) is valuable
3. Removing engaged users would bias model against super-fans
4. Clipping handles the constraint violation without discarding data

**Trade-off:**
- **Risk**: Keeping corrupted data if these are truly random errors
- **Benefit**: Preserving engagement signals from replay behavior
- **Decision**: When in doubt, preserve signal rather than remove data

**Broader Lesson:**
This reflects a fundamental tension in data science: **when to trust unusual data vs when to treat it as noise**. I erred on the side of trust, but future validation (e.g., checking if these users replay other content) could confirm or refute this decision.

---

## 6. Conclusion

### Key Takeaways

1. **Bimodal engagement** (64.5% bounce rate) is the defining characteristic of this dataset
2. **Language bias** (100% RP recommendations) reveals model learns publisher-specific patterns
3. **Data cleaning** alone (duplicates + low-engagement) may be insufficient without feature engineering
4. **Signal quality vs quantity trade-off** will be validated by user feedback

### Next Steps

**For Week 4:**
- Add language filtering if recommendations are rejected
- Incorporate content metadata (genre, release date) as features
- Consider user demographics (age, region, favorite_genre)
- Explore diversity metrics to combat filter bubble
- Test whether power user patterns generalize to broader population

### Limitations

- Only tested on single publisher's audience (Oozon continent bias)
- Item-item CF ignores content/user features
- Binary implicit feedback loses engagement strength signal
- No temporal dynamics (trending content, seasonality)

---

## Appendix: Technical Details

**Code Structure:**
- `data_audit.py`: Systematic data quality checks
- `eda_viz.py`: Generates 4 visualization figures
- `recommender.py`: Item-item CF with cleaning pipeline
- `deep_investigation.py`: Exploratory analysis (bimodality discovery)

**Reproducibility:**
All code and data available in `week3/` folder of GitHub repository.

**Computational Resources:**
- Dataset: 237,667 views, 25,770 users, 982 content items
- Runtime: ~3 seconds for model training + recommendations
- Environment: Python 3.12, pandas, scikit-learn, matplotlib