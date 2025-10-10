## Data Quality Findings

### 1. Bimodal Watch Percentage Distribution
Following Professor Pierce's Snapchat camera example, I discovered 
a severe bimodal distribution in watch percentages:

- **64.5% of views** have <30% engagement (likely accidental clicks)
- **20.6% of views** have >70% engagement (true interest)
- Only **14.9%** fall in the middle range

This suggests the data contains significant noise from:
- Accidental playlist clicks
- Poor recommendations that users immediately abandon
- Auto-play scenarios where users weren't actively watching

**Impact on modeling:** Training on this noisy signal would teach 
the model that low-quality matches are acceptable.

### 2. Language Class Imbalance
Reptilian (RP) language dominates with 56% of all views, creating 
a severe class imbalance that could bias recommendations toward 
Oozon continent content regardless of user preferences.

### 3. No Temporal Leakage Detected
All view timestamps occur after content creation dates - no 
time-travel issues like the Uber example from lecture.



## Week 3 Improvements

### Change 1: Low-Engagement Filtering
**Rationale:** The bimodal distribution analysis revealed 37.4% of 
views had <10% watch time. These likely represent:
- Accidental clicks in playlists
- Poor recommendations users immediately rejected
- Auto-play noise

**Implementation:** Removed views with <5% watch AND <30 seconds 
viewed (28,784 views filtered out).

**Expected Impact:** Higher precision by training only on engaged 
views that reflect true preferences.

### Change 2: Duplicate Removal
Removed 8,467 duplicate (adventurer, content) pairs, keeping the 
instance with highest watch time.

### Why These Changes?
My goal is to test whether **signal quality** matters more than 
**data quantity**. If the bimodal pattern indicates two distinct 
behaviors (engaged vs noise), filtering noise should improve 
recommendations.


## Recommendation Strategy

I selected the **10 most active users** to test my hypothesis about 
engaged vs noisy views. These power users have:
- High view counts (demonstrating platform engagement)
- Diverse content consumption
- Likely to have genuine preferences vs casual browsers

### What I Hope to Learn:
1. **Does filtering improve precision?** Will removing low-engagement 
   views result in better recommendations for active users?
2. **Language bias:** Will recommendations over-index on Reptilian 
   content due to training data imbalance?
3. **Genre patterns:** Do active users accept diverse genres or 
   stick to their favorites?

By focusing on power users, I can validate whether my cleaning 
improved signal quality before rolling out to casual users.





🚨 MAJOR DISCOVERY: LANGUAGE BIAS!
Language distribution in recommendations:
  RP: 100.0% (overall: 22.2%) [OVER]
THIS IS HUGE! Your model is severely biased toward Reptilian (RP) language!

100% of recommendations are RP content
But RP is only 22.2% of overall content
This is a 4.5x over-representation!

Why? All 10 recommended users have primary_language: RP and are from Oozon continent regions (Slurpington, Dripwater Delta, Soggy Hollow - all Wastelands/Slimoria kingdoms!)




## Individual Reflection

### Alternative Approach Considered: Language Filtering

During my analysis, I discovered my model has severe language bias - 
100% of recommendations are Reptilian (RP) language content, despite 
RP representing only 22.2% of the overall catalog.

**My interpretation:** This occurred because I selected the publisher 
with most subscribers (wn32), and that publisher's audience happens 
to be primarily RP speakers from the Oozon continent.

**Alternative approach I considered:**
- Filter recommendations to match user's `primary_language`
- This would ensure language-appropriate content
- However, I chose NOT to do this for Week 3

**Reasoning:** I wanted to test whether my data cleaning improvements 
(duplicate removal + low-engagement filtering) would naturally surface 
better recommendations, even with language mismatch. By keeping language 
unrestricted, I can measure the "pure" effect of my cleaning changes.

**Expected learning:** If recommendations are rejected due to language 
mismatch, Week 4 improvements should add language filtering. If they're 
accepted despite language mismatch, it suggests other factors (genre, 
content quality) matter more than language.

**Trade-off:** Higher risk of poor recommendations now, but clearer 
signal about what features matter most for future iterations.







## Recommendation Strategy

I selected the **10 most active users from publisher wn32** 
(8,405 subscribers - the largest publisher). These users represent 
**power users** who have demonstrated consistent engagement.

### Why This Strategy?

Based on my bimodality analysis (64.5% of views have <30% engagement), 
I hypothesized that filtering low-engagement views would improve 
recommendations for truly engaged users. By selecting power users, 
I can test this hypothesis on users most likely to have genuine 
preferences rather than random clicking behavior.

### What I Hope to Learn:

1. **Does noise filtering improve precision?**
   - Removed 103,069 low-engagement views (45% of data)
   - Will recommendations be better without this noise?

2. **Is language bias a problem?**
   - 100% of recommendations are RP language
   - All users are RP speakers from Oozon continent
   - Is this appropriate targeting or over-fitting?

3. **Genre concentration effects:**
   - My model over-represents Action (2.4x) and Horror
   - Do power users actually prefer these genres?

4. **Content diversity:**
   - Only recommending 7% of available content
   - Is this concentration good (popular items) or bad (filter bubble)?

### Expected Outcomes:

If my hypothesis is correct, these engaged users should accept 
recommendations at higher rates than Week 2, validating that 
**signal quality > data quantity**. However, the severe language 
bias may limit acceptance - teaching me that demographic features 
(like language) are critical for Week 4 improvements.



## Critical Data Quality Issue: Impossible Watch Percentages

### Problem
239 views (0.1%) have watch_percentage > 100%, meaning users 
supposedly watched MORE than the content's total duration.

### Possible Causes:
1. **Replays counted cumulatively** - User watched multiple times, 
   seconds summed
2. **Seeking/scrubbing** - Fast-forward/rewind inflated the count
3. **Data corruption** - Timestamps or durations incorrectly recorded
4. **Auto-play loops** - Content replayed automatically

### Decision:
I kept these records because:
- Only 0.1% of data (minimal impact)
- Clipping watch_pct at 1.0 handles the issue
- Indicates VERY high engagement (even if measurement is flawed)

Removing them would be too aggressive for such engaged users.




### Alternative Interpretation: The >100% Watch Percentage

During data cleaning, I discovered 239 views with watch_percentage 
> 100% (impossible values). 

**My interpretation:** These represent highly engaged users who 
replayed content multiple times, but the data pipeline summed their 
total watch time instead of capping at 100%.

**Alternative interpretation I considered:** These could be data 
corruption artifacts that should be removed entirely as invalid records.

**My decision:** I kept them (capped at 100%) because:
- They represent only 0.1% of data
- The underlying signal (high engagement) is valuable
- Removing engaged users would bias the model against super-fans

**Trade-off:** Risk keeping corrupted data vs losing signal about 
highly engaged users. I prioritized preserving engagement signals, 
but this could backfire if these are truly random errors rather than 
replay behavior.

This reflects a broader tension in data science: **when to trust 
unusual data vs when to treat it as noise**.