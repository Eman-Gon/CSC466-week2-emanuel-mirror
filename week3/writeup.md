# Week 3: Exploratory Data Analysis
**Emanuel Gonzalez - egonz279@calpoly.edu**  
**CSC-466 Fall 2025**

## Data Quality Audit 
The data quality audit revealed that 93.2% of content views lack ratings (221,568 out of 237,667), making ratings too sparse to serve as a reliable feature, while all other datasets are complete with no missing values. I identified 9,399 duplicate (user, content) pairs where the same user viewed the same content multiple times; to ensure accuracy, I retained only the record with the highest seconds_viewed value. The age data ranges from 10 to 9,975 years with a mean of 390, including 3,188 adventurers over 1,000 years old, valid outliers representing dragon born adventurers from Honors Coil, consistent with the worlds lore. Additionally, 239 views exceed 100% watch percentage, likely due to replays being counted cumulatively; these records were capped at 100% to preserve engagement signals. All views occur after content creation dates, confirming there are no temporal inconsistencies, though the same 239 records technically violate content duration constraints. Finally, the dataset exhibits moderate class imbalance, with Reptilian (RP) as the dominant language (56% of views), while gender representation remains balanced across Female (47.7%), Male (47.4%), and Nonbinary (4.8%) users.

## insaneeeeeee Bimodal Engagement
Following Professor Pierce's Snapchat flash/no-flash example, I discovered a severe bimodal distribution in watch percentages. 37.4% of views have 0-10% watch time (quick bounces), while 11.5% have 90-100% watch time (full engagement). Only 14.9% fall in the middle range of 30-70% watch time.

Overall, 64.5% of views have less than 30% engagement, likely representing noise from accidental clicks and poor recommendations.

![Bimodal Pattern](eda_bimodality.png)

The impact of this pattern is significant. Training on unfiltered data would teach the model that low-quality matches are acceptable.

## Recommender Improvements

### Changes Made

I made three changes to my recommender. First, I removed 9,399 duplicates and kept the record with the highest watch time. Second, I filtered out 103,069 low-engagement views (45% of the data) by removing views with less than 5% watch AND less than 30 seconds viewed. Third, I kept the same core algorithm, item-item collaborative filtering with cosine similarity, to allow direct comparison to Week 2.

My hypothesis is that signal quality matters more than data quantity.

### Results Analysis

I selected publisher wn32, which has 8,405 subscribers and is the largest publisher. I focused on the 10 most active power users. The model only recommends 69 out of 982 items, which is 7% content coverage.
The genre distribution shows bias. Action is recommended at 18.5% but only represents 7.8% of the catalog, making it over-represented by 2.4x. Horror is recommended at 18.5% versus 15.3% in the catalog (1.2x over). Romance is balanced at 14.8% in both recommendations and catalog.
The language bias is a critical issue. 100% of recommendations are Reptilian (RP) language, but RP only represents 22.2% of the catalog. All 10 selected users speak RP and live in Oozon continent regions like Slurpington, Dripwater Delta, and Soggy Hollow.
The root cause is that my model learns publisher-specific patterns. Publisher wn32 serves an Oozon continent audience exclusively.

## Recommendation Strategy

I selected the 10 most active users to test whether filtering low-engagement views improves precision.

### What I Hope to Learn

I want to answer four questions. First, does removing noise (45% of data) improve recommendations? Second, is 100% RP language targeting appropriate or is it over-fitting? Third, do power users actually prefer Action and Horror genres as the model suggests? Fourth, is 7% content coverage good because it focuses on popular items, or bad because it creates a filter bubble? My expected outcome is that if cleaning helps, users will accept recommendations at higher rates than Week 2. If not, language bias overwhelms the cleaning improvements and feature engineering will be needed.

## Individual Reflection


My interpretation is that 100% RP recommendations occurred because I selected the largest publisher, whose audience happens to be RP speakers. Item-item collaborative filtering learned this specific audience's patterns.
An alternative approach I considered was to filter recommendations to match each user's primary_language field, which would guarantee language-appropriate content.I chose not to implement language filtering because I wanted to test whether data cleaning alone improves recommendations without feature engineering. By keeping language unrestricted, I can measure the pure effect of my cleaning changes.The trade-off is that I risk poor recommendations due to language mismatch, but I gain a clear signal about what features matter most for Week 4 improvements. I decided to prioritize learning over short-term performance.
The issue is that 239 views exceed 100% watch time. My interpretation is that these represent replays counted cumulatively and serve as a high engagement signal.
An alternative interpretation I considered is that these could be data corruption artifacts that should be removed entirely as invalid records. I decided to keep them (capped at 100%) because they represent only 0.1% of data, they signal super-fans who replay content, and clipping handles the constraint violation without discarding potentially valuable data.The broader lesson is that when uncertain about unusual data, I should preserve signal rather than discard data.

## Visualizations

![User Activity](eda_users.png)

User engagement follows a power-law distribution with a mean of 9.2 views per user.

![Content Analysis](eda_content.png)

Genre distribution is balanced across content, but RP language dominates views at 56%.

![Demographics](eda_demographics.png)

Age distribution is right-skewed due to dragon-born outliers. Honor's Coil kingdom dominates the regional distribution.

## Conclusion
Three key findings emerged from this analysis. First, bimodal engagement with a 64.5% bounce rate defines the dataset. Second, language bias with 100% RP recommendations reveals that the model learns publisher-specific patterns rather than universal preferences. Third, I removed 45% of the data as noise and will validate whether this improves recommendations.
For Week 4, I will add language filtering if recommendations are rejected due to language mismatch. I will incorporate content metadata like genre and release date as features. I will consider user demographics including age, region, and favorite_genre. I will test whether power user patterns generalize to the broader population.
This analysis has several limitations. I only tested on a single publisher's audience, creating Oozon continent bias. My item-item collaborative filtering approach ignores content and user features. Binary implicit feedback loses information about engagement strength. The model does not capture temporal dynamics like trending content or seasonality.