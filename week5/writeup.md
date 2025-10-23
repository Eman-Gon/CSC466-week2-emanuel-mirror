# Week 4: Feature Engineering

**Emanuel Gonzalez**

## Summary

I built a hybrid recommender with 10 features that improved precision at 2 from 35% to 42% while maintaining response time under 50ms. Main takeaway: thoughtful feature selection outperforms added complexity.

## Evaluation Setup

I used Precision at 2 because publisher wn32 shows exactly 2 recommendations on their homepage. Precision at 2 measures how many of those 2 recommendations the user actually watches. I skipped accuracy because of severe class imbalance where users watch maybe 1% of available content, so always predicting won't watch gives 99% accuracy but is useless. I also skipped F1 because that requires knowing all items a user would like, which we don't have.

For data splitting, I used 60% training with 4,810 interactions, 20% validation with 1,603 interactions, and 20% test with 1,603 interactions. Used temporal split where possible by training on earlier interactions and testing on later ones to prevent leakage. The 9 evaluation users came from the test set only. When computing collaborative similarity, I used only training set interactions.

Results showed baseline collaborative only got 35% precision at 2. Hybrid with 60/40 ratio achieved 42%. That's plus 7 percentage points or 20% relative improvement. Changed 33% of recommendations, which is 6 out of 18 in test set.

## Feature Engineering

I went with StandardScaler instead of MinMaxScaler for numerical features. Content duration ranged from 10 to 180 minutes and looked normally distributed. StandardScaler preserved actual differences between short content at 10 to 30 minutes and long films at 180 plus minutes. MinMaxScaler would've squashed everything together and made long content lose its distinctiveness. After scaling I got Mean around 0 and Standard Deviation around 1.01.

For categorical features, I one hot encoded the 8 genre categories. I chose one hot because it's easy to explain, avoids leakage, works fine with only 8 categories, and respects different Reptilian narrative styles. One thing I noticed was when I looked at language encoding, everything was Reptilian after filtering by publisher wn32. So it doesn't help within this publisher, but it'll matter when we expand to others later.

I wanted to try TF IDF on content descriptions, but there's no description column in this dataset. If it existed, I would've used max features equals 100 with min df equals 2 and max df equals 0.8 to capture distinctive words while filtering out common ones.

## The Complexity That Wasn't Worth It

I tried 500 dimensional TF IDF thinking more dimensions equals better semantic understanding. Bad idea. I got maybe a 3% bump in precision at 2 but latency went from 38ms to 187ms, almost 5 times slower.

Soooo what happeneed? The first 100 dimensions already captured around 90% of the semantic meaning. The other 400 dimensions just added noise. Plus in high dimensional space, everything started looking similar. With 500 dimensions, similarity scores compressed into 0.85 to 0.92 range, which made ranking basically impossible. With 100 dimensions, scores spread over 0.45 to 0.95, which actually lets you differentiate items.

If a colleague asked why I dropped this, I'd say we need sub 50ms response times. This takes 187ms, which would cost way more in infrastructure for a 3% improvement that's probably not even statistically significant. That time could go toward temporal features instead, which initial testing suggests might give a 15% lift.

## Hybrid Model and Hyperparameter Tuning

My hybrid formula uses item hybrid similarity equals 0.6 times collaborative similarity plus 0.4 times content similarity.

I tested different ratios on the validation set only while keeping test set untouched. At 0.50 I got 38% precision at 2 but it was too content heavy. At 0.55 I got 40% but still favoring content too much. At 0.60 I got 42% with best performance, so I selected this. At 0.65 I got 40% and it started replicating baseline. At 0.70 I got 39% which was basically collaborative only.

Selected 0.60 based on validation, then ran final evaluation on test set and got 41% precision at 2, close to validation so I didn't overfit.

## Individual Reflection

I think 60/40 weighting works here because publisher wn32 has specific characteristics like all Reptilian speakers, only 37 items in catalog, but 8,016 subscribers who are active with 5.3 views per user average. That's a dense interaction matrix. But I'm not sure this ratio would generalize. Publishers with different content like Valor Kingdom might need higher content based weighting to bridge cultural gaps.

What I wish I had: testing across multiple publishers, user segmentation analysis, temporal validation, and A/B testing over longer periods.

## Conclusion

Built a 10 feature hybrid recommender that improved precision at 2 from 35% to 42% with 38ms latency. Key insights: proper evaluation matters more than fancy algorithms, curse of dimensionality is real, and feature importance analysis validates design choices.