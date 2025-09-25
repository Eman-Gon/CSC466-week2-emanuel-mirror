**What data are you using for your recommendation and why did you choose that data?**

The datasets I decided to use were content_metadata, content_views, adventurers, subscriptions, and cancellations. 

I used subscriptions and cancellations to find three adventurers who were currently subscribed to the same publisher. I converted the subscriptions and cancellation dates to ordinals in order to use a max function. I then checked to see which one was more recent. If a subscription was more recent, they were currently subscribed. If a cancellation was more recent, they weren’t subscribed.

As for the actual recommendation, I decided to keep it simple and use average age, gender ratio, and genre. I initially used studio and language code in my feature space, but found it led to worse recommendations (at least by eyeballing it). The average age and gender ratio was calculated by the adventurers who’ve viewed the content.

**Which adventurers does your recommender serve well? Why?** 

My recommender serves the adventurers who’ve watched a lot of content well. My recommender heavily utilizes the content that the adventurer has watched to recommend new content. My recommender also filters out content that the viewer hasn’t watched the majority of, so the adventurer must have watched a significant chunk of the videos they’ve watched.

**Which adventurers does your recommender not serve well? Why?**

My recommender fails to serve the adventurers who haven’t watched many things. The less they’ve watched, the more inaccurate the recommendation becomes because there is less to work with. The nearest neighbors start to become more of a stretch. Although the content that’s recommended is the nearest neighbor, it doesn’t mean that it’s actually near.

**Why did you pick those three adventurers from the publisher you chose?**

First, I began by choosing a publisher. I chose the publisher with the largest amount of content published. With more content, the recommender has when recommending content. 

Then, once I found the top publisher, I found the three adventurers subscribed to that publisher with the most amount of content watched. As I said earlier, the recommender serves adventurers with more view history. Thus, I decided to choose the adventurers with the most content watched.

**Why do you believe your recommender chose the content it did for those adventurers? Be specific.**

I’d say my recommender was decent at achieving what I intended. In order to eyeball the effectiveness of the recommendation, I compared the suggestions with the content they’ve already watched. I filtered out the content with small watchtime percentages in order to get a more accurate training dataset. 

Adventurer 4jfn is a 12-year-old female whose favorite genre is Horror. However, I viewed the content they’ve actually watched, which was Comedy and Fantasy. The two suggestions that my recommender chose were Documentary and Fantasy, with average ages ~16-17, with a more female-dominant fan base. I believe my recommender chose this content because of the genre they typically watched, as well as their age. 

With Adventurer 3zg2, my recommender didn’t really choose a genre they frequently watched. I think this was because the adventurer had a large variety of genres they watched. (ROM, COM, FNT, DOC, FNT, RLG). Instead, my recommender chose the content based on age. The only thing to add is that the adventurer’s favorite genre is DOC, which my recommender did suggest. However, this is likely coincidental.

My last adventurer, 4886, had recommendations close to their age and gender but further away from genre. 

Overall, if the adventurer watches a bigger variety of genres, the content that is recommended is heavily based on average age and gender ratio of a content viewer’s.
