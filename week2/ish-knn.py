import pandas as pd
from sklearn.neighbors import NearestNeighbors

#loading in data
df_content_views = pd.read_parquet("content_views.parquet", engine="pyarrow")
df_subscriptions = pd.read_parquet("subscriptions.parquet", engine="pyarrow")
df_cancellations = pd.read_parquet("cancellations.parquet", engine="pyarrow")


#dropping movies with no ratings
df_content_views = df_content_views.dropna(subset=['rating'])
#content_views_train = df_content_views.iloc[0:int((len(df_content_views)*0.8))]
#content_views_test = df_content_views.drop(content_views_train.index)

#print(len(content_views_train))
#print(len(content_views_test))


MONTH_ORDER = [
    "Frostmere", "Emberfall", "Lunaris", "Verdantia", "Solstice",
    "Duskveil", "Starshade", "Aurorath", "Mysthaven", "Eclipsion"
]

MONTH_TO_INDEX = {m: i for i, m in enumerate(MONTH_ORDER)}


df_content_views = df_content_views.drop_duplicates(['adventurer_id', 'content_id'])
df_content_views = df_content_views.dropna()

df_matrix = df_content_views.pivot(index="content_id", columns='adventurer_id', values="rating").fillna(0)
knn_model = NearestNeighbors(metric='cosine', algorithm = 'brute')
knn_model.fit(df_matrix.values)

user_means = df_matrix.mean(axis=1)
df_matrix = df_matrix.sub(user_means, axis=0)


def recommend(item_id, user_id):   
    item = df_matrix.loc[item_id]
    dists, indices = knn_model.kneighbors([item.values], n_neighbors=10)
    new = list(df_matrix.index[indices[0]])
    rec = []
    watched = list(df_content_views[df_content_views["adventurer_id"] == user_id]["content_id"])

    for i in new:
        if i not in watched:
            rec.append(i)
    
    return rec
    

def find_content(user_id):
    c = list(df_content_views.loc[
            (df_content_views["adventurer_id"] == user_id) & (df_content_views["rating"] == 5), "content_id"])
    
    return c[0]



def mystical_to_ordinal(year: int, month: str, day: int) -> int:
    """Convert the mystical calendar date to an absolute ordinal."""
    month_index = MONTH_TO_INDEX[month]   
    return year * (10 * 24) + month_index * 24 + (day - 1)


def choose_three_adventurers():
    """Returns a list of three adventurers. This function picks adventurers subscribed to the publisher with the most content."""

    # Read parquet files
    subscriptions = df_subscriptions
    cancellations = df_cancellations
    content_views = df_content_views

    subscriptions["ordinal"] = subscriptions.apply(
        lambda r: mystical_to_ordinal(r["year"], r["month"], r["day_of_month"]),
        axis=1
    )
    cancellations["ordinal"] = subscriptions.apply(
        lambda r: mystical_to_ordinal(r["year"], r["month"], r["day_of_month"]),
        axis=1
    )

    # Find active subs by getting the most recent date between cancellations and subscriptions.
    active_subs = pd.merge(subscriptions, cancellations, on=['adventurer_id','publisher_id'],how='outer',suffixes=("_sub", "_cancel"))
    active_subs["last_event_date"] = active_subs[["ordinal_sub", "ordinal_cancel"]].max(axis=1)
    active_subs["is_active"] = ((active_subs["ordinal_sub"] == active_subs["last_event_date"]))
    current_active = active_subs[active_subs["is_active"]]

    # Find the three adventurers with the most amount of views of content from the 'top publisher' :P
    pubs = df_subscriptions["publisher_id"].value_counts().sort_values()
    top_pub_id = pubs.index[-1] #finding the most popular publisher
    active_subs_for_top_pub = current_active[current_active["publisher_id"] == top_pub_id]

    active_views_for_top_pub = content_views[
        (content_views["adventurer_id"].isin(active_subs_for_top_pub["adventurer_id"])) &
        (content_views["publisher_id"] == top_pub_id)
    ]

    '''
    most_viewed_active = (
        active_views_for_top_pub.groupby("adventurer_id")["content_id"]
        .nunique()
        .reset_index(name="viewed_count")
        .nlargest(3, "viewed_count")
    )
    '''
    five_star_active = (
    active_views_for_top_pub[active_views_for_top_pub["rating"] == 5]
    .drop_duplicates("adventurer_id")[["adventurer_id"]]
)

    return five_star_active['adventurer_id']
    
    

    
    



chosen_ones = choose_three_adventurers()[0:10]


#doing 10 because some have the same top rated content, so only taking the unique recomendations to add some variety
for i in chosen_ones:
    content = find_content(i) #find the most top rated content each adventurer rated
    recced = recommend(content, i) #find similar content to what the adventurer liked
    print(f"recced to {i}: {recced}")
