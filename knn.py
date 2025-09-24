
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

MONTH_ORDER = [
    "Frostmere", "Emberfall", "Lunaris", "Verdantia", "Solstice",
    "Duskveil", "Starshade", "Aurorath", "Mysthaven", "Eclipsion"
]
# come back to this
MONTH_TO_INDEX = {m: i for i, m in enumerate(MONTH_ORDER)}


def mystical_to_ordinal(year: int, month: str, day: int) -> int:
    """Convert the mystical calendar date to an absolute ordinal (integer)."""
    month_index = MONTH_TO_INDEX[month]   
    return year * (10 * 24) + month_index * 24 + (day - 1)

def find_top_publisher() -> str:
    """Returns the publisher with the most amount of content."""
    content_views = pd.read_parquet('content_views.parquet')
    counts = content_views.groupby("publisher_id")["content_id"].nunique().reset_index(name="content_count")

    # Find the publisher with the maximum content
    top_publisher = counts.loc[counts["content_count"].idxmax()]

    return top_publisher['publisher_id']
    

def choose_three_adventurers():
    """Returns a list of three adventurers"""

    # Read parquet files
    subscriptions = pd.read_parquet('subscriptions.parquet')
    cancellations = pd.read_parquet('cancellations.parquet')
    content_views = pd.read_parquet('content_views.parquet')

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

    top_pub_id = find_top_publisher()
    active_subs_for_top_pub = current_active[current_active["publisher_id"] == top_pub_id]


    active_views_for_top_pub = content_views[
        (content_views["adventurer_id"].isin(active_subs_for_top_pub["adventurer_id"])) &
        (content_views["publisher_id"] == top_pub_id)
    ]

    most_viewed_active = (
        active_views_for_top_pub.groupby("adventurer_id")["content_id"]
        .nunique()
        .reset_index(name="viewed_count")
        .nlargest(3, "viewed_count")
    )
    return most_viewed_active['adventurer_id']

def recommend_content(adventurer_id):
    content_metadata = pd.read_parquet('content_metadata.parquet')
    content_views = pd.read_parquet('content_views.parquet')

    adventurer_views = content_views.loc[content_views['adventurer_id'] == adventurer_id]
    viewed_content = pd.merge(adventurer_views, content_metadata, on=['content_id'],how='left')

    unseen_content = content_metadata.loc[
        (~content_metadata['content_id'].isin(viewed_content))
    ].copy()

    
    numeric_features = ['minutes']  # numeric column(s)
    
    # Convert categorical columns to numeric codes
    categorical_features = ['genre_id', 'language_code']  # example categorical columns
    for col in categorical_features:
        content_metadata[col] = content_metadata[col].astype('category').cat.codes
        viewed_content[col] = viewed_content[col].astype('category').cat.codes
        unseen_content[col] = unseen_content[col].astype('category').cat.codes
        numeric_features.append(col)


    X_train = viewed_content[numeric_features].fillna(0).to_numpy()
    X_test = unseen_content[numeric_features].fillna(0).to_numpy()

    neigh = NearestNeighbors(n_neighbors=2, metric='euclidean')
    neigh.fit(X_train)

    distances, indices = neigh.kneighbors(X_test)

    top_indices = np.argsort(distances.mean(axis=1))[:2]  
    recommended_content = unseen_content.iloc[top_indices]['content_id'].tolist()

    print(f"Recommendations for adventurer {adventurer_id}: {recommended_content}")
    return recommended_content




if __name__ == "__main__":
    three_adventurers = choose_three_adventurers()

    for adv in three_adventurers:
        recommend_content(adv)
        


    