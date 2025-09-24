
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
    """Convert the mystical calendar date to an absolute ordinal."""
    month_index = MONTH_TO_INDEX[month]   
    return year * (10 * 24) + month_index * 24 + (day - 1)

def find_top_publisher() -> str:
    """Returns the publisher with the most amount of content."""
    content_views = pd.read_parquet('content_views.parquet')
    counts = content_views.groupby("publisher_id")["content_id"].nunique().reset_index(name="content_count") 
    top_publisher = counts.loc[counts["content_count"].idxmax()]
    return top_publisher['publisher_id']
    

def choose_three_adventurers():
    """Returns a list of three adventurers. This function picks adventurers subscribed to the publisher with the most content."""

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

    # Find the three adventurers with the most amount of views of content from the 'top publisher' :P
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
    """Returns a list of content_ids for recommended content."""
    # Read parquet files
    content_metadata = pd.read_parquet('content_metadata.parquet')
    content_views = pd.read_parquet('content_views.parquet')
    adventurers = pd.read_parquet('adventurer_metadata.parquet')

    # Add more columns (avg age and gender ratio)
    views = content_views.merge(adventurers[['adventurer_id','age','gender']], on='adventurer_id', how='left')
    demographics = views.groupby('content_id').agg(
        avg_age=('age','mean'),
        gender_ratio=('gender', lambda x: (x=='M').mean())
    ).reset_index()
    content_metadata = pd.merge(content_metadata, demographics, on='content_id', how='left')

    # Get unseen content
    adventurer_views = content_views.loc[content_views['adventurer_id'] == adventurer_id]
    viewed_content = pd.merge(adventurer_views, content_metadata, on=['content_id'],how='left')
    
    viewed_content['watch_percentage'] = (viewed_content['seconds_viewed'] / (viewed_content['minutes'] * 60)).clip(0,1)

    unseen_content = content_metadata.loc[(~content_metadata['content_id'].isin(viewed_content))].copy()
    
    # Turn categorical columns to numerical columns
    numeric_cols = ['avg_age','gender_ratio'] 
    categorical_features = ['genre_id', 'language_code', 'studio']
    for col in categorical_features:
        content_metadata[col] = content_metadata[col].astype('category').cat.codes
        viewed_content[col] = viewed_content[col].astype('category').cat.codes
        unseen_content[col] = unseen_content[col].astype('category').cat.codes
        numeric_cols.append(col)


    X_train = viewed_content[numeric_cols].fillna(0).multiply(viewed_content['watch_percentage'], axis=0).to_numpy()
    X_test = unseen_content[numeric_cols].fillna(0).to_numpy()

    neigh = NearestNeighbors(n_neighbors=2, metric='euclidean')
    neigh.fit(X_train)

    distances, indices = neigh.kneighbors(X_test)

    top_indices = np.argsort(distances.mean(axis=1))[:2]  
    recommended_content = unseen_content.iloc[top_indices]['content_id'].tolist()

    return recommended_content




if __name__ == "__main__":
    three_adventurers = choose_three_adventurers()

    with open("eval.csv", "w") as f:
        print("adventurer_id,rec1,rec2")
        f.write("adventurer_id,rec1,rec2\n")
        for adv in three_adventurers:
            res = recommend_content(adv)
            new_str = adv + "," + ",".join(res)
            f.write(new_str + "\n")
            print(new_str)
            
        
        


    