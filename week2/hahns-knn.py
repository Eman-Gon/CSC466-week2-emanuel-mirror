from typing import List
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

MONTH_ORDER = [
    "Frostmere", "Emberfall", "Lunaris", "Verdantia", "Solstice",
    "Duskveil", "Starshade", "Aurorath", "Mysthaven", "Eclipsion"
]
MONTH_TO_INDEX = {m: i for i, m in enumerate(MONTH_ORDER)}

def mystical_to_ordinal(year: int, month: str, day: int) -> int:
    month_index = MONTH_TO_INDEX[month]
    return year * (10 * 24) + month_index * 24 + (day - 1)

def find_top_publisher() -> str:
    content_views = pd.read_parquet('./week2/content_views.parquet')
    counts = (content_views.groupby("publisher_id")["content_id"]
              .nunique()
              .reset_index(name="content_count"))
    top_publisher = counts.loc[counts["content_count"].idxmax()]
    return top_publisher['publisher_id']

def choose_three_adventurers() -> List[str]:
    subscriptions = pd.read_parquet('./week2/subscriptions.parquet')
    cancellations = pd.read_parquet('./week2/cancellations.parquet')
    content_views = pd.read_parquet('./week2/content_views.parquet')

    subscriptions["ordinal"] = subscriptions.apply(
        lambda r: mystical_to_ordinal(r["year"], r["month"], r["day_of_month"]),
        axis=1
    )
    cancellations["ordinal"] = cancellations.apply(
        lambda r: mystical_to_ordinal(r["year"], r["month"], r["day_of_month"]),
        axis=1
    )

    active_subs = pd.merge(
        subscriptions, cancellations,
        on=['adventurer_id', 'publisher_id'],
        how='outer', suffixes=("_sub", "_cancel")
    )
    active_subs["last_event_date"] = active_subs[["ordinal_sub", "ordinal_cancel"]].max(axis=1)
    active_subs["is_active"] = (active_subs["ordinal_sub"] == active_subs["last_event_date"])
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
    return most_viewed_active['adventurer_id'].astype(str).tolist()

def recommend_content(adventurer_id : str) -> List[str]:
    content_metadata = pd.read_parquet('./week2/content_metadata.parquet')
    content_views = pd.read_parquet('./week2/content_views.parquet')
    adventurers = pd.read_parquet('./week2/adventurer_metadata.parquet')

    split_idx = int(len(content_views) * 0.8)
    content_views_train = content_views.iloc[:split_idx].copy()
    content_views_test  = content_views.iloc[split_idx:].copy()

    cv_train_len = content_views_train.merge(
        content_metadata[['content_id', 'minutes']], on='content_id', how='left'
    )
    denom = (cv_train_len['minutes'].replace(0, np.nan) * 60)
    cv_train_len['watch_pct'] = (cv_train_len['seconds_viewed'] / denom).clip(0, 1).fillna(0.0)

    watchability = (cv_train_len
        .groupby('content_id', as_index=False)
        .agg(avg_watchability=('watch_pct', 'mean'))
    )
    content_metadata = content_metadata.merge(watchability, on='content_id', how='left')
    watch_median = content_metadata['avg_watchability'].median()

    views = content_views_train.merge(
        adventurers[['adventurer_id','age','gender']], on='adventurer_id', how='left'
    )
    demographics = views.groupby('content_id').agg(
        avg_age=('age','mean'),
        gender_ratio=('gender', lambda x: ((x == 'M')*1 + (x == 'NB')*0.5).mean())
    ).reset_index()
    content_metadata = pd.merge(content_metadata, demographics, on='content_id', how='left')

    adventurer_views = content_views_train.loc[content_views_train['adventurer_id'] == adventurer_id]
    viewed_content = pd.merge(adventurer_views, content_metadata, on=['content_id'], how='left')

    denom_user = (viewed_content['minutes'].replace(0, np.nan) * 60)
    viewed_content['watch_percentage'] = (
        viewed_content['seconds_viewed'] / denom_user
    ).clip(0,1).fillna(0.0)

    unseen_content = content_metadata.loc[
        ~content_metadata['content_id'].isin(viewed_content['content_id'])
    ].copy()

    viewed_content_maj = viewed_content[viewed_content['watch_percentage'] > 0.5]
    if len(viewed_content_maj) > 2:
        viewed_content = viewed_content_maj

    cur_adv = adventurers.loc[adventurers['adventurer_id'] == adventurer_id]
    if cur_adv.empty:
        return []
    unseen_content = unseen_content[
        unseen_content['language_code'] == cur_adv.iloc[0]['primary_language']
    ]

    unseen_content = unseen_content[unseen_content['avg_watchability'] >= watch_median]

    print("length of unseen before cutting:", len(unseen_content))
    content_views_test = content_views_test.loc[
        content_views_test['adventurer_id'] == adventurer_id
    ]
    print(content_views_test)
    unseen_content = unseen_content.loc[
        unseen_content['content_id'].isin(content_views_test['content_id'])
    ].copy()
    print(len(unseen_content))

    if unseen_content.empty or viewed_content.empty:
        return []

    numeric_features = ['avg_age','gender_ratio','avg_watchability']
    categorical_features = ['genre_id']

    try:
       encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
     encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)


    encoder.fit(pd.concat([
        viewed_content[categorical_features].astype(str),
        unseen_content[categorical_features].astype(str)
    ]).astype(str))

    viewed_cat = encoder.transform(viewed_content[categorical_features].astype(str))
    unseen_cat = encoder.transform(unseen_content[categorical_features].astype(str))

    X_train = np.hstack([viewed_content[numeric_features].fillna(0).to_numpy(), viewed_cat])
    X_test  = np.hstack([unseen_content[numeric_features].fillna(0).to_numpy(), unseen_cat])

    n_neighbors = min(2, max(1, len(viewed_content)))
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    neigh.fit(X_train)

    distances, _ = neigh.kneighbors(X_test)
    top_indices = np.argsort(distances.mean(axis=1))

    recommended_content = unseen_content.iloc[top_indices]['content_id'].astype(str).drop_duplicates().tolist()
    return recommended_content[:2]

if __name__ == "__main__":
    three_adventurers = choose_three_adventurers()
    with open("./week2/hahns-eval.csv", "w") as f:
        print("adventurer_id,rec1,rec2")
        f.write("adventurer_id,rec1,rec2\n")
        for adv in three_adventurers:
            res = recommend_content(adv)
            new_str = adv + "," + ",".join(res)
            f.write(new_str + "\n")
            print(new_str)