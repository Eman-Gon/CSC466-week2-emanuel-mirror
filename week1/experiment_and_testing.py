
from typing import List
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import csv


if __name__ == "__main__":
    content_views = pd.read_parquet('./week1/content_views.parquet')
    content_metadata = pd.read_parquet('./week1/content_metadata.parquet')
    adventurers = pd.read_parquet('./week1/adventurer_metadata.parquet')

    views_with_demo = content_views.merge(
        adventurers[['adventurer_id', 'age', 'gender']], 
        on='adventurer_id', 
        how='left'
    )

    agg_demo = views_with_demo.groupby('content_id').agg(
        avg_age=('age', 'mean'),
        gender_ratio=('gender', lambda x: ((x == 'M')*1 + (x == 'NB')*0.5).mean())
    ).reset_index()


    content_metadata = content_metadata.merge(agg_demo, on='content_id', how='left')

    with open('week1/eval.csv', 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        header = next(csvreader)
        print(f"Header: {header}")

        for i, row in enumerate(csvreader):
            adv = row[0]
     

            print(f"Adventurer {i + 1}: {adv}")
            print(adventurers.loc[adventurers['adventurer_id'] == adv])
            viewed_content = content_views.loc[content_views['adventurer_id'] == adv].merge(content_metadata, on="content_id", how="left")
            viewed_content['watch_percentage'] = (viewed_content['seconds_viewed'] / (viewed_content['minutes'] * 60)).clip(0,1)
            print(viewed_content.sort_values(by='watch_percentage')[['content_id', 'avg_age','gender_ratio','genre_id','language_code','playlist_id','watch_percentage']])

            print("Suggestions")
            suggestion1 = content_metadata.loc[content_metadata['content_id'] == row[1]]
            
            suggestion2 = content_metadata.loc[content_metadata['content_id'] == row[2]]
            
            print(suggestion1[['content_id', 'avg_age','gender_ratio','genre_id','language_code']])
            print(suggestion2[['content_id', 'avg_age','gender_ratio','genre_id','language_code']])
            print()
            
        
        


    