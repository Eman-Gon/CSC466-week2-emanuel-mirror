
from typing import List
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve,  roc_curve, auc
import csv
import matplotlib.pyplot as plt

def get_y(rows, threshold):
    content_views = pd.read_parquet('./week2/content_views.parquet')
    content_metadata = pd.read_parquet('./week2/content_metadata.parquet')
    adventurers = pd.read_parquet('./week2/adventurer_metadata.parquet')

    y_actual = []
    y_pred = []

    for i, row in enumerate(rows):
        adv = row[0]
        content = row[1:]

        #print(f"Adventurer {i + 1}: {adv}")
        #print(adventurers.loc[adventurers['adventurer_id'] == adv])

        content_views_train = content_views.iloc[0:int((len(content_views)*0.8))]
        content_views_test = content_views.drop(content_views_train.index)
        
        test_content = content_views_test.loc[content_views_test['adventurer_id'] == adv].merge(content_metadata, on="content_id", how="left")

        test_content['watch_percentage'] = (test_content['seconds_viewed'] / (test_content['minutes'] * 60)).clip(0,1)
        
        #print(test_content.sort_values(ascending=False,by='watch_percentage')[['content_id', 'genre_id','language_code','playlist_id','watch_percentage']])
        test_content = test_content.drop_duplicates(subset="content_id", keep="first")

        # actual labels
        test_content['top_half_actual'] = 0
        n = int(len(test_content) * threshold)
        test_content.loc[test_content.index[:n], 'top_half_actual'] = 1

        # predicted labels
        test_content['top_half_pred'] = 0
        top_half_rec_cont = content[:int(len(content) * 0.5)]
        test_content.loc[test_content['content_id'].isin(top_half_rec_cont), 'top_half_pred'] = 1


        #print(test_content[['content_id','watch_percentage','top_half_actual','top_half_pred']])
        
        y_actual += test_content['top_half_actual'].tolist()
        y_pred  += test_content['top_half_pred'].tolist()
    f1 = f1_score(y_actual, y_pred, average="binary")

    return y_actual, y_pred,f1





if __name__ == "__main__":


    with open('week2/hahns-eval.csv', 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        header = next(csvreader)
        print(f"Header: {header}")

        y_actual = []
        y_pred = []

        rows = []
        for i, row in enumerate(csvreader):
            print(row)
            rows.append(row)
        
        test_thresholds = np.linspace(0.9,0.1,9)
        print(test_thresholds)
        for t in test_thresholds:
            print(get_y(rows, t))
            
           


           

    
             

        


"""
        print(y_actual)
        print(y_pred)
        print(y_scores)

        # -- F1 Score --
        f1_binary = f1_score(y_actual, y_pred, average="binary")
        print(f"F1 Score = {f1_binary}")
        
        # -- ROC-AUC --
        fpr, tpr, _ = roc_curve(y_actual, y_pred)
        roc_auc= roc_auc_score(y_actual, y_pred)
        print(f"ROC-AUC = {roc_auc}")

        # -- PR-AUC --
        precision, recall, thresholds = precision_recall_curve(y_actual, y_pred)
        pr_auc = auc(recall, precision)
        print(f"PR-AUC = {pr_auc}")

        # -- Plot -- 
        plt.figure(figsize=(12,5))

        # Plot ROC
        
        plt.subplot(1,2,1)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], linestyle='--', color='gray') 
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

        # Plot PR
        plt.subplot(1,2,2)
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # F1 Threshold
   
            








         
        print()
        """
            
        
        


    