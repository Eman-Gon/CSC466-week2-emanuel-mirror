import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("COMPREHENSIVE EVALUATION - ALL METHODS")
print("="*60)

# Load ground truth data
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))

# Calculate watch_pct for ground truth
df_merged = df_views.merge(df_metadata[['content_id', 'minutes']], on='content_id', how='left')
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / (df_merged['minutes'] * 60)).clip(0, 1)

# Define what counts as "liked" - using 50% threshold
WATCH_THRESHOLD = 0.5

def get_user_likes(user_id):
    """Get content that user actually liked (watched >50%)"""
    user_views = df_merged[df_merged['adventurer_id'] == user_id]
    liked = user_views[user_views['watch_pct'] >= WATCH_THRESHOLD]['content_id'].unique()
    return set(liked)

def evaluate_recommendations(csv_file, method_name):
    """Evaluate a recommendation CSV file"""
    try:
        recs_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"\n❌ {csv_file} not found")
        return None
    
    total_users = len(recs_df)
    precision_scores = []
    recall_scores = []
    
    for _, row in recs_df.iterrows():
        user_id = row['adventurer_id']
        recommended = []
        
        if pd.notna(row.get('rec1')):
            recommended.append(row['rec1'])
        if pd.notna(row.get('rec2')):
            recommended.append(row['rec2'])
        
        if len(recommended) == 0:
            continue
            
        # Get what user actually likes
        liked = get_user_likes(user_id)
        
        if len(liked) == 0:
            continue
        
        # Calculate precision and recall
        hits = len(set(recommended) & liked)
        precision = hits / len(recommended) if len(recommended) > 0 else 0
        recall = hits / len(liked) if len(liked) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    if len(precision_scores) == 0:
        return None
    
    results = {
        'method': method_name,
        'precision@2': np.mean(precision_scores),
        'recall@2': np.mean(recall_scores),
        'users_evaluated': len(precision_scores)
    }
    
    return results

# Evaluate all methods
methods = [
    ('collaborative_eval.csv', 'Collaborative Filtering'),
    ('content_based_eval.csv', 'Content-Based'),
    ('heuristic_eval.csv', 'Global Heuristic'),
    ('post_eval.csv', 'Hybrid (60/40)'),
    ('pre_eval.csv', 'Baseline KNN')
]

all_results = []

for csv_file, method_name in methods:
    result = evaluate_recommendations(P(csv_file), method_name)
    if result:
        all_results.append(result)

# Print results
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('precision@2', ascending=False)

print(f"\n{'Method':<25} {'Precision@2':<15} {'Recall@2':<15} {'Users':<10}")
print("-"*65)

for _, row in results_df.iterrows():
    print(f"{row['method']:<25} {row['precision@2']:<15.3f} {row['recall@2']:<15.3f} {row['users_evaluated']:<10.0f}")

# Calculate improvement over random baseline (assume random = 2%)
random_baseline = 0.02

print("\n" + "="*60)
print("IMPROVEMENT OVER RANDOM BASELINE (2%)")
print("="*60)

for _, row in results_df.iterrows():
    improvement = ((row['precision@2'] - random_baseline) / random_baseline) * 100
    print(f"{row['method']:<25} +{improvement:>6.0f}%")

# Diversity analysis
print("\n" + "="*60)
print("DIVERSITY ANALYSIS")
print("="*60)

for csv_file, method_name in methods:
    try:
        recs_df = pd.read_csv(P(csv_file))
        unique_items = set()
        for col in ['rec1', 'rec2']:
            if col in recs_df.columns:
                unique_items.update(recs_df[col].dropna().unique())
        
        total_items = df_metadata['content_id'].nunique()
        coverage = len(unique_items) / total_items * 100
        
        print(f"{method_name:<25} Unique: {len(unique_items):>3} / {total_items} ({coverage:>5.1f}%)")
    except:
        pass

print("\n" + "="*60)
print("COPY THIS OUTPUT TO YOUR REPORT!")
print("="*60)
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("COMPREHENSIVE EVALUATION - ALL METHODS")
print("="*60)

# Load ground truth data
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))

# Calculate watch_pct for ground truth
df_merged = df_views.merge(df_metadata[['content_id', 'minutes']], on='content_id', how='left')
df_merged['watch_pct'] = (df_merged['seconds_viewed'] / (df_merged['minutes'] * 60)).clip(0, 1)

# Define what counts as "liked" - using 50% threshold
WATCH_THRESHOLD = 0.5

def get_user_likes(user_id):
    """Get content that user actually liked (watched >50%)"""
    user_views = df_merged[df_merged['adventurer_id'] == user_id]
    liked = user_views[user_views['watch_pct'] >= WATCH_THRESHOLD]['content_id'].unique()
    return set(liked)

def evaluate_recommendations(csv_file, method_name):
    """Evaluate a recommendation CSV file"""
    try:
        recs_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"\n❌ {csv_file} not found")
        return None
    
    total_users = len(recs_df)
    precision_scores = []
    recall_scores = []
    
    for _, row in recs_df.iterrows():
        user_id = row['adventurer_id']
        recommended = []
        
        if pd.notna(row.get('rec1')):
            recommended.append(row['rec1'])
        if pd.notna(row.get('rec2')):
            recommended.append(row['rec2'])
        
        if len(recommended) == 0:
            continue
            
        # Get what user actually likes
        liked = get_user_likes(user_id)
        
        if len(liked) == 0:
            continue
        
        # Calculate precision and recall
        hits = len(set(recommended) & liked)
        precision = hits / len(recommended) if len(recommended) > 0 else 0
        recall = hits / len(liked) if len(liked) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    if len(precision_scores) == 0:
        return None
    
    results = {
        'method': method_name,
        'precision@2': np.mean(precision_scores),
        'recall@2': np.mean(recall_scores),
        'users_evaluated': len(precision_scores)
    }
    
    return results

# Evaluate all methods
methods = [
    ('collaborative_eval.csv', 'Collaborative Filtering'),
    ('content_based_eval.csv', 'Content-Based'),
    ('heuristic_eval.csv', 'Global Heuristic'),
    ('post_eval.csv', 'Hybrid (60/40)'),
    ('pre_eval.csv', 'Baseline KNN')
]

all_results = []

for csv_file, method_name in methods:
    result = evaluate_recommendations(P(csv_file), method_name)
    if result:
        all_results.append(result)

# Print results
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('precision@2', ascending=False)

print(f"\n{'Method':<25} {'Precision@2':<15} {'Recall@2':<15} {'Users':<10}")
print("-"*65)

for _, row in results_df.iterrows():
    print(f"{row['method']:<25} {row['precision@2']:<15.3f} {row['recall@2']:<15.3f} {row['users_evaluated']:<10.0f}")

# Calculate improvement over random baseline (assume random = 2%)
random_baseline = 0.02

print("\n" + "="*60)
print("IMPROVEMENT OVER RANDOM BASELINE (2%)")
print("="*60)

for _, row in results_df.iterrows():
    improvement = ((row['precision@2'] - random_baseline) / random_baseline) * 100
    print(f"{row['method']:<25} +{improvement:>6.0f}%")

# Diversity analysis
print("\n" + "="*60)
print("DIVERSITY ANALYSIS")
print("="*60)

for csv_file, method_name in methods:
    try:
        recs_df = pd.read_csv(P(csv_file))
        unique_items = set()
        for col in ['rec1', 'rec2']:
            if col in recs_df.columns:
                unique_items.update(recs_df[col].dropna().unique())
        
        total_items = df_metadata['content_id'].nunique()
        coverage = len(unique_items) / total_items * 100
        
        print(f"{method_name:<25} Unique: {len(unique_items):>3} / {total_items} ({coverage:>5.1f}%)")
    except:
        pass

print("\n" + "="*60)
print("COPY THIS OUTPUT TO YOUR REPORT!")
print("="*60)