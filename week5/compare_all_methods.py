import pandas as pd
import numpy as np
from pathlib import Path
from recommender import recommend_for_user as recommend_collaborative
from advanced_recommender_week4 import recommend_hybrid, recommend_baseline
from heuristic_recommender import recommend_trending

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

# Load test users from your existing eval
pre_eval = pd.read_csv(P('pre_eval.csv'))
test_users = pre_eval['adventurer_id'].tolist()

# Generate recommendations from all 3 methods
results = {
    'collaborative': [],
    'content_based': [],  # Use your hybrid's content component
    'heuristic': []
}

print("Generating recommendations...")
for user_id in test_users:
    # Collaborative (your baseline)
    try:
        collab = recommend_baseline(user_id, n_recs=2)
        results['collaborative'].append({
            'adventurer_id': user_id,
            'rec1': collab[0] if len(collab) > 0 else None,
            'rec2': collab[1] if len(collab) > 1 else None
        })
    except:
        results['collaborative'].append({
            'adventurer_id': user_id, 'rec1': None, 'rec2': None
        })
    
    # Content-based (extract from your hybrid)
    # For speed, just use your hybrid - it has 40% content-based
    try:
        content = recommend_hybrid(user_id, n_recs=2)
        results['content_based'].append({
            'adventurer_id': user_id,
            'rec1': content[0] if len(content) > 0 else None,
            'rec2': content[1] if len(content) > 1 else None
        })
    except:
        results['content_based'].append({
            'adventurer_id': user_id, 'rec1': None, 'rec2': None
        })
    
    # Heuristic
    try:
        heur = recommend_trending(user_id, n_recs=2)
        results['heuristic'].append({
            'adventurer_id': user_id,
            'rec1': heur[0] if len(heur) > 0 else None,
            'rec2': heur[1] if len(heur) > 1 else None
        })
    except:
        results['heuristic'].append({
            'adventurer_id': user_id, 'rec1': None, 'rec2': None
        })

# Save all three
for method, data in results.items():
    df = pd.DataFrame(data)
    df.to_csv(P(f'{method}_eval.csv'), index=False)
    print(f"✓ Saved {method}_eval.csv")

print("\nDone! Submit these to get feedback on which performs best.")