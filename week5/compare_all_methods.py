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

print(f"Generating recommendations for {len(test_users)} users...")
print("="*60)

# Generate recommendations from all 3 methods
results = {
    'collaborative': [],
    'content_based': [],  # Use your hybrid's content component
    'heuristic': []
}

error_counts = {'collaborative': 0, 'content_based': 0, 'heuristic': 0}

for i, user_id in enumerate(test_users, 1):
    print(f"\n[{i}/{len(test_users)}] User: {user_id}")
    
    # Collaborative (your baseline)
    try:
        collab = recommend_baseline(user_id, n_recs=2)
        results['collaborative'].append({
            'adventurer_id': user_id,
            'rec1': collab[0] if len(collab) > 0 else None,
            'rec2': collab[1] if len(collab) > 1 else None
        })
        print(f"  ✓ Collaborative: {collab[:2]}")
    except Exception as e:
        print(f"  ✗ Collaborative error: {e}")
        error_counts['collaborative'] += 1
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
        print(f"  ✓ Content-based: {content[:2]}")
    except Exception as e:
        print(f"  ✗ Content-based error: {e}")
        error_counts['content_based'] += 1
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
        print(f"  ✓ Heuristic: {heur[:2]}")
    except Exception as e:
        print(f"  ✗ Heuristic error: {e}")
        error_counts['heuristic'] += 1
        results['heuristic'].append({
            'adventurer_id': user_id, 'rec1': None, 'rec2': None
        })

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

# Save all three
for method, data in results.items():
    df = pd.DataFrame(data)
    df.to_csv(P(f'{method}_eval.csv'), index=False)
    
    # Count successful recommendations
    success = df[['rec1', 'rec2']].notna().sum().sum()
    total = len(df) * 2
    
    print(f"\n{method.upper()}:")
    print(f"  ✓ Saved {method}_eval.csv")
    print(f"  ✓ Successful: {success}/{total} recommendations ({success/total*100:.1f}%)")
    if error_counts[method] > 0:
        print(f"  ⚠️  Errors: {error_counts[method]} users")

print("\n" + "="*60)
print("✅ Done! Now run: python evaluate_all_methods.py")
print("="*60)