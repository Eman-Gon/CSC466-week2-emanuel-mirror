import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
ROOT = Path.cwd()
P = lambda name: ROOT / name

def validate_submission(filename='eval.csv'):
    """
    Validate the competition submission file
    """
    print("="*60)
    print(f"VALIDATING SUBMISSION: {filename}")
    print("="*60)
    
    # Check file exists
    if not P(filename).exists():
        print(f"❌ ERROR: {filename} not found!")
        return False
    
    # Load submission
    try:
        df = pd.read_csv(P(filename))
        print(f"✓ File loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        return False
    
    # Load data for validation
    print("\nLoading validation data...")
    df_views = pd.read_parquet(P("content_views.parquet"))
    df_metadata = pd.read_parquet(P("content_metadata.parquet"))
    df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))
    
    valid_users = set(df_adventurers['adventurer_id'].unique())
    valid_content = set(df_metadata['content_id'].unique())
    
    print(f"  Valid users: {len(valid_users):,}")
    print(f"  Valid content: {len(valid_content):,}")
    
    # Run validation checks
    print("\n" + "="*40)
    print("REQUIREMENT CHECKS")
    print("="*40)
    
    passed = []
    failed = []
    
    # Check 1: Column names
    required_cols = ['adventurer_id', 'rec1', 'rec2', 'rec3']
    if list(df.columns) == required_cols:
        print("✓ Correct column names")
        passed.append("columns")
    else:
        print(f"❌ Wrong columns. Expected: {required_cols}, Got: {list(df.columns)}")
        failed.append("columns")
    
    # Check 2: Number of rows
    if len(df) == 30:
        print("✓ Exactly 30 adventurers")
        passed.append("row_count")
    else:
        print(f"❌ Wrong number of rows. Expected: 30, Got: {len(df)}")
        failed.append("row_count")
    
    # Check 3: No duplicate adventurers
    duplicates = df['adventurer_id'].duplicated().sum()
    if duplicates == 0:
        print("✓ No duplicate adventurer_ids")
        passed.append("no_duplicates")
    else:
        print(f"❌ Found {duplicates} duplicate adventurer_ids")
        failed.append("duplicates")
    
    # Check 4: No missing values
    missing_total = df.isna().sum().sum()
    if missing_total == 0:
        print("✓ No missing values")
        passed.append("no_missing")
    else:
        for col in required_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"❌ {missing} missing values in {col}")
        failed.append("missing_values")
    
    # Check 5: Valid adventurer IDs
    invalid_users = set(df['adventurer_id']) - valid_users
    if len(invalid_users) == 0:
        print("✓ All adventurer_ids are valid")
        passed.append("valid_users")
    else:
        print(f"❌ {len(invalid_users)} invalid adventurer_ids: {list(invalid_users)[:5]}")
        failed.append("invalid_users")
    
    # Check 6: Valid content IDs
    all_recs = set()
    for col in ['rec1', 'rec2', 'rec3']:
        all_recs.update(df[col].dropna())
    
    invalid_content = all_recs - valid_content
    if len(invalid_content) == 0:
        print("✓ All content recommendations are valid")
        passed.append("valid_content")
    else:
        print(f"❌ {len(invalid_content)} invalid content_ids: {list(invalid_content)[:5]}")
        failed.append("invalid_content")
    
    # Check 7: No duplicate recommendations per user
    duplicates_per_user = 0
    for _, row in df.iterrows():
        recs = [row['rec1'], row['rec2'], row['rec3']]
        if len(recs) != len(set(recs)):
            duplicates_per_user += 1
    
    if duplicates_per_user == 0:
        print("✓ No duplicate recommendations per user")
        passed.append("no_rec_duplicates")
    else:
        print(f"⚠ {duplicates_per_user} users have duplicate recommendations")
        # This is a warning, not a failure
    
    print("\n" + "="*40)
    print("QUALITY METRICS")
    print("="*40)
    
    # Diversity analysis
    unique_items = len(all_recs)
    total_recs = len(df) * 3
    coverage = unique_items / len(valid_content) * 100
    
    print(f"Unique items recommended: {unique_items}/{total_recs} possible")
    print(f"Catalog coverage: {unique_items}/{len(valid_content)} ({coverage:.1f}%)")
    print(f"Diversity ratio: {unique_items/total_recs:.2f}")
    
    # Quality indicators
    if unique_items >= 20:
        print("✓ Good diversity (20+ unique items)")
    elif unique_items >= 10:
        print("⚠ Moderate diversity (10-19 unique items)")
    else:
        print("⚠ Low diversity (<10 unique items) - might hurt score")
    
    # Popular items check
    popular_items = df_views['content_id'].value_counts().head(3).index.tolist()
    popular_count = sum([all_recs.count(item) for item in popular_items])
    print(f"\nPopular items usage: {popular_count}/{total_recs} ({popular_count/total_recs*100:.1f}%)")
    if popular_count/total_recs > 0.5:
        print("⚠ Over-relying on popular items")
    
    print("\n" + "="*40)
    print("FINAL VERDICT")
    print("="*40)
    
    if len(failed) == 0:
        print("✅ SUBMISSION IS VALID AND READY!")
        print(f"   All {len(passed)} checks passed")
        print(f"   Diversity: {unique_items} unique items")
        print(f"   Coverage: {coverage:.1f}% of catalog")
        return True
    else:
        print("❌ SUBMISSION HAS ISSUES")
        print(f"   Passed: {len(passed)} checks")
        print(f"   Failed: {len(failed)} checks - {failed}")
        return False
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Validate the main submission
    print("Checking eval.csv...")
    is_valid = validate_submission('eval.csv')
    
    # If eval.csv doesn't exist or is invalid, check the simple version
    if not is_valid and P('eval_simple.csv').exists():
        print("\n\nChecking eval_simple.csv as backup...")
        validate_submission('eval_simple.csv')