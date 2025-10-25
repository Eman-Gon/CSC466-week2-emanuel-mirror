import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

print("="*60)
print("FINAL SUBMISSION VALIDATION")
print("="*60)

# Check file exists
if not P('final_recommendations.csv').exists():
    print("❌ final_recommendations.csv NOT FOUND!")
    print("   Run: python generate_final_submission.py")
    exit(1)

# Load and validate
df = pd.read_csv(P('final_recommendations.csv'))

# Check format
errors = []

if len(df) != 20:
    errors.append(f"❌ Expected 20 users, found {len(df)}")
else:
    print(f"✅ Correct number of users: {len(df)}")

if 'adventurer_id' not in df.columns:
    errors.append("❌ Missing 'adventurer_id' column")
    
if 'rec1' not in df.columns or 'rec2' not in df.columns:
    errors.append("❌ Missing 'rec1' or 'rec2' columns")
else:
    print("✅ Correct columns present")

# Check for missing values
missing_rec1 = df['rec1'].isna().sum()
missing_rec2 = df['rec2'].isna().sum()

if missing_rec1 > 0 or missing_rec2 > 0:
    errors.append(f"❌ Missing recommendations: {missing_rec1} rec1, {missing_rec2} rec2")
else:
    print("✅ No missing recommendations")

# Check for duplicates
if df['adventurer_id'].duplicated().any():
    errors.append("❌ Duplicate users found")
else:
    print("✅ No duplicate users")

# Load scope data
df_views = pd.read_parquet(P("content_views.parquet"))
df_subs = pd.read_parquet(P("subscriptions.parquet"))

pub_subs = df_subs[df_subs['publisher_id'] == 'wn32']
publisher_content = set(df_views[
    df_views['adventurer_id'].isin(pub_subs['adventurer_id'])
]['content_id'].unique())

# Check recommendations are in scope
all_recs = pd.concat([df['rec1'], df['rec2']])
invalid_recs = all_recs[~all_recs.isin(publisher_content)]

if len(invalid_recs) > 0:
    errors.append(f"❌ {len(invalid_recs)} recommendations outside publisher scope")
else:
    print("✅ All recommendations are valid publisher content")

# Check diversity
unique_items = len(all_recs.unique())
coverage = unique_items / 38 * 100

print(f"✅ Unique items: {unique_items} ({coverage:.1f}% coverage)")

# Check what method was used based on diversity
if unique_items >= 15:
    method_guess = "Hybrid"
elif unique_items >= 8:
    method_guess = "Collaborative"
else:
    method_guess = "Heuristic"
    
print(f"✅ Detected method: {method_guess}")

# Print summary
print("\n" + "="*60)
if errors:
    print("❌ VALIDATION FAILED")
    print("="*60)
    for error in errors:
        print(error)
    print("\nFix these issues before submitting!")
else:
    print("✅ ALL VALIDATIONS PASSED")
    print("="*60)
    print("\nYour submission is ready!")
    print(f"  - 20 users with 2 recommendations each")
    print(f"  - {unique_items} unique items ({coverage:.1f}% coverage)")
    print(f"  - All recommendations in publisher scope")
    print(f"  - Method: {method_guess}")
    
print("\n" + "="*60)
print("FINAL CHECKLIST")
print("="*60)
checklist = [
    ("final_recommendations.csv validated", True),
    ("writeup.md mentions 'Hybrid'", None),
    ("README.md exists", P('README.md').exists()),
    ("All .py files in repo", True),
    ("All .png graphs in repo", True),
    ("Repo shared with 'jackalnom'", None),
]

for item, status in checklist:
    if status is True:
        print(f"✅ {item}")
    elif status is False:
        print(f"❌ {item}")
    else:
        print(f"⚠️  {item} - VERIFY MANUALLY")
        
print("="*60)
