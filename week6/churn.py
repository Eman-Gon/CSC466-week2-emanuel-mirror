import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
P = lambda name: ROOT / name

MONTH_ORDER = ["Frostmere", "Emberfall", "Lunaris", "Verdantia", "Solstice",
               "Duskveil", "Starshade", "Aurorath", "Mysthaven", "Eclipsion"]
MONTH_TO_INDEX = {m: i for i, m in enumerate(MONTH_ORDER)}
DAYS_PER_MONTH = 24

CURRENT_YEAR = 10235  
CURRENT_MONTH = "Verdantia" 
CURRENT_DAY = 1  


print("CHURN PREDICTION - Week 6 (OPTIMIZED + FIXED)")

print(f"Current date: {CURRENT_MONTH} {CURRENT_DAY}, {CURRENT_YEAR}")
print(f"Prediction target: Next {DAYS_PER_MONTH} days")


def date_to_ordinal(year, month, day):
    month_idx = MONTH_TO_INDEX.get(month, 0)
    return year * (len(MONTH_ORDER) * DAYS_PER_MONTH) + month_idx * DAYS_PER_MONTH + (day - 1)



print("\n[1] Loading data")
df_subs = pd.read_parquet(P("subscriptions.parquet"))
df_cancels = pd.read_parquet(P("cancellations.parquet"))
df_views = pd.read_parquet(P("content_views.parquet"))
df_metadata = pd.read_parquet(P("content_metadata.parquet"))
df_adventurers = pd.read_parquet(P("adventurer_metadata.parquet"))

print(f"   Subscriptions: {len(df_subs):,}")
print(f"   Cancellations: {len(df_cancels):,}")
print(f"   Content Views: {len(df_views):,}")
print(f"   Adventurers: {len(df_adventurers):,}")


print("\n[2] Preprocessing data")

df_subs['sub_ordinal'] = df_subs.apply(
    lambda x: date_to_ordinal(x['year'], x['month'], x['day_of_month']), axis=1
)
df_cancels['cancel_ordinal'] = df_cancels.apply(
    lambda x: date_to_ordinal(x['year'], x['month'], x['day_of_month']), axis=1
)
df_views['view_ordinal'] = df_views.apply(
    lambda x: date_to_ordinal(x['year'], x['month'], x['day_of_month']), axis=1
)

df_subs = df_subs.merge(
    df_cancels[['adventurer_id', 'publisher_id', 'cancel_ordinal']],
    on=['adventurer_id', 'publisher_id'],
    how='left'
)

current_ordinal = date_to_ordinal(CURRENT_YEAR, CURRENT_MONTH, CURRENT_DAY)

train_cutoff = current_ordinal - (DAYS_PER_MONTH * 2)

print(f"   Current ordinal: {current_ordinal}")
print(f"   Training cutoff: {train_cutoff} (2 months back)")



print("\n[3] Creating labels")

active_at_train = df_subs[
    (df_subs['sub_ordinal'] <= train_cutoff) &
    ((df_subs['cancel_ordinal'].isna()) | (df_subs['cancel_ordinal'] > train_cutoff))
].copy()

active_at_train['churn'] = (
    (active_at_train['cancel_ordinal'].notna()) &
    (active_at_train['cancel_ordinal'] > train_cutoff) &
    (active_at_train['cancel_ordinal'] <= train_cutoff + DAYS_PER_MONTH)
).astype(int)

print(f"   Active before dedup: {len(active_at_train):,}")
active_at_train = active_at_train.sort_values('sub_ordinal').groupby(
    ['adventurer_id', 'publisher_id'], as_index=False
).last()

print(f"   Active at training cutoff: {len(active_at_train):,}")
print(f"   Churn rate: {active_at_train['churn'].mean():.2%}")


print("\n[4] Engineering features (vectorized approach)")

def engineer_features_fast(df_active, cutoff_ordinal):
    """
    Fast vectorized feature engineering using pandas groupby
    """
    features = df_active[['adventurer_id', 'publisher_id', 'sub_ordinal']].copy()
    
    print("   [4.1] Subscription features")
    features['days_subbed'] = cutoff_ordinal - features['sub_ordinal']
    
    sub_counts = df_subs[df_subs['sub_ordinal'] <= cutoff_ordinal].groupby(
        ['adventurer_id', 'publisher_id']
    ).size().reset_index(name='num_subscriptions')
    features = features.merge(sub_counts, on=['adventurer_id', 'publisher_id'], how='left')
    features['num_subscriptions'] = features['num_subscriptions'].fillna(1)
    
    user_subs = df_subs[
        (df_subs['sub_ordinal'] <= cutoff_ordinal) &
        (df_subs['cancel_ordinal'].notna())
    ].copy()
    user_subs['sub_length'] = user_subs['cancel_ordinal'] - user_subs['sub_ordinal']
    avg_sub_length = user_subs.groupby('adventurer_id')['sub_length'].mean().reset_index()
    avg_sub_length.columns = ['adventurer_id', 'avg_sub_length_user']
    features = features.merge(avg_sub_length, on='adventurer_id', how='left')
    features['avg_sub_length_user'] = features['avg_sub_length_user'].fillna(0)
    
    print("   [4.2] Engagement features")

    view_features = []
    for idx, row in features.iterrows():
        user_views = df_views[
            (df_views['adventurer_id'] == row['adventurer_id']) &
            (df_views['view_ordinal'] >= row['sub_ordinal']) &
            (df_views['view_ordinal'] <= cutoff_ordinal)
        ]
        
        view_features.append({
            'adventurer_id': row['adventurer_id'],
            'publisher_id': row['publisher_id'],
            'num_content_viewed': len(user_views),
            'total_seconds_viewed': user_views['seconds_viewed'].sum() if len(user_views) > 0 else 0,
            'median_seconds_viewed': user_views['seconds_viewed'].median() if len(user_views) > 0 else 0,
            'days_since_last_view': cutoff_ordinal - user_views['view_ordinal'].max() if len(user_views) > 0 else row['days_subbed']
        })
        
        if idx % 5000 == 0:
            print(f"      Progress: {idx:,}/{len(features):,}")
    
    view_df = pd.DataFrame(view_features)
    features = features.merge(view_df, on=['adventurer_id', 'publisher_id'], how='left')
    print("   [4.3] User demographics")
    features = features.merge(
        df_adventurers[['adventurer_id', 'age']],
        on='adventurer_id',
        how='left'
    )
    features['age'] = features['age'].fillna(25)
    
    print("   [4.4] Publisher features")
    pub_stats = df_subs[df_subs['sub_ordinal'] <= cutoff_ordinal].groupby('publisher_id').agg({
        'cancel_ordinal': lambda x: x.notna().mean()
    }).reset_index()
    pub_stats.columns = ['publisher_id', 'pub_churn_rate']
    features = features.merge(pub_stats, on='publisher_id', how='left')
    features['pub_churn_rate'] = features['pub_churn_rate'].fillna(0.5)

    pub_cancelled = df_subs[
        (df_subs['sub_ordinal'] <= cutoff_ordinal) &
        (df_subs['cancel_ordinal'].notna())
    ].copy()
    pub_cancelled['sub_length'] = pub_cancelled['cancel_ordinal'] - pub_cancelled['sub_ordinal']
    pub_avg = pub_cancelled.groupby('publisher_id')['sub_length'].mean().reset_index()
    pub_avg.columns = ['publisher_id', 'pub_avg_sub_length']
    features = features.merge(pub_avg, on='publisher_id', how='left')
    features['pub_avg_sub_length'] = features['pub_avg_sub_length'].fillna(30)
    
    return features

train_features = engineer_features_fast(active_at_train, train_cutoff)
train_features['churn'] = active_at_train['churn'].values

print(f"   Feature matrix: {train_features.shape}")
print(f"   Labels match: {len(train_features) == len(active_at_train)}")


print("\n[5] Baseline model")

y_train = train_features['churn']
baseline_pred = np.ones(len(y_train))

print(f"   Accuracy: {(baseline_pred == y_train).mean():.3f}")
print(f"   F1 Score: {f1_score(y_train, baseline_pred):.3f}")


print("\n[6] Training models")

feature_cols = [c for c in train_features.columns 
                if c not in ['adventurer_id', 'publisher_id', 'churn', 'sub_ordinal']]
X_train = train_features[feature_cols].fillna(0)
y_train = train_features['churn']

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

print("\n   [Model 1] Logistic Regression")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_tr_scaled, y_tr)
lr_pred = lr_model.predict(X_val_scaled)
lr_prob = lr_model.predict_proba(X_val_scaled)[:, 1]
print(f"     Accuracy: {(lr_pred == y_val).mean():.3f}")
print(f"     F1 Score: {f1_score(y_val, lr_pred):.3f}")
print(f"     ROC-AUC: {roc_auc_score(y_val, lr_prob):.3f}")

print("\n   [Model 2] Random Forest")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_tr, y_tr)
rf_pred = rf_model.predict(X_val)
rf_prob = rf_model.predict_proba(X_val)[:, 1]
print(f"     Accuracy: {(rf_pred == y_val).mean():.3f}")
print(f"     F1 Score: {f1_score(y_val, rf_pred):.3f}")
print(f"     ROC-AUC: {roc_auc_score(y_val, rf_prob):.3f}")

print("\n   Top 5 Features:")
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importances.head(5).to_string(index=False))


print("\n[7] Making final predictions")

active_now = df_subs[
    (df_subs['sub_ordinal'] <= current_ordinal) &
    ((df_subs['cancel_ordinal'].isna()) | (df_subs['cancel_ordinal'] > current_ordinal))
].copy()

print(f"   Active before dedup: {len(active_now):,}")
active_now = active_now.sort_values('sub_ordinal').groupby(
    ['adventurer_id', 'publisher_id'], as_index=False
).last()

print(f"   Currently active: {len(active_now):,}")

current_features = engineer_features_fast(active_now, current_ordinal)
X_current = current_features[feature_cols].fillna(0)

final_predictions = rf_model.predict_proba(X_current)[:, 1]
current_features['churn_probability'] = final_predictions
current_features['predicted_churn'] = (final_predictions > 0.5).astype(int)

predicted_churners = current_features[current_features['predicted_churn'] == 1]

print(f"   Predicted churners: {len(predicted_churners):,}")
print(f"   Predicted churn rate: {len(predicted_churners) / len(current_features):.2%}")

print("\n[8] Saving predictions")

output = predicted_churners[['adventurer_id', 'publisher_id']].copy()
output.to_csv(P('churn_pred.csv'), index=False)
print(f"   ✓ Saved {len(output):,} predictions to churn_pred.csv")

detailed = current_features[['adventurer_id', 'publisher_id', 'churn_probability', 'predicted_churn']].copy()
detailed.to_csv(P('churn_predictions_detailed.csv'), index=False)
