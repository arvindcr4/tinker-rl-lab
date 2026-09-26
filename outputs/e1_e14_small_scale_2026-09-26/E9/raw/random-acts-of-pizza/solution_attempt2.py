import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load data
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

print("Loading data...")
train_data = load_json('train.json')
test_data = load_json('test.json')

# Convert to DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Identify numeric and categorical columns
numeric_cols = [
    'number_of_downvotes_of_request_at_retrieval',
    'number_of_upvotes_of_request_at_retrieval',
    'request_number_of_comments_at_retrieval',
    'requester_account_age_in_days_at_request',
    'requester_account_age_in_days_at_retrieval',
    'requester_days_since_first_post_on_raop_at_request',
    'requester_days_since_first_post_on_raop_at_retrieval',
    'requester_number_of_comments_at_request',
    'requester_number_of_comments_at_retrieval',
    'requester_number_of_comments_in_raop_at_request',
    'requester_number_of_comments_in_raop_at_retrieval',
    'requester_number_of_posts_at_request',
    'requester_number_of_posts_at_retrieval',
    'requester_number_of_posts_on_raop_at_request',
    'requester_number_of_posts_on_raop_at_retrieval',
    'requester_number_of_subreddits_at_request',
    'requester_upvotes_minus_downvotes_at_request',
    'requester_upvotes_minus_downvotes_at_retrieval',
    'requester_upvotes_plus_downvotes_at_request',
    'requester_upvotes_plus_downvotes_at_retrieval',
    'post_was_edited'
]

# post_was_edited is boolean, convert to int if it exists
if 'post_was_edited' in train_df.columns:
    train_df['post_was_edited'] = train_df['post_was_edited'].astype(int)
if 'post_was_edited' in test_df.columns:
    test_df['post_was_edited'] = test_df['post_was_edited'].astype(int)

# Handle missing values in numeric columns by filling with 0 or median
for col in numeric_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(0)

# Feature engineering from text
def extract_text_features(df):
    # Ensure text columns exist
    if 'request_title' not in df.columns:
        df['request_title'] = ''
    if 'request_text_edit_aware' not in df.columns:
        df['request_text_edit_aware'] = ''
        
    df['title_length'] = df['request_title'].str.len()
    df['text_length'] = df['request_text_edit_aware'].str.len()
    df['title_word_count'] = df['request_title'].str.split().str.len()
    df['text_word_count'] = df['request_text_edit_aware'].str.split().str.len()
    
    # Check for specific keywords that might indicate success
    df['has_trade'] = df['request_text_edit_aware'].str.lower().str.contains('trade|swap|exchange', na=False).astype(int)
    df['has_deployment'] = df['request_text_edit_aware'].str.lower().str.contains('deploy|military|army|navy|air force|marines', na=False).astype(int)
    df['has_medical'] = df['request_text_edit_aware'].str.lower().str.contains('hospital|sick|ill|injury|pain|doctor|nurse', na=False).astype(int)
    df['has_emergency'] = df['request_text_edit_aware'].str.lower().str.contains('emergency|desperate|broke|no money|homeless', na=False).astype(int)
    df['has_politeness'] = df['request_text_edit_aware'].str.lower().str.contains('please|thank|thanks|grateful|appreciate', na=False).astype(int)
    df['has_story'] = df['request_text_edit_aware'].str.lower().str.contains('story|because|since|when|while', na=False).astype(int)
    
    # Title features
    df['title_has_request'] = df['request_title'].str.lower().str.contains('request', na=False).astype(int)
    df['title_has_location'] = df['request_title'].str.lower().str.contains('ca|usa|uk|canada|australia|ny|la|chicago|houston|phoenix|philadelphia|san antonio|san diego|dallas|san jose|austin|jacksonville|fort worth|columbus|charlotte|san francisco|indianapolis|seattle|denver|washington|boston|el paso|detroit|nashville|portland|memphis|oklahoma city|las vegas|louisville|baltimore|milwaukee|albuquerque|tucson|fresno|sacramento|mesa|virginia beach|atlanta|colorado springs|omaha|raleigh|miami|long beach|kansas city|virginia beach|atlanta|colorado springs|omaha|raleigh|miami|long beach|kansas city', na=False).astype(int)
    
    return df

print("Extracting text features...")
train_df = extract_text_features(train_df)
test_df = extract_text_features(test_df)

# Prepare features
feature_cols = numeric_cols + [
    'title_length', 'text_length', 'title_word_count', 'text_word_count',
    'has_trade', 'has_deployment', 'has_medical', 'has_emergency', 'has_politeness', 'has_story',
    'title_has_request', 'title_has_location'
]

# Ensure all feature columns exist in both dataframes
for col in feature_cols:
    if col not in train_df.columns:
        train_df[col] = 0
    if col not in test_df.columns:
        test_df[col] = 0

X_train = train_df[feature_cols].values
y_train = train_df['requester_received_pizza'].values
X_test = test_df[feature_cols].values

# LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'min_child_samples': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42
}

# Cross-validation to get predictions and evaluate
print("Training with LightGBM...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    train_data_lgb = lgb.Dataset(X_tr, label=y_tr)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
    
    model = lgb.train(
        params,
        train_data_lgb,
        num_boost_round=1000,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / 5
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

# Calculate overall OOF AUC
print(f"Overall OOF AUC: {roc_auc_score(y_train, oof_preds):.4f}")

# Clip predictions to [0, 1]
test_preds = np.clip(test_preds, 0, 1)

# Create submission
submission = pd.DataFrame({
    'request_id': test_df['request_id'],
    'requester_received_pizza': test_preds
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
