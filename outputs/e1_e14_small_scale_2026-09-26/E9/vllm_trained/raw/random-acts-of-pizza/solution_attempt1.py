import json
import pandas as pd
import numpy as np
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
    'post_was_edited',
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
    'unix_timestamp_of_request'
]

# Text features to engineer
text_cols = ['request_text', 'request_title', 'request_text_edit_aware']

# Categorical features
cat_cols = ['giver_username_if_known', 'requester_user_flair', 'requester_username']

# Function to extract text features
def extract_text_features(df):
    df = df.copy()
    
    # Length features
    for col in text_cols:
        df[f'{col}_len'] = df[col].str.len()
        df[f'{col}_word_count'] = df[col].str.split().str.len()
        df[f'{col}_char_count'] = df[col].str.count(r'\S')
        
    # Title specific
    df['title_has_brackets'] = df['request_title'].str.contains(r'\[.*\]').astype(int)
    df['title_has_request'] = df['request_title'].str.contains(r'(?i)request').astype(int)
    
    # Text specific
    df['text_has_request'] = df['request_text'].str.contains(r'(?i)request').astype(int)
    df['text_has_pizza'] = df['request_text'].str.contains(r'(?i)pizza').astype(int)
    df['text_has_money'] = df['request_text'].str.contains(r'(?i)money|cash|pay|paycheck|broke|poor|unemployed|job|work').astype(int)
    df['text_has_emergency'] = df['request_text'].str.contains(r'(?i)emergency|desperate|need|help|crisis|hard|tough|struggle').astype(int)
    df['text_has_politeness'] = df['request_text'].str.contains(r'(?i)please|thank|thanks|grateful|appreciate|kind|nice').astype(int)
    df['text_has_trade'] = df['request_text'].str.contains(r'(?i)trade|swap|exchange|give|offer|willing').astype(int)
    df['text_has_location'] = df['request_text'].str.contains(r'(?i)ca\.|usa|california|new york|texas|florida|ny|tx|fl').astype(int)
    df['text_has_military'] = df['request_text'].str.contains(r'(?i)military|army|navy|air force|marine|deploy|deployment|war|afghanistan|iraq').astype(int)
    df['text_has_student'] = df['request_text'].str.contains(r'(?i)student|college|university|school|class|exam|homework').astype(int)
    df['text_has_medical'] = df['request_text'].str.contains(r'(?i)hospital|sick|ill|health|medical|doctor|nurse|pain|injury').astype(int)
    df['text_has_food_related'] = df['request_text'].str.contains(r'(?i)hungry|eat|food|meal|dinner|lunch|breakfast|starving').astype(int)
    df['text_has_gaming'] = df['request_text'].str.contains(r'(?i)game|gaming|console|pc|playstation|xbox|steam|level|account').astype(int)
    df['text_has_exclamation'] = df['request_text'].str.contains(r'!').astype(int)
    df['text_has_question'] = df['request_text'].str.contains(r'\?').astype(int)
    df['text_has_capital_ratio'] = df['request_text'].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    df['text_has_lowercase_ratio'] = df['request_text'].apply(lambda x: sum(1 for c in x if c.islower()) / max(len(x), 1))
    
    return df

print("Extracting text features...")
train_df = extract_text_features(train_df)
test_df = extract_text_features(test_df)

# Handle categorical features
# Encode giver_username_if_known
train_df['giver_is_known'] = (train_df['giver_username_if_known'] != 'N/A').astype(int)
test_df['giver_is_known'] = (test_df['giver_username_if_known'] != 'N/A').astype(int)

# Encode requester_user_flair
flair_map = {'shroom': 1, 'PIF': 2, None: 0, 'N/A': 0}
train_df['flair_encoded'] = train_df['requester_user_flair'].map(flair_map).fillna(0).astype(int)
test_df['flair_encoded'] = test_df['requester_user_flair'].map(flair_map).fillna(0).astype(int)

# Create interaction features
train_df['upvote_ratio'] = train_df['number_of_upvotes_of_request_at_retrieval'] / (train_df['number_of_upvotes_of_request_at_retrieval'] + train_df['number_of_downvotes_of_request_at_retrieval'] + 1)
test_df['upvote_ratio'] = test_df['number_of_upvotes_of_request_at_retrieval'] / (test_df['number_of_upvotes_of_request_at_retrieval'] + test_df['number_of_downvotes_of_request_at_retrieval'] + 1)

train_df['comment_ratio'] = train_df['request_number_of_comments_at_retrieval'] / (train_df['number_of_upvotes_of_request_at_retrieval'] + 1)
test_df['comment_ratio'] = test_df['request_number_of_comments_at_retrieval'] / (test_df['number_of_upvotes_of_request_at_retrieval'] + 1)

train_df['account_age_ratio'] = train_df['requester_account_age_in_days_at_request'] / (train_df['requester_account_age_in_days_at_retrieval'] + 1)
test_df['account_age_ratio'] = test_df['requester_account_age_in_days_at_request'] / (test_df['requester_account_age_in_days_at_retrieval'] + 1)

train_df['posts_per_subreddit'] = train_df['requester_number_of_posts_at_request'] / (train_df['requester_number_of_subreddits_at_request'] + 1)
test_df['posts_per_subreddit'] = test_df['requester_number_of_posts_at_request'] / (test_df['requester_number_of_subreddits_at_request'] + 1)

# Select final features
exclude_cols = ['request_id', 'requester_received_pizza', 'requester_username', 'requester_subreddits_at_request', 'giver_username_if_known', 'requester_user_flair']
feature_cols = [c for c in train_df.columns if c not in exclude_cols and train_df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'uint8', 'bool']]

# Ensure all features are present in test set
for col in feature_cols:
    if col not in test_df.columns:
        test_df[col] = 0

print(f"Number of features: {len(feature_cols)}")

# Prepare data
X_train = train_df[feature_cols].fillna(0)
y_train = train_df['requester_received_pizza'].astype(int)
X_test = test_df[feature_cols].fillna(0)

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

# Cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

print("Training with LightGBM...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)]
    )
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

# Calculate overall OOF AUC
oof_auc = roc_auc_score(y_train, oof_preds)
print(f"Overall OOF AUC: {oof_auc:.4f}")

# Create submission
submission = pd.DataFrame({
    'request_id': test_df['request_id'],
    'requester_received_pizza': test_preds
})

# Clip predictions to [0, 1]
submission['requester_received_pizza'] = submission['requester_received_pizza'].clip(0, 1)

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
print(f"Submission shape: {submission.shape}")
print(submission.head())
