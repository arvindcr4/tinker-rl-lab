import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
import warnings
import gc

warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Clean data
print("Cleaning data...")
# Remove rows with missing comments in training set
train = train.dropna(subset=['Comment'])
train['Comment'] = train['Comment'].astype(str)
test['Comment'] = test['Comment'].astype(str)

# Basic text cleaning: lowercase, remove extra whitespace
train['Comment'] = train['Comment'].str.lower().str.replace(r'\s+', ' ', regex=True)
test['Comment'] = test['Comment'].str.lower().str.replace(r'\s+', ' ', regex=True)

# Feature Engineering: Text Features
print("Extracting TF-IDF features...")
# Use a moderate number of features to keep it fast and memory efficient
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents='unicode'
)

# Fit on train, transform both
X_train_tfidf = tfidf.fit_transform(train['Comment'])
X_test_tfidf = tfidf.transform(test['Comment'])

# Additional simple features
def extract_features(df):
    features = pd.DataFrame()
    features['comment_length'] = df['Comment'].str.len()
    features['word_count'] = df['Comment'].str.split().str.len()
    features['exclamation_count'] = df['Comment'].str.count('!')
    features['question_count'] = df['Comment'].str.count(r'\?')
    features['uppercase_ratio'] = df['Comment'].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    features['has_numbers'] = df['Comment'].apply(lambda x: 1 if any(c.isdigit() for c in x) else 0)
    features['has_urls'] = df['Comment'].apply(lambda x: 1 if 'http' in x or 'www' in x else 0)
    features['has_mentions'] = df['Comment'].apply(lambda x: 1 if '@' in x else 0)
    features['has_hashtags'] = df['Comment'].apply(lambda x: 1 if '#' in x else 0)
    features['avg_word_length'] = df['Comment'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
    features['punctuation_ratio'] = df['Comment'].apply(lambda x: sum(1 for c in x if c in '.,;:!?') / max(len(x), 1))
    return features

print("Extracting additional features...")
X_train_extra = extract_features(train)
X_test_extra = extract_features(test)

# Convert sparse matrix to dense for LightGBM if memory allows, or use sparse directly
# LightGBM supports sparse matrices directly
print("Preparing data for LightGBM...")
y_train = train['Insult']

# Combine TF-IDF and extra features
# LightGBM can handle sparse matrices, but we need to stack them
from scipy.sparse import hstack

X_train = hstack([X_train_tfidf, X_train_extra])
X_test = hstack([X_test_tfidf, X_test_extra])

# Convert to CSR format for efficient row slicing
X_train = X_train.tocsr()
X_test = X_test.tocsr()

# Define LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
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
    'scale_pos_weight': 1  # Adjust if class imbalance is severe
}

# Check class balance
print(f"Class distribution: {y_train.value_counts().to_dict()}")
# If imbalanced, adjust scale_pos_weight
if y_train.sum() < len(y_train) * 0.5:
    params['scale_pos_weight'] = (len(y_train) - y_train.sum()) / y_train.sum()

# Cross-validation to get out-of-fold predictions and estimate performance
print("Starting Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Fold {fold+1}/5")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / 5

# Calculate OOF log loss
oof_log_loss = log_loss(y_train, oof_preds)
print(f"OOF Log Loss: {oof_log_loss}")

# Final predictions on test set
submission = pd.DataFrame({
    'Insult': test_preds
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
