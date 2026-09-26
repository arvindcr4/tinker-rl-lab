import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Clean data
print("Cleaning data...")
# Handle missing values
train['Comment'] = train['Comment'].fillna('')
test['Comment'] = test['Comment'].fillna('')

# Remove quotes if present (common in this dataset)
train['Comment'] = train['Comment'].str.strip('"')
test['Comment'] = test['Comment'].str.strip('"')

# Basic text cleaning
def clean_text(text):
    # Replace newlines and tabs with spaces
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

train['Comment'] = train['Comment'].apply(clean_text)
test['Comment'] = test['Comment'].apply(clean_text)

# Feature Engineering: TF-IDF
print("Extracting TF-IDF features...")
# Use a moderate ngram range to balance performance and speed
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents='unicode',
    token_pattern=r'(?u)\b\w+\b'
)

# Fit on train and transform both
X_train_tfidf = tfidf.fit_transform(train['Comment'])
X_test_tfidf = tfidf.transform(test['Comment'])

print(f"TF-IDF shape: {X_train_tfidf.shape}")

# Prepare labels
y_train = train['Insult'].values

# LightGBM Model
print("Training LightGBM model...")

# Define parameters tuned for this type of text classification
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# Cross-validation to get out-of-fold predictions for evaluation and final model
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tfidf, y_train)):
    print(f"Training fold {fold+1}/5...")
    
    X_tr = X_train_tfidf[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train_tfidf[val_idx]
    y_val = y_train[val_idx]
    
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
    test_preds += model.predict(X_test_tfidf) / 5

# Clip predictions to avoid log(0)
oof_preds = np.clip(oof_preds, 1e-7, 1 - 1e-7)
test_preds = np.clip(test_preds, 1e-7, 1 - 1e-7)

# Calculate OOF Log Loss
loss = log_loss(y_train, oof_preds)
print(f"Out-of-Fold Log Loss: {loss:.4f}")

# Create submission
print("Creating submission...")
submission = pd.DataFrame({
    'Insult': test_preds.round(6)
})

# Ensure the submission format matches the sample submission
# The sample submission has 'Insult' as the first column
submission.to_csv('submission.csv', index=False)

print("Submission saved to submission.csv")
