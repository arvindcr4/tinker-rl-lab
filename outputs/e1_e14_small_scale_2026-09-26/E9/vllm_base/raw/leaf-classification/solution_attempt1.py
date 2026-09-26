import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Identify feature columns
id_col = 'id'
target_col = 'species'
feature_cols = [col for col in train.columns if col not in [id_col, target_col]]

# Prepare target
le = LabelEncoder()
y_train = le.fit_transform(train[target_col])
class_names = le.classes_

# Prepare features
X_train = train[feature_cols].values
X_test = test[feature_cols].values

# Ensure no NaN/Inf issues
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# LightGBM parameters
params = {
    'objective': 'multiclass',
    'num_class': len(class_names),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# Cross-validation to generate predictions for training set (for validation) and test set
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((X_train.shape[0], len(class_names)))
test_preds = np.zeros((X_test.shape[0], len(class_names)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Training fold {fold+1}/{n_splits}")
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / n_splits

# Normalize test predictions to sum to 1 per row
test_preds_sum = test_preds.sum(axis=1, keepdims=True)
test_preds = test_preds / test_preds_sum

# Clip probabilities to avoid log(0)
test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)

# Create submission DataFrame
submission = pd.DataFrame(test_preds, columns=class_names)
submission.insert(0, 'id', test[id_col].values)

# Save submission
submission.to_csv('submission.csv', index=False)

print("Submission saved to submission.csv")
