import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Features
feature_cols = [
    'spacegroup',
    'number_of_total_atoms',
    'percent_atom_al',
    'percent_atom_ga',
    'percent_atom_in',
    'lattice_vector_1_ang',
    'lattice_vector_2_ang',
    'lattice_vector_3_ang',
    'lattice_angle_alpha_degree',
    'lattice_angle_beta_degree',
    'lattice_angle_gamma_degree'
]

# Target columns
target_cols = ['formation_energy_ev_natom', 'bandgap_energy_ev']

# Prepare features
X_train = train[feature_cols]
y_train = train[target_cols]
X_test = test[feature_cols]

# Encode spacegroup as categorical
# LightGBM can handle categorical features, but for simplicity and robustness,
# we can treat it as a numerical feature or use ordinal encoding.
# Given the small number of unique spacegroups, ordinal encoding is fine.
# However, LightGBM natively supports categorical features if specified.
# Let's use ordinal encoding for compatibility with all sklearn models if needed,
# but LightGBM is specified. We will let LightGBM handle it as a categorical feature
# by specifying it in the model, or just treat it as numeric since the spacegroup
# numbers are somewhat ordered by crystal system (cubic, tetragonal, etc.).
# Actually, spacegroup is nominal. Let's use one-hot encoding or just treat as numeric.
# Treating as numeric might introduce false ordering. Let's use one-hot encoding.
X_train_encoded = pd.get_dummies(X_train, columns=['spacegroup'], prefix='sg')
X_test_encoded = pd.get_dummies(X_test, columns=['spacegroup'], prefix='sg')

# Ensure test has same columns as train
missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for col in missing_cols:
    X_test_encoded[col] = 0
X_test_encoded = X_test_encoded[X_train_encoded.columns]

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train_encoded, y_train, test_size=0.1, random_state=42)

# Train LightGBM for each target
models = {}
for target in target_cols:
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr[target],
        eval_set=[(X_val, y_val[target])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    models[target] = model

# Predict on test set
predictions = {}
for target in target_cols:
    predictions[target] = models[target].predict(X_test_encoded)

# Create submission
submission = sample_submission.copy()
submission['formation_energy_ev_natom'] = predictions['formation_energy_ev_natom']
submission['bandgap_energy_ev'] = predictions['bandgap_energy_ev']

# Save submission
submission.to_csv('submission.csv', index=False)

print("Submission saved to submission.csv")
