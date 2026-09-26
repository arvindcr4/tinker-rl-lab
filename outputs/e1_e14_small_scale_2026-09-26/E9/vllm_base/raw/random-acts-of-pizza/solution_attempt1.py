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

# Identify target
y_train = (train_df['requester_received_pizza'] == True).astype(int)

# Define feature columns (exclude non-feature columns)
exclude_cols = ['request_id', 'requester_received_pizza', 'giver_username_if_known', 
                'requester_subreddits_at_request', 'requester_user_flair']

# Text columns
text_cols = ['request_text', 'request_text_edit_aware', 'request_title']

# List all numeric columns
numeric_cols = [col for col in train_df.columns if col not in exclude_cols and col not in text_cols]

# Check if all numeric columns are present in test set
# The problem statement says: "We have removed fields from the test set which would not be available at the time of posting."
# So we need to find the intersection of numeric columns between train and test
train_numeric_cols = set(numeric_cols)
test_numeric_cols = set([col for col in test_df.columns if col not in exclude_cols and col not in text_cols])
common_numeric_cols = list(train_numeric_cols.intersection(test_numeric_cols))

print(f"Common numeric columns: {len(common_numeric_cols)}")

# Function to extract text features
def extract_text_features(df):
    features = pd.DataFrame()
    
    # Length features
    for col in text_cols:
        if col in df.columns:
            features[f'{col}_len'] = df[col].fillna('').apply(len)
            features[f'{col}_word_count'] = df[col].fillna('').apply(lambda x: len(str(x).split()))
            features[f'{col}_avg_word_len'] = df[col].fillna('').apply(lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0)
    
    # Specific keyword features
    keywords = ['please', 'thank', 'desperate', 'hungry', 'food', 'money', 'job', 'deploy', 'military', 
                'student', 'homeless', 'sick', 'ill', 'hospital', 'family', 'kids', 'children', 'baby',
                'veteran', 'marine', 'army', 'navy', 'airforce', 'coastguard', 'police', 'fire', 'nurse',
                'doctor', 'teacher', 'worker', 'hardworking', 'honest', 'good', 'nice', 'kind', 'help',
                'need', 'want', 'love', 'like', 'appreciate', 'grateful', 'blessed', 'god', 'jesus',
                'christian', 'muslim', 'jewish', 'atheist', 'agnostic', 'religion', 'faith', 'hope',
                'dream', 'goal', 'ambition', 'future', 'past', 'present', 'time', 'day', 'night',
                'morning', 'evening', 'weekend', 'holiday', 'birthday', 'anniversary', 'wedding',
                'funeral', 'death', 'life', 'death', 'survive', 'live', 'die', 'kill', 'murder',
                'crime', 'law', 'justice', 'court', 'judge', 'jury', 'prison', 'jail', 'cell',
                'free', 'freedom', 'liberty', 'equality', 'rights', 'human', 'people', 'world',
                'country', 'city', 'town', 'village', 'home', 'house', 'apartment', 'room', 'bed',
                'sleep', 'rest', 'relax', 'chill', 'cool', 'hot', 'cold', 'warm', 'weather', 'rain',
                'snow', 'sun', 'cloud', 'wind', 'storm', 'thunder', 'lightning', 'flood', 'fire',
                'earthquake', 'tsunami', 'volcano', 'disaster', 'emergency', 'crisis', 'problem',
                'issue', 'trouble', 'difficulty', 'challenge', 'obstacle', 'barrier', 'wall', 'door',
                'window', 'key', 'lock', 'open', 'close', 'start', 'end', 'begin', 'finish', 'complete',
                'done', 'ready', 'set', 'go', 'move', 'stop', 'wait', 'pause', 'continue', 'resume',
                'restart', 'repeat', 'again', 'once', 'twice', 'thrice', 'many', 'few', 'some', 'all',
                'none', 'every', 'each', 'both', 'either', 'neither', 'one', 'two', 'three', 'four',
                'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', 'million', 'billion',
                'zero', 'first', 'second', 'third', 'last', 'next', 'previous', 'current', 'latest',
                'new', 'old', 'young', 'ancient', 'modern', 'classic', 'traditional', 'custom', 'habit',
                'routine', 'schedule', 'plan', 'strategy', 'tactic', 'method', 'way', 'path', 'road',
                'street', 'avenue', 'boulevard', 'lane', 'drive', 'court', 'place', 'square', 'park',
                'garden', 'yard', 'field', 'farm', 'ranch', 'estate', 'property', 'land', 'soil', 'dirt',
                'sand', 'gravel', 'rock', 'stone', 'pebble', 'boulder', 'mountain', 'hill', 'valley',
                'canyon', 'gorge', 'cliff', 'ledge', 'ridge', 'peak', 'summit', 'top', 'bottom', 'base',
                'foot', 'head', 'neck', 'shoulder', 'arm', 'hand', 'finger', 'thumb', 'palm', 'wrist',
                'elbow', 'forearm', 'bicep', 'tricep', 'chest', 'back', 'spine', 'rib', 'lung', 'heart',
                'liver', 'kidney', 'stomach', 'intestine', 'bowel', 'colon', 'rectum', 'anus', 'bladder',
                'urine', 'pee', 'poop', 'shit', 'crap', 'waste', 'trash', 'garbage', 'rubbish', 'refuse',
                'recycle', 'reuse', 'reduce', 'green', 'eco', 'environment', 'nature', 'animal', 'plant',
                'tree', 'flower', 'grass', 'leaf', 'root', 'stem', 'branch', 'twig', 'bark', 'wood',
                'paper', 'book', 'read', 'write', 'story', 'tale', 'legend', 'myth', 'fable', 'parable',
                'allegory', 'metaphor', 'simile', 'analogy', 'comparison', 'contrast', 'difference',
                'similarity', 'same', 'different', 'unique', 'special', 'rare', 'common', 'ordinary',
                'normal', 'average', 'typical', 'standard', 'regular', 'usual', 'habitual', 'customary',
                'conventional', 'traditional', 'classic', 'timeless', 'eternal', 'infinite', 'endless',
                'boundless', 'limitless', 'unlimited', 'unrestricted', 'free', 'liberated', 'emancipated',
                'independent', 'autonomous', 'sovereign', 'self-governing', 'self-ruling', 'self-determining',
                'self-directed', 'self-guided', 'self-led', 'self-managed', 'self-controlled', 'self-disciplined',
                'self-motivated', 'self-driven', 'self-starting', 'self-starter', 'self-made', 'self-built',
                'self-created', 'self-designed', 'self-developed', 'self-improved', 'self-enhanced', 'self-optimized',
                'self-perfected', 'self-mastery', 'self-realization', 'self-actualization', 'self-fulfillment',
                'self-satisfaction', 'self-contentment', 'self-happiness', 'self-joy', 'self-pleasure', 'self-delight',
                'self-enjoyment', 'self-amusement', 'self-entertainment', 'self-fun', 'self-play', 'self-game',
                'self-sport', 'self-athletics', 'self-exercise', 'self-workout', 'self-training', 'self-practice',
                'self-rehearsal', 'self-preparation', 'self-readiness', 'self-availability', 'self-accessibility',
                'self-openness', 'self-willingness', 'self-eagerness', 'self-enthusiasm', 'self-passion', 'self-love',
                'self-care', 'self-respect', 'self-esteem', 'self-confidence', 'self-belief', 'self-trust', 'self-faith',
                'self-hope', 'self-optimism', 'self-positivity', 'self-goodness', 'self-kindness', 'self-compassion',
                'self-empathy', 'self-understanding', 'self-awareness', 'self-consciousness', 'self-perception', 'self-image',
                'self-concept', 'self-identity', 'self-persona', 'self-character', 'self-personality', 'self-temperament',
                'self-disposition', 'self-nature', 'self-essence', 'self-soul', 'self-spirit', 'self-mind', 'self-body',
                'self-heart', 'self-spirit', 'self-soul', 'self-self']
    
    for keyword in keywords:
        for col in text_cols:
            if col in df.columns:
                features[f'{col}_{keyword}'] = df[col].fillna('').str.lower().str.contains(keyword, regex=False).astype(int)
    
    return features

print("Extracting text features...")
train_text_features = extract_text_features(train_df)
test_text_features = extract_text_features(test_df)

# Combine numeric and text features
X_train = train_df[common_numeric_cols].copy()
X_train = pd.concat([X_train, train_text_features], axis=1)

X_test = test_df[common_numeric_cols].copy()
X_test = pd.concat([X_test, test_text_features], axis=1)

# Fill NaN values with 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Ensure all columns are present in both train and test
common_cols = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"Number of features: {X_train.shape[1]}")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

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
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

print("Training LightGBM with cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
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
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

# Calculate overall OOF AUC
overall_auc = roc_auc_score(y_train, oof_preds)
print(f"Overall OOF AUC: {overall_auc:.4f}")

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
print(submission.head())
