import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Define authors
authors = ['EAP', 'HPL', 'MWS']

# Feature Engineering: Text length and basic stats
def add_text_features(df):
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['avg_word_length'] = df['text_length'] / (df['word_count'] + 1)
    df['num_sentences'] = df['text'].str.count(r'[.!?]+')
    df['num_exclamations'] = df['text'].str.count(r'!')
    df['num_question_marks'] = df['text'].str.count(r'\?')
    df['num_capitals'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['num_punctuation'] = df['text'].str.count(r'[^\w\s]')
    return df

train = add_text_features(train)
test = add_text_features(test)

# TF-IDF Features
# Use unigrams and bigrams
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english'
)

# Fit on train, transform both
X_train_tfidf = tfidf.fit_transform(train['text'])
X_test_tfidf = tfidf.transform(test['text'])

# Combine numerical features
num_features = ['text_length', 'word_count', 'avg_word_length', 'num_sentences', 
                'num_exclamations', 'num_question_marks', 'num_capitals', 'num_punctuation']

X_train_num = train[num_features].values
X_test_num = test[num_features].values

# Combine TF-IDF and numerical features
from scipy.sparse import hstack, csr_matrix

X_train = hstack([X_train_tfidf, csr_matrix(X_train_num)])
X_test = hstack([X_test_tfidf, csr_matrix(X_test_num)])

# Encode target
y = train['author'].map({'EAP': 0, 'HPL': 1, 'MWS': 2})

# LightGBM Model
params = {
    'objective': 'multiclass',
    'num_class': 3,
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

# Cross-validation to get out-of-fold predictions for training set (optional, but good for checking)
# And to generate final predictions via averaging if desired. 
# For simplicity and speed, we will train on all data and predict.
# However, to avoid overfitting and get better probabilities, let's use a simple ensemble or just LGBM.

# Train LightGBM
train_data = lgb.Dataset(X_train, label=y)
test_data = lgb.Dataset(X_test, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Predict probabilities
probs = model.predict(X_test)

# Ensure probabilities are valid (clip to avoid log(0))
probs = np.clip(probs, 1e-15, 1 - 1e-15)

# Normalize rows to sum to 1 (though logloss is invariant to scaling, it's good practice)
row_sums = probs.sum(axis=1, keepdims=True)
probs = probs / row_sums

# Create submission DataFrame
submission = pd.DataFrame(probs, columns=authors)
submission['id'] = test['id']

# Reorder columns to match sample submission
submission = submission[['id', 'EAP', 'HPL', 'MWS']]

# Save submission
submission.to_csv('submission.csv', index=False)

print("Submission saved to submission.csv")
print(submission.head())
