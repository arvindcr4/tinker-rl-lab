import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Define authors
authors = ['EAP', 'HPL', 'MWS']

# Feature Engineering: Text features
# 1. TF-IDF features
# Using a combination of unigrams and bigrams, with specific parameters for short text
tfidf = TfidfVectorizer(
    analyzer='char_wb', 
    ngram_range=(3, 5), 
    min_df=2, 
    max_df=0.95, 
    sublinear_tf=True,
    max_features=50000
)

# Fit on train and transform both
X_train_tfidf = tfidf.fit_transform(train['text'])
X_test_tfidf = tfidf.transform(test['text'])

# 2. Additional text statistics
def add_text_features(df):
    df['text_len'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['avg_word_len'] = df['text_len'] / (df['word_count'] + 1)
    df['num_caps'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['num_punct'] = df['text'].apply(lambda x: sum(1 for c in x if c in '.,!?;:'))
    df['num_digits'] = df['text'].apply(lambda x: sum(1 for c in x if c.isdigit()))
    return df

train = add_text_features(train)
test = add_text_features(test)

# Extract numeric features
numeric_cols = ['text_len', 'word_count', 'avg_word_len', 'num_caps', 'num_punct', 'num_digits']
X_train_num = train[numeric_cols].values
X_test_num = test[numeric_cols].values

# Convert numeric features to sparse matrix to stack with TF-IDF
X_train_num_sparse = csr_matrix(X_train_num)
X_test_num_sparse = csr_matrix(X_test_num)

# Combine features
X_train = hstack([X_train_tfidf, X_train_num_sparse])
X_test = hstack([X_test_tfidf, X_test_num_sparse])

# Target encoding
y_train = train['author']

# Model: Logistic Regression with L2 regularization
# LR works very well on TF-IDF features for this type of problem
model = LogisticRegression(
    C=1.0, 
    solver='lbfgs', 
    multi_class='multinomial', 
    max_iter=1000,
    class_weight='balanced'
)

# Cross-validation to get out-of-fold predictions for better calibration and to check performance
# Although for submission we just need the final model, OOF helps in tuning if needed.
# Here we will train on full data for final submission as is standard for simple pipelines.

# Train final model
model.fit(X_train, y_train)

# Predict probabilities for test set
probs = model.predict_proba(X_test)

# Create submission dataframe
submission = sample_submission.copy()
submission[authors] = probs

# Ensure probabilities are clipped to avoid log(0) issues in scoring
submission[authors] = submission[authors].clip(1e-15, 1 - 1e-15)

# Save submission
submission.to_csv('submission.csv', index=False)

print("Submission saved to submission.csv")
print(submission.head())
