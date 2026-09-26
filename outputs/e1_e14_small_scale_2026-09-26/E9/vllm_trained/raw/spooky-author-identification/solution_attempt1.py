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

# --- Feature Engineering ---

# 1. Basic Text Features
def extract_features(df):
    df['text_len'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['avg_word_len'] = df['text_len'] / (df['word_count'] + 1)
    df['punct_count'] = df['text'].apply(lambda x: sum(1 for c in x if c in '.,!?;:\'"()'))
    df['upper_count'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['digit_count'] = df['text'].apply(lambda x: sum(1 for c in x if c.isdigit()))
    return df

train = extract_features(train)
test = extract_features(test)

# 2. TF-IDF Features
# Use both character n-grams and word n-grams for better authorship attribution
tfidf_char = TfidfVectorizer(
    analyzer='char_wb', 
    ngram_range=(3, 5), 
    max_features=50000, 
    min_df=2, 
    sublinear_tf=True
)

tfidf_word = TfidfVectorizer(
    analyzer='word', 
    ngram_range=(1, 2), 
    max_features=50000, 
    min_df=2, 
    sublinear_tf=True
)

# Fit on combined data to handle unseen words in test set
all_text = pd.concat([train['text'], test['text']], ignore_index=True)

X_train_char = tfidf_char.fit_transform(all_text[:len(train)])
X_test_char = tfidf_char.transform(all_text[len(train):])

X_train_word = tfidf_word.fit_transform(all_text[:len(train)])
X_test_word = tfidf_word.transform(all_text[len(train):])

# Combine features
X_train = hstack([X_train_char, X_train_word, train.iloc[:, 2:].values])
X_test = hstack([X_test_char, X_test_word, test.iloc[:, 2:].values])

y_train = train['author']

# --- Model Training ---

# Use Logistic Regression as it works well with TF-IDF and is fast
# We will use stratified k-fold cross validation to get good probability estimates
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train), len(authors)))
test_preds = np.zeros((len(test), len(authors)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Training fold {fold+1}")
    
    X_tr = X_train[train_idx]
    y_tr = y_train.iloc[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train.iloc[val_idx]
    
    # Create label encoder mapping
    label_map = {author: i for i, author in enumerate(authors)}
    y_tr_encoded = y_tr.map(label_map)
    y_val_encoded = y_val.map(label_map)
    
    # Train Logistic Regression
    # C is inverse regularization strength. Higher C means less regularization.
    # We tune this slightly for performance.
    model = LogisticRegression(
        C=1.0, 
        solver='lbfgs', 
        max_iter=1000, 
        multi_class='multinomial',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr_encoded)
    
    # Predict probabilities
    oof_preds[val_idx] = model.predict_proba(X_val)
    test_preds += model.predict_proba(X_test) / 5

# Clip probabilities to avoid log(0)
epsilon = 1e-15
oof_preds = np.clip(oof_preds, epsilon, 1 - epsilon)
test_preds = np.clip(test_preds, epsilon, 1 - epsilon)

# Normalize test predictions so they sum to 1 (optional but good practice for log loss)
# The problem statement says they are rescaled prior to scoring, but normalization helps stability.
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Create submission DataFrame
submission = sample_submission.copy()
submission[authors] = test_preds

# Save submission
submission.to_csv('submission.csv', index=False)

print("Submission saved to submission.csv")
print(f"Sample submission:\n{submission.head()}")
