import pandas as pd
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load spaCy English tokenizer
nlp = spacy.load('en_core_web_sm')

# Example function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Load your email dataset
df = pd.read_csv('Spam Email raw text for NLP.csv')

# Preprocess the text data
df['processed_text'] = df['MESSAGE'].apply(preprocess_text)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['CATEGORY'], test_size=0.2, random_state=42)

# Convert text to vectors
def text_to_vector(text):
    doc = nlp(text)
    vectors = np.array([token.vector for token in doc if not token.is_stop and not token.is_punct])
    if vectors.size == 0:
        return np.zeros(nlp.vocab.vectors.shape[1])
    return np.mean(vectors, axis=0)

X_train_vectors = np.array([text_to_vector(text) for text in X_train])
X_test_vectors = np.array([text_to_vector(text) for text in X_test])

# Create and train the model
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

# Make predictions
y_pred = model.predict(X_test_vectors)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))
