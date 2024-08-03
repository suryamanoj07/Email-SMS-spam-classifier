import streamlit as st
import pickle
import spacy
import numpy as np
import time

model = pickle.load(open('model2.pkl', 'rb'))

nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text)  # Tokenize and analyze the text
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Function to convert preprocessed text to a numerical vector
def text_to_vector(preprocessed_text):
    doc = nlp(preprocessed_text)  # Tokenize the preprocessed text
    vectors = np.array([token.vector for token in doc if not token.is_stop and not token.is_punct])
    if vectors.size == 0:
        return np.zeros(nlp.vocab.vectors.shape[1])
    return np.mean(vectors, axis=0)

st.title("Email/SMS Spam Classifier")

email_text = st.text_area("Enter / Paste your message here")
result_placeholder = st.empty()

if st.button("Classify"):
    if email_text:
        result_placeholder.empty()
        with st.spinner("Classifying the email..."):
            time.sleep(2)
            preprocessed_text = preprocess_text(email_text)
            vector = text_to_vector(preprocessed_text)  
            prediction = model.predict([vector])
            result = "Spam" if prediction[0] == 1 else "Not Spam"
            st.write(f'The email is predicted to be: "{result.upper()}"')
    else:
        st.write("Please enter some text in the email field.")


footer = """
    <style>
    .footer {
        position: fixed;
        height:40px;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: blue;
        font-weight:900;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        <pre>© Streamlit app || manojsurya463@gmail.com || All rights reserved 2024</pre>
    </div>
"""

# Render the footer
st.markdown(footer, unsafe_allow_html=True)