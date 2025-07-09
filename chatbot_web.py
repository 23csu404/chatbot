import streamlit as st
import json
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

with open("C:/Users/Moksh/OneDrive/Desktop/internship.py/intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)



# Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# Prepare training data
corpus = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(preprocess(pattern))
        labels.append(intent["tag"])

# Train model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(corpus, labels)

# Get response
def get_response(user_input):
    processed = preprocess(user_input)
    predicted_tag = model.predict([processed])[0]

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

# Web UI
st.set_page_config(page_title="Mind Reader Chatbot")
st.title("🤖 Mind Reader Chatbot")
st.markdown("Ask me anything about your college — admission, fees, hostel, and more!")

user_input = st.text_input("You:", placeholder="Type your message here...")

if user_input:
    response = get_response(user_input)
    st.markdown(f"**Bot:** {response}")
