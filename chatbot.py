import json
import random
import re
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download("punkt")

# ✅ Load dataset
with open("C:/Users/Moksh/OneDrive/Desktop/internship.py/intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

# ✅ Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ✅ Prepare training data
corpus = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(preprocess(pattern))
        labels.append(intent["tag"])

# ✅ Train the Naive Bayes model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(corpus, labels)

# ✅ Response function
def get_response(user_input):
    processed = preprocess(user_input)
    predicted_tag = model.predict([processed])[0]

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

# ✅ Chat loop (terminal mode)
print("🤖 Mind Reader is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break
    response = get_response(user_input)
    print("Bot:", response)
