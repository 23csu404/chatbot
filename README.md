# Chatbot using CNN & Intent Classification

This is a simple AI-powered chatbot that uses a Convolutional Neural Network (CNN) for classifying user intents and generating appropriate responses.

## 🔧 Features

- Intent classification using a CNN model (`cnn_model.h5`)
- Intents stored in a structured JSON format (`intents.json`)
- Web interface via Flask (`chatbot_web.py`)
- Model logic in `chatbot.py`

## 🚀 How to Run

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the chatbot:
```bash
python chatbot_web.py
```

## 📁 Project Files

| File             | Purpose                            |
|------------------|------------------------------------|
| `chatbot.py`     | Loads the model and handles user input |
| `chatbot_web.py` | Provides a simple web interface     |
| `cnn_model.h5`   | Trained CNN model                   |
| `intents.json`   | Intent dataset with tags & responses|

## 🧠 How It Works

The chatbot reads user input, classifies it using the trained CNN model, and returns an appropriate response based on the predicted intent from `intents.json`.

## 📜 License

MIT – Free to use, modify, and share.
