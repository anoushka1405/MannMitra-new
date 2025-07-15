# MannMitra 🧠
*A Conversational Mental Health Assistant*

MannMitra is an AI-powered chatbot designed to offer empathetic, supportive conversations to individuals experiencing stress, anxiety, or emotional overwhelm. Developed as part of the IGDTUW–Sansoftech Generative AI Internship Capstone, the goal is to make emotional support more accessible, private, and always available.

## 🌟 Objectives

- Provide a 24/7 mental health support chatbot.
- Use NLP and LLMs (like Gemini) to generate thoughtful, caring responses.
- Detect the user’s emotional tone and respond accordingly

## 🛠 Tech Stack

- **Python** & **Flask** Web app and backend logic
- **Gemeni API** Generative AI for emotionally intelligent responses
- **Hugging Face Transformers** Emotion detection (GoEmotions model)
- **Render** Cloud Deployment
- **HTML/CSS/JS** Frontend

## 📌 Features

- ✨ Natural, empathetic conversation flow

- 🧠 Emotion classification using pre-trained models

- 🌐 Language detection & Hindi-to-English translation

- 🌿 "Grounding Mode" for panic & overwhelm (with sounds, affirmations)

- 🔒 No data storage — user privacy is respected by design

- 🎤 Voice input and output controls for hands-free chatting  
- 🧠 Session memory to remember past conversations  
- ⬇ Option to download chat transcripts


## 📂 Project Structure

```
MannMitra/
├── app.py               # Main Flask backend
├── templates/           # HTML templates (chat UI)
│   └── index.html
├── static/              # CSS, JS, media (Grounding Mode UI)
│   ├── css/
│   ├── js/
│   └── audio/
├── aasha_chatbot.py     # Emotion + translation logic 
├── grounding.py          # Grounding Mode component logic
├── faq.json              # Predefined FAQs
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not tracked)
├── .gitignore            # Files to ignore in version control
└── README.md
```



## ⚠️ Disclaimer
This tool is for support only and is **not a substitute for professional mental health services**.

---

 Want the full project proposal? [View our Synopsis](https://docs.google.com/document/d/1PxGemqIbzDmdyxpeJ1Ucoc_Pxwl3MXhxG3F3IVVCp5I/edit?usp=sharing)
