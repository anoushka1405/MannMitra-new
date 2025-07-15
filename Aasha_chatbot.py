import os
import random
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline

# Load and configure Gemini API key securely
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model with memory
aasha_session = genai.GenerativeModel("models/gemini-2.5-flash").start_chat(history=[])

# GoEmotions → Aasha categories
GOEMOTION_TO_CORE = {
    "admiration": "love",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "neutral",
    "caring": "love",
    "confusion": "neutral",
    "curiosity": "neutral",
    "desire": "love",
    "disappointment": "anger",  # Changed from sadness
    "disapproval": "anger",     # May change based on testing
    "disgust": "anger",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "love",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "neutral",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral"
}


# Emotion-specific suggestions
emotion_responses = {
    "sadness": {"reflection": "That sounds incredibly heavy — I’m really sorry you're carrying this.",
                "ideas": ["Wrap up in a soft blanket and sip something warm", "Try writing what you’re feeling, even messily", "Listen to a soft, comforting song"]},
    "fear": {"reflection": "It’s completely okay to feel scared — you’re not alone in this.",
             "ideas": ["Try naming five things around you to ground yourself", "Take a few slow belly breaths", "Hold onto something soft and familiar"]},
    "anger": {"reflection": "That kind of anger can feel overwhelming — and it’s valid.",
              "ideas": ["Scribble or draw your emotions without judgment", "Write down what you wish you could say", "Move around — shake out your arms or take a brisk walk"]},
    "joy": {"reflection": "That’s so lovely to hear — I’m smiling with you.",
            "ideas": ["Close your eyes and really soak it in", "Capture it in a photo or note to remember", "Share it with someone who cares"]},
    "love": {"reflection": "That warm feeling is so special — thank you for sharing it.",
             "ideas": ["Text someone what they mean to you", "Write down how that love feels", "Breathe deeply and just hold onto the moment"]},
    "surprise": {"reflection": "That must’ve caught you off guard — surprises stir up so much.",
                 "ideas": ["Pause and take a slow breath", "Note your first thoughts about what happened", "Just sit quietly and let it settle"]},
    "neutral": {"reflection": "Whatever you're feeling, I'm right here with you.",
                "ideas": ["Take a short pause — maybe a breath or gentle stretch", "Write down anything on your mind", "Put on some soft background music"]}
}

# Load FAQs
with open("faq.json", "r") as f:
    faq_data = json.load(f)

def match_faq(user_input):
    clean = user_input.lower().strip()
    for entry in faq_data:
        for q in entry["questions"]:
            if q in clean:
                return entry["answer"]
    return None

def detect_celebration_type(message):
    msg = message.lower()

    celebration_keywords = {
        "hearts": [
            "anniversary", "wedding day", "years together", "relationship milestone", "got engaged",
            "proposal", "engagement", "married", "got married", "we tied the knot", "my partner", "soulmate",
            "love of my life", "our journey", "special someone", "my person", "celebrating love"
        ],
        "balloons": [
            "birthday", "bday", "birth anniversary", "turned", "i turned", "it's my birthday", "my cake day",
            "celebrating my birthday", "birthday girl", "birthday boy", "cutting cake", "party hat", "balloons",
            "another trip around the sun", "growing older", "level up", "🎂", "🎈", "🎉"
        ],
        "confetti": [
            "got a job", "got hired", "new job", "job offer", "promotion", "landed a job", "work anniversary",
            "career milestone", "accepted offer", "selected for", "internship offer", "placement", "campus placed",
            "new opportunity", "we're celebrating", "we celebrated", "success party", "housewarming", "party",
            "throwing a party", "🎊", "🥳", "🎉", "cheers to", "big news", "exciting news"
        ]
    }

    for category, keywords in celebration_keywords.items():
        for phrase in keywords:
            if phrase in msg:
                return category

    return None

def detect_emotion_keywords(text):
    text_lower = text.lower()
    keywords = {
        "sadness": ["sad", "grief", "loss", "hopeless", "down", "depressed", "cry", "alone", "tired", "numb", "hurt", "lonely", "broken", "regret", "disappointed", "devastated","pretending", "exhausted", "don't care", "over it", "numb"],
        "joy": ["happy", "excited", "yay", "glad", "smile", "fun", "cheerful", "bright", "laugh", "peaceful", "grateful", "thrilled", "delighted", "wonderful"],
        "anger": ["angry", "mad", "furious", "rage", "irritated", "annoyed", "hate", "frustrated", "pissed", "bitter", "livid", "outraged", "disgusted"],
        "fear": ["anxious", "worried", "scared", "afraid", "panic", "nervous", "terrified", "unsafe", "shaking", "tension", "frightened", "alarmed"],
        "love": ["love", "loved", "cared", "affection", "heartfelt", "close to me", "bond", "sweet", "caring", "adore", "cherish"],
        "surprise": ["shocked", "surprised", "unexpected", "can't believe", "wow", "unbelievable", "sudden", "mind blown", "astonished", "amazed"],
        "neutral": ["okay", "fine", "meh", "nothing", "normal", "usual", "bored", "whatever", "idk", "alright"]
    }
    emotion_scores = {}
    for emotion, emotion_keywords in keywords.items():
        score = sum(1 for keyword in emotion_keywords if re.search(rf"\b{re.escape(keyword)}\b", text_lower) or keyword in text_lower)
        emotion_scores[emotion] = score
    if any(score > 0 for score in emotion_scores.values()):
        return max(emotion_scores.items(), key=lambda x: x[1])[0]
    return None

def detect_emotion_keywords_improved(text):
    """
    Improved emotion detection that handles negation and context
    """
    text_lower = text.lower()
    
    # Negation patterns - words that flip meaning
    negation_patterns = [
        r'\b(not|no|never|dont|don\'t|doesnt|doesn\'t|didnt|didn\'t|wont|won\'t|cant|can\'t|isnt|isn\'t|arent|aren\'t|wasnt|wasn\'t|werent|weren\'t)\b',
        r'\b(without|lack|absence|missing|lose|lost|failed|fail)\b'
    ]
    
    # Enhanced keywords with context awareness
    keywords = {
        "sadness": {
            "direct": ["sad", "grief", "loss", "hopeless", "down", "depressed", "cry", "alone", "tired", "numb", "hurt", "lonely", "broken", "regret", "disappointed", "devastated", "miserable", "empty", "worthless"],
            "contextual": ["love", "care", "support", "help", "friend", "family"]  # These become sad when negated
        },
        "joy": {
            "direct": ["happy", "excited", "yay", "glad", "smile", "fun", "cheerful", "bright", "laugh", "peaceful", "grateful", "thrilled", "delighted", "wonderful", "amazing", "fantastic", "great", "awesome"],
            "contextual": []
        },
        "anger": {
            "direct": ["angry", "mad", "furious", "rage", "irritated", "annoyed", "hate", "frustrated", "pissed", "bitter", "livid", "outraged", "disgusted", "awful", "terrible"],
            "contextual": []
        },
        "fear": {
            "direct": ["anxious", "worried", "scared", "afraid", "panic", "nervous", "terrified", "unsafe", "shaking", "tension", "frightened", "alarmed", "stress", "overwhelmed"],
            "contextual": []
        },
        "love": {
            "direct": ["adore", "cherish", "treasure", "devoted", "affection", "heartfelt", "sweet", "caring", "warm", "tender"],
            "contextual": ["love", "loved", "cared", "close to me", "bond", "support", "help", "friend", "family"]  # Only count if NOT negated
        },
        "surprise": {
            "direct": ["shocked", "surprised", "unexpected", "can't believe", "wow", "unbelievable", "sudden", "mind blown", "astonished", "amazed", "wtf", "omg"],
            "contextual": []
        },
        "neutral": {
            "direct": ["okay", "fine", "meh", "nothing", "normal", "usual", "bored", "whatever", "idk", "alright", "regular"],
            "contextual": []
        }
    }
    
    def has_negation_before(text, keyword_pos):
        """Check if there's negation within 5 words before the keyword"""
        # Look for negation in 5 words before the keyword
        start_pos = max(0, keyword_pos - 50)  # Approximate 5 words
        text_before = text[start_pos:keyword_pos]
        
        for pattern in negation_patterns:
            if re.search(pattern, text_before):
                return True
        return False
    
    emotion_scores = {}
    
    for emotion, emotion_keywords in keywords.items():
        direct_score = 0
        contextual_score = 0
        
        # Count direct keywords (always count these)
        for keyword in emotion_keywords["direct"]:
            matches = list(re.finditer(rf"\b{re.escape(keyword)}\b", text_lower))
            direct_score += len(matches)
        
        # Handle contextual keywords (flip meaning if negated)
        for keyword in emotion_keywords["contextual"]:
            matches = list(re.finditer(rf"\b{re.escape(keyword)}\b", text_lower))
            
            for match in matches:
                keyword_pos = match.start()
                is_negated = has_negation_before(text_lower, keyword_pos)
                
                if emotion == "love" and not is_negated:
                    # Love keywords only count for love if NOT negated
                    contextual_score += 1
                elif emotion == "sadness" and is_negated:
                    # Love/care keywords count for sadness if negated
                    contextual_score += 2  # Weight negated positive words heavily for sadness
        
        emotion_scores[emotion] = direct_score + contextual_score
    
    # Special handling for common negated phrases
    negated_phrases = {
        "sadness": [
            "don't love me", "doesn't love me", "no one loves me", "nobody loves me",
            "don't care", "doesn't care", "no one cares", "nobody cares",
            "not happy", "not good", "not okay", "not fine", "not well",
            "can't be happy", "won't be happy", "not feeling good"
        ]
    }
    
    for emotion, phrases in negated_phrases.items():
        for phrase in phrases:
            if phrase in text_lower:
                emotion_scores[emotion] += 3  # High weight for explicit negated phrases
    
    # Find the emotion with highest score
    if any(score > 0 for score in emotion_scores.values()):
        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        return top_emotion
    
    return None

import re

def is_negated(keyword, text):
    """
    Check if the keyword is negated in the context (e.g., "not happy", "never sad").
    """
    pattern = rf"(not|no|never|isn['’]t|wasn['’]t|don['’]t|doesn['’]t|didn['’]t)\s+(\w+\s+)?{re.escape(keyword)}"
    return re.search(pattern, text.lower()) is not None

import re

# Helper: Check if a keyword is negated
def is_negated(keyword, text):
    pattern = rf"(not|no|never|isn['’]t|wasn['’]t|don['’]t|doesn['’]t|didn['’]t)\s+(\w+\s+)?{re.escape(keyword)}"
    return re.search(pattern, text.lower()) is not None

# GOEMOTION_TO_CORE should be defined in your code earlier
# emotion_classifier should also be available

def get_emotion_label(text, threshold=2):

    if not hasattr(get_emotion_label, "model"):
        get_emotion_label.model = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=5
        )
    emotion_classifier = get_emotion_label.model

    try:
        print(f"\n[Emotion Detection] Input: {text}")
        text_lower = text.lower()

        # Step 1: Keyword-based detection with negation handling
        keywords = {
            "sadness": ["sad", "grief", "loss", "hopeless", "down", "depressed", "cry", "alone", "tired", "numb", "hurt", "lonely", "broken", "regret", "disappointed", "devastated"],
            "joy": ["happy", "excited", "yay", "glad", "smile", "fun", "cheerful", "bright", "laugh", "peaceful", "grateful", "thrilled", "delighted", "wonderful"],
            "anger": ["angry", "mad", "furious", "rage", "irritated", "annoyed", "hate", "frustrated", "pissed", "bitter", "livid", "outraged", "disgusted"],
            "fear": ["anxious", "worried", "scared", "afraid", "panic", "nervous", "terrified", "unsafe", "shaking", "tension", "frightened", "alarmed"],
            "love": ["love", "loved", "cared", "affection", "heartfelt", "close to me", "bond", "sweet", "caring", "adore", "cherish"],
            "surprise": ["shocked", "surprised", "unexpected", "can't believe", "wow", "unbelievable", "sudden", "mind blown", "astonished", "amazed"],
            "neutral": ["okay", "fine", "meh", "nothing", "normal", "usual", "bored", "whatever", "idk", "alright"]
        }

        emotion_scores = {}
        for emotion, emotion_keywords in keywords.items():
            score = 0
            for keyword in emotion_keywords:
                if keyword in text_lower and not is_negated(keyword, text_lower):
                    if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                        print(f"[Keyword Match] {emotion}: '{keyword}' matched")
                        score += 1
            emotion_scores[emotion] = score

        top_emotion, top_score = max(emotion_scores.items(), key=lambda x: x[1])
        print(f"[Keyword Detection Result] Top Emotion: {top_emotion}, Score: {top_score}")

        if top_score >= threshold:
            print(f"[Keyword Override] Using keyword-based label: {top_emotion}")
            return top_emotion

        # Step 2: Model fallback
        print("[Model Fallback] No keyword match — using model.")
        results = emotion_classifier(text)
        print(f"[Model Raw Output] {results}")

        if isinstance(results, list) and len(results) > 0:
            results = results[0] if isinstance(results[0], list) else results

        core_scores = {}
        for r in results:
            label = r["label"].lower()
            score = r["score"]
            core = GOEMOTION_TO_CORE.get(label, "neutral")
            core_scores[core] = core_scores.get(core, 0) + score
            print(f"[Model Mapping] Label: {label}, Core: {core}, Score: {score:.3f}")

        top_emotion, top_score = max(core_scores.items(), key=lambda x: x[1])
        print(f"[Model Final] Top Emotion: {top_emotion}, Score: {top_score:.3f}")
        return top_emotion if top_score >= 0.3 else "neutral"

    except Exception as e:
        print(f"[Emotion Detection Error] {e}")
        return "neutral"


SHORT_REACT = {
    "joy": [
        "That’s such good energy! I’m smiling with you.",
        "That kind of joy deserves to be felt fully — I’m so glad you shared it with me.",
        "A moment like this? Worth holding onto. I’m right here with you."
    ],
    "love": [
        "That warmth really comes through.",
        "What a lovely moment to hold onto.",
        "That's such a tender feeling."
    ],
    "surprise": [
        "That must’ve caught you off guard!",
        "Wow, I wasn’t expecting that either.",
        "Life’s full of little twists, isn’t it?"
    ]
}

JOY_INVITES = [
    "Want to tell me what made your day feel this good?",
    "If you feel like sharing what’s lighting you up — I’d love to hear.",
    "What’s been bringing this kind of smile today?",
    "What’s been going *right* for you lately?"
]

INVITE_LINES = [
    "If you want to say more, I’m right here.",
    "Want to talk a little more about this?",
    "I’d love to hear more if you’re open to sharing.",
    "If there’s more on your mind, I’m listening."
]

# ---------------------- First Message Handler ----------------------

def first_message(user_input):
    faq_response = match_faq(user_input)
    if faq_response:
        return faq_response, {"emotion": "neutral", "celebration_type": None}
    emotion = get_emotion_label(user_input)
    celebration = detect_celebration_type(user_input)
    word_count = len(re.findall(r'\w+', user_input))
    if emotion in ["joy", "love", "surprise"] and word_count < 12:
        reaction = random.choice(SHORT_REACT[emotion])
        suggestion = random.choice(emotion_responses[emotion]["ideas"])
        invite = random.choice(JOY_INVITES if emotion == "joy" else INVITE_LINES)
        response = f"{reaction} {suggestion}. {invite}"
        return response, {"emotion": emotion, "celebration_type": celebration}
    prompt = f'''
You are Aasha, a deeply emotionally intelligent AI companion.
Speak with warmth, empathy, and clarity — like a close, thoughtful friend.

This is the user's first message:
"{user_input}"

Please:
- Start with a short emotional reflection (2 lines max)
- Offer 2 gentle, supportive ideas based on their emotion
- End with a soft invitation to share more, if they’d like
- Keep the tone human, warm, not robotic
- Never use endearments like "dear" or "sweetheart"
- If the message is vague or low-detail, be brief
'''
    try:
        response = aasha_session.send_message(prompt)
        return response.text.strip(), {"emotion": emotion, "celebration_type": celebration}
    except Exception as e:
        print("Gemini error:", e)
        return "I'm here with you, but something's a little off on my side. Want to try again?", {
            "emotion": "neutral", "celebration_type": None
        }

# ---------------------- Continue Conversation ----------------------

def continue_convo(user_input):
    faq = match_faq(user_input)
    if faq:
        return faq, {"emotion": "neutral", "celebration_type": None}
    emotion = get_emotion_label(user_input)
    celebration = detect_celebration_type(user_input)
    prompt = f'''
You are Aasha — an emotionally intelligent AI companion who remembers past conversations and emotions.
Your tone is warm, clear, and comforting — like a close friend who truly listens.

Here’s the user’s message:
"{user_input}"

Please:
- Respond in 3 to 4 short, natural sentences.
- Acknowledge what they’re feeling now (even if it's vague or mixed).
- Offer 1 or 2 soft, grounded suggestions or reflections.
- End with a warm invitation to keep talking.
- Avoid clinical language, repetitive advice, or endearments.
'''
    try:
        response = aasha_session.send_message(prompt)
        return response.text.strip(), {"emotion": emotion, "celebration_type": celebration}
    except Exception as e:
        print("Gemini error:", e)
        return "I'm here with you, but something's a little off on my side. Want to try again?", {
            "emotion": "neutral", "celebration_type": None
        }

# ---------------------- Exit Detection ----------------------

intent_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", top_k=1)

def is_exit_intent(text):
    try:
        lowered = text.lower()
        exit_phrases = [
            "bye", "goodbye", "see you", "talk to you later", "ttyl",
            "gotta go", "i have to go", "logging off", "i’m done", "signing off", "good night",
            "ok bye", "bye aasha", "i'm leaving now", "enough for now","done for today", "that's enough", "logging off"
        ]
        if any(p in lowered for p in exit_phrases):
            return True
        res = intent_classifier(text)
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        if isinstance(res, dict) and "label" in res:
            return "goodbye" in res["label"].lower() or "gratitude" in res["label"].lower()
    except Exception as e:
        print("Exit intent error:", e)
    return False

# ---------------------- CLI for Testing ----------------------

if __name__ == "__main__":
    print("Hi, I’m Aasha. What’s on your mind today?")
    user_input = input("You: ")
    response, meta = first_message(user_input)
    print("Aasha:", response)
    while True:
        user_input = input("You: ")
        if is_exit_intent(user_input):
            print("Aasha: I'm really glad we talked today. Please take care 💙")
            break
        response, meta = continue_convo(user_input)
        print("Aasha:", response)
