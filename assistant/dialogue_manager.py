import random
import time
import sqlite3
import os
import config

class ProfessionalDialogueManager:
    """
    Manages context-aware, empathetic, and professional conversational dialogues.
    Designed for deployment in healthcare, affective computing, and corporate virtual assistance.
    """
    def __init__(self, db_path=os.path.join(config.BASE_DIR, "assistant_memory.db")):
        self.db_path = db_path
        self._init_database()
        
        # Professional responses indexed by detected emotional state
        self.emotion_responses = {
            "sad": [
                "I sense that you may be experiencing some distress or fatigue. Please take all the time you need; I am here to assist and support you.",
                "It sounds like things are challenging right now. How can I best assist you with your current workload or schedule?",
                "I am here to support you. Would you like to review your high-priority items, or take a short breather before continuing?"
            ],
            "fearful": [
                "I understand this situation may feel overwhelming. Let us break things down step-by-step to address each concern methodically.",
                "Your well-being is the priority. Please let me know what immediate information or support would be most helpful right now.",
                "We can approach this systematically. Would you like me to organize the key points or provide relevant reference documentation?"
            ],
            "angry": [
                "I acknowledge your frustration and am committed to resolving this issue thoroughly and efficiently.",
                "I understand this has been inconvenient. Let us directly examine what went wrong and identify the corrective action.",
                "Thank you for your patience. Please allow me to focus directly on finding an optimal solution for you."
            ],
            "happy": [
                "It is great to hear positive momentum! How can we build on this progress today?",
                "Excellent to hear! What objectives or tasks would you like to focus on next?",
                "Wonderful. I am ready to collaborate with you on your next milestone."
            ],
            "surprised": [
                "That appears unexpected. Would you like me to pull up recent records or analytics to verify the details?",
                "An unexpected update can alter priorities. Shall we adjust our current schedule accordingly?",
                "Let us review the latest inputs together to confirm the context and determine the next steps."
            ],
            "neutral": [
                "Hello. I am ready to assist you. How may I be of service today?",
                "Good day. Please let me know which tasks, queries, or analyses you would like to proceed with.",
                "Standing by and ready to assist with your inquiries or daily agenda."
            ]
        }
        
        # Proactive well-being & productivity prompts
        self.proactive_topics = [
            {"category": "wellness_check", "prompt": "Just checking in: remember to take a brief break to hydrate and stretch during prolonged working sessions.", "weight": 0.4},
            {"category": "agenda_review", "prompt": "Would you like a summary of pending action items and upcoming scheduled deadlines?", "weight": 0.3},
            {"category": "task_support", "prompt": "If you require analytical summaries, documentation review, or scheduling coordination, feel free to command.", "weight": 0.3}
        ]

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_transcript TEXT,
                detected_emotion TEXT,
                sentiment_score REAL,
                assistant_response TEXT
            )
        """)
        conn.commit()
        conn.close()

    def generate_response(self, user_transcript, detected_emotion, sentiment_polarity):
        """
        Generates an empathetic and professionally calibrated response based on multimodal emotion and sentiment.
        """
        emotion = detected_emotion.lower() if detected_emotion.lower() in self.emotion_responses else "neutral"
        
        # Select contextual base response
        base_responses = self.emotion_responses.get(emotion, self.emotion_responses["neutral"])
        selected_reply = random.choice(base_responses)
        
        # Append topic-specific feedback if query asks about task, time, or wellness
        lower_input = user_transcript.lower()
        if "time" in lower_input or "schedule" in lower_input:
            current_time = time.strftime("%I:%M %p")
            selected_reply += f" The current system time is {current_time}."
        elif "status" in lower_input or "system" in lower_input:
            selected_reply += " All speech emotion recognition and affective inference modules are operating normally."
            
        # Log to database
        self._log_interaction(user_transcript, emotion, sentiment_polarity, selected_reply)
        return selected_reply

    def _log_interaction(self, transcript, emotion, sentiment, response):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversation_logs (timestamp, user_transcript, detected_emotion, sentiment_score, assistant_response) VALUES (?, ?, ?, ?, ?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), transcript, emotion, sentiment, response)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database logging error: {e}")
