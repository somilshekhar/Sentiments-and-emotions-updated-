import os
import pyttsx3

class EmotionAdaptiveTTSEngine:
    """
    Emotion-Adaptive Text-to-Speech Engine.
    Adjusts pitch, speech rate, and acoustic intensity based on inferred emotional state
    as described in Section 5 of the paper.
    """
    def __init__(self):
        self.engine = pyttsx3.init()
        # Default baseline configuration
        self.base_rate = 150
        self.base_volume = 0.9
        self.engine.setProperty('rate', self.base_rate)
        self.engine.setProperty('volume', self.base_volume)

    def configure_voice_for_emotion(self, emotion):
        """
        Modulates prosodic speech parameters according to the interlocutor's state:
        - Sad/Anxious: Softer volume, slower cadence (supportive & calming).
        - Angry: Calm, steady pace, measured volume.
        - Happy/Surprised: Slightly accelerated cadence, dynamic inflection.
        - Neutral: Professional standard rate.
        """
        emotion = emotion.lower()
        if emotion in ["sad", "fearful"]:
            self.engine.setProperty('rate', int(self.base_rate * 0.88))
            self.engine.setProperty('volume', 0.85)
        elif emotion == "angry":
            self.engine.setProperty('rate', int(self.base_rate * 0.92))
            self.engine.setProperty('volume', 0.90)
        elif emotion in ["happy", "surprised"]:
            self.engine.setProperty('rate', int(self.base_rate * 1.05))
            self.engine.setProperty('volume', 0.95)
        else:
            self.engine.setProperty('rate', self.base_rate)
            self.engine.setProperty('volume', self.base_volume)

    def speak(self, text, emotion="neutral"):
        """
        Synthesizes speech with emotion-adapted prosody.
        """
        self.configure_voice_for_emotion(emotion)
        print(f"[Assistant ({emotion.capitalize()} Mode)]: {text}")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS playback notice: {e}")

    def save_to_file(self, text, output_path, emotion="neutral"):
        """
        Saves synthesized audio to a WAV file.
        """
        self.configure_voice_for_emotion(emotion)
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()
        return output_path
