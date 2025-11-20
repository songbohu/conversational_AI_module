from abc import ABC, abstractmethod
from datetime import datetime
import json
import os

class DialogueSystem(ABC):
    """
    Abstract base class for dialogue systems.
    Handles essential logics for dialogue systems.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset dialogue state and history."""
        self.conversation_history = []

    def append_turn(self, speaker: str, utterance: str, meta: dict = None):
        """Add a turn to the dialogue history, with timestamp and optional metadata."""
        timestamp = datetime.now().strftime("%H:%M:%S")  # e.g. "10:20:02"
        turn = {
            "timestamp": timestamp,
            "speaker": speaker,
            "utterance": utterance
        }
        if meta:
            turn["meta"] = meta
        self.conversation_history.append(turn)

    @abstractmethod
    def chat(self, utterance: str) -> dict:
        """
        Process a user utterance and return a structured result.
        Must include at least a 'text' field for the assistant reply.
        Example return:
        {
            "text": "Here is the weather forecast for tomorrow.",
            "action": {"type": "query_weather", "location": "Cambridge"}
        }
        """
        pass

    def start_a_chat(self):
        print("The system is ready. Type `bye` or `exit` to end the conversation.\n")
        while True:
            user_input = input("User: ")

            if user_input.lower() in ["exit", "bye"]:
                print("Bot: Goodbye!")
                self.save_history()
                break

            result = self.chat(user_input)

            # Display assistant text
            print(f"\nBot: {result['text']}\n")

            self.append_turn("user", user_input)

            # Append structured result
            self.append_turn("assistant", result["text"], meta={k: v for k, v in result.items() if k != "text"})

    def get_history(self):
        return self.conversation_history


    def save_history(self, filepath=None):
        """
        Save the current conversation history to a JSON file.
        If no path is given, automatically generate one with timestamp.
        """
        if filepath is None:
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join("logs", f"conversation_{timestamp}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, indent=4, ensure_ascii=False)

        print(f"Conversation saved to {filepath}")
        return filepath

