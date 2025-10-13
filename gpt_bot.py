from dialogue_system import DialogueSystem
from openai import OpenAI
import os
import sys


class GPTBot(DialogueSystem):
    """
    A dialogue agent powered by OpenAI's GPT-5 Nano model.
    Extends the DialogueSystem base class to generate responses
    using the OpenAI API.
    """

    def __init__(self, model: str = "gpt-5-nano-2025-08-07", key_path: str = "openai.key"):
    # def __init__(self, model: str = "gpt-4o", key_path: str = "openai.key"):

        super().__init__()
        self.model = model
        self._load_openai_key(key_path)
        self.client = OpenAI()

    def _load_openai_key(self, key_path: str):
        if not os.path.exists(key_path):
            sys.exit(f"Error: The API key file '{key_path}' was not found.")
        with open(key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
        print("OpenAI API key loaded successfully.\n")

    def chat(self, utterance: str) -> dict:
        """
        Send the dialogue history and the latest user utterance to the LLM,
        and return the generated response.
        """
        # Combine dialogue context
        # messages = []

        messages = [{"role": "system",
                     "content": "You are a friendly and knowledgeable Cambridge student who enjoys helping others learn about college life."}]

        for turn in self.conversation_history:
            role = "assistant" if turn["speaker"] == "assistant" else "user"
            messages.append({"role": role, "content": turn["utterance"]})

        messages.append({"role": "user", "content": utterance})

        # Call the GPT model
        response = self.client.responses.create(
            model=self.model,
            input=messages
            # temperature=0.3,
            # max_output_tokens=150,
            # top_p=0.9
        )

        reply_text = response.output_text.strip()

        return {
            "text": reply_text
        }


if __name__ == "__main__":
    bot = GPTBot()
    bot.start_a_chat()
