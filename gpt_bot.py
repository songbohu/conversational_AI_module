import os

from openai import OpenAI
from typing import Literal
from openai.types.responses import ResponseInputParam
from dialogue_system import DialogueSystem


class GPTBot(DialogueSystem):
    """A dialogue system powered by OpenAI's GPT-6 Luna model."""

    def __init__(
        self,
        model: str = "gpt-6-luna",
        key_path: str = "openai.key",
    ):
        super().__init__()
        self.model = model
        self._load_openai_key(key_path)
        self.client = OpenAI()

    def _load_openai_key(self, key_path: str):

        if os.environ.get("OPENAI_API_KEY"):
            return

        with open(key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()

        if not api_key:
            raise ValueError(f"The API key file '{key_path}' is empty.")

        os.environ["OPENAI_API_KEY"] = api_key

    def chat(self, utterance: str) -> dict:
        """Generate a reply using the previous dialogue history."""

        messages: ResponseInputParam = [{
            "role": "system",
            "content": (
                "You are an unfriendly and knowledgeable Cambridge student who enjoys helping others learn about college life."
            ),
        }]

        for turn in self.conversation_history:
            role: Literal["user", "assistant"] = (
                "assistant" if turn["speaker"] == "assistant" else "user"
            )

            messages.append({
                "role": role,
                "content": turn["utterance"],
            })

        messages.append({
            "role": "user",
            "content": utterance,
        })

        response = self.client.responses.create(
            model=self.model,
            input=messages,
        )

        if response.status != "completed":
            raise RuntimeError(
                "The model response was not completed. Check the output-token limit or try again."
            )

        reply_text = response.output_text.strip()
        if not reply_text:
            raise RuntimeError("The model returned no text response.")

        return {"text": reply_text}


if __name__ == "__main__":
    bot = GPTBot()
    bot.start_a_chat()