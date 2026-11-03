from openai import OpenAI
from openai.types.responses import ResponseInputParam
from gelato_api import get_gelato
from gelato_semantic_parser import parse_gelato_order
import os
import json
import sys


class GelatoBot:
    """
    A dialogue system that parses gelato orders and generates responses.
    When the order is complete, it calls the Gelato API
    to generate an ice cream image.
    """

    def __init__(self, model="gpt-6-luna", key_path="openai.key"):
        self._load_openai_key(key_path)
        self.client = OpenAI()
        self.model = model
        self.key_path = key_path
        self.conversation_history = []

    def _load_openai_key(self, key_path: str):
        if not os.path.exists(key_path):
            sys.exit(f"Error: The API key file '{key_path}' was not found.")

        with open(key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()

        os.environ["OPENAI_API_KEY"] = api_key
        print("OpenAI API key loaded successfully.\n")

    def chat(self, user_input: str):
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Combine full conversation into one text block
        conversation = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in self.conversation_history
        )

        # Step 1: Semantic parsing
        order = parse_gelato_order(
            conversation,
            model=self.model,
            key_path=self.key_path
        )

        print("\nParsed Order:\n", json.dumps(order, indent=4))

        # Step 2: Check whether the order is complete
        order_complete = bool(
            order["flavours"] and order["size"] and order["container"]
        )

        # Step 3: Use the LLM to generate a response
        system_prompt = (
            "You are GelatoBot, a friendly and polite assistant "
            "working at Jack's Gelato. "
            "You help customers place ice cream orders "
            "and make small talk if needed. "
            "When the order is complete, confirm the details cheerfully. "
            "If something is missing (flavours, size, or container), "
            "ask naturally for clarification."
        )

        user_prompt = (
            f"Here is the current parsed order:\n"
            f"{json.dumps(order, indent=4)}\n\n"
            f"Conversation so far:\n{conversation}\n\n"
            "Please write the next assistant message."
        )

        messages: ResponseInputParam = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.responses.create(
            model=self.model,
            input=messages,
            store=False
        )

        reply_text = response.output_text.strip()
        print("\nAssistant:", reply_text)

        # Step 4: Generate the gelato image if the order is complete
        if order_complete:
            image_path = get_gelato(order)
            print(f"Gelato image saved to: {image_path}\n")

        self.conversation_history.append({
            "role": "assistant",
            "content": reply_text
        })

    def start(self):
        print("Welcome to GelatoBot! Type 'bye' or 'exit' to exit.\n")

        while True:
            user_input = input("You: ")

            if user_input.strip().lower() in {"bye", "exit"}:
                print("Goodbye!")
                break

            self.chat(user_input)


if __name__ == "__main__":
    bot = GelatoBot()
    bot.start()