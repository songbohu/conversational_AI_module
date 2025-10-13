from openai import OpenAI
from gelato_api import get_gelato
from gelato_semantic_parser import parse_gelato_order
import os, json, sys


class GelatoBot:
    """
    A conversational agent that understands user utterances about gelato,
    parses them into structured orders, and uses an LLM to generate natural responses.
    When the order is complete, it calls the Gelato API to generate the ice cream image.
    """

    def __init__(self, model="gpt-5-nano-2025-08-07", key_path="openai.key"):
        self._load_openai_key(key_path)
        self.client = OpenAI()
        self.model = model
        self.conversation_history = []

    def _load_openai_key(self, key_path: str):
        if not os.path.exists(key_path):
            sys.exit(f"Error: The API key file '{key_path}' was not found.")
        with open(key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
        print("OpenAI API key loaded successfully.\n")

    def chat(self, user_input: str):
        self.conversation_history.append({"role": "user", "content": user_input})

        # Combine full conversation into one text block
        conversation = "\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in self.conversation_history]
        )

        # --- Step 1: Semantic parsing ---
        order = parse_gelato_order(conversation)
        print("\nParsed Order:\n", json.dumps(order, indent=4))

        # --- Step 2: Check if order complete ---
        order_complete = bool(order["flavours"] and order["size"] and order["container"])

        # --- Step 3: Use LLM to generate response ---
        system_prompt = (
            "You are GelatoBot, a friendly and polite assistant working at Jack's Gelato. "
            "You help customers place ice cream orders and make small talk if needed. "
            "When the order is complete, confirm the details cheerfully. "
            "If something is missing (flavours, size, or container), ask naturally for clarification."
        )

        user_prompt = (
            f"Here is the current parsed order:\n{json.dumps(order, indent=4)}\n\n"
            f"Conversation so far:\n{conversation}\n\n"
            f"Please write the next assistant message."
        )

        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt}]
        )
        reply_text = response.output_text.strip()

        print("\nAssistant:", reply_text)

        # --- Step 4: Generate gelato image if ready ---
        if order_complete:
            image_path = get_gelato(order)
            print(f"Gelato image saved to: {image_path}\n")

        self.conversation_history.append({"role": "assistant", "content": reply_text})

    def start(self):
        print("Welcome to GelatoBot! Type 'bye' or 'exit' to exit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() in {"bye", "exit"}:
                print("Goodbye!")
                break
            self.chat(user_input)


if __name__ == "__main__":
    bot = GelatoBot()
    bot.start()
