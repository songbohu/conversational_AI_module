import os
import sys
from openai import OpenAI

def load_openai_key(key_path: str = "openai.key"):
    if not os.path.exists(key_path):
        sys.exit(f"Error: The API key file '{key_path}' was not found.")

    with open(key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    os.environ["OPENAI_API_KEY"] = api_key
    print("OpenAI API key loaded successfully.")

def test_openai_api(model: str = "gpt-6-luna"):
    client = OpenAI()

    response = client.responses.create(
        model=model,
        input="Write a one-sentence fun fact about Cambridge.",
        temperature=0.3,
        max_output_tokens=512,
        store=False,
    )

    reply = response.output_text.strip()
    print(f"\n--- Model Response ({model}) ---")
    print(reply)
    print("----------------------\n")

if __name__ == "__main__":
    load_openai_key()
    test_openai_api()