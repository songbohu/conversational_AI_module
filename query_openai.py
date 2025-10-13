from openai import OpenAI
import os
import sys

def load_openai_key(key_path: str = "openai.key"):
    if not os.path.exists(key_path):
        sys.exit(f"Error: The API key file '{key_path}' was not found.")

    with open(key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    os.environ["OPENAI_API_KEY"] = api_key
    print("OpenAI API key loaded successfully.")

def test_openai_api():
    client = OpenAI()
    response = client.responses.create(
        model="gpt-5-nano-2025-08-07",
        input="Write a one-sentence fun fact about Cambridge."
    )

    print("\n--- Model Response ---")
    print(response.output_text)
    print("----------------------\n")

if __name__ == "__main__":
    load_openai_key()
    test_openai_api()
