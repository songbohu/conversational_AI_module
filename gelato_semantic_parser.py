from openai import OpenAI
import os, json
from gelato_api import get_gelato


def parse_gelato_order(conversation, model="gpt-5-nano-2025-08-07", key_path="openai.key"):
    """
    Use OpenAI LLM to parse a conversation into a structured gelato order.
    Any missing fields are returned as empty strings.
    """

    # --- Load API key ---
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"API key file not found at {key_path}")
    with open(key_path, "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

    client = OpenAI()

    system_prompt = (
        "You are a semantic parser for an ice cream shop called Jack's Gelato. "
        "Given a conversation between a user and assistant, extract the user's final order "
        "as a JSON object with the following fields: "
        "flavours (list of strings), size (string), and container (string). "
        "If any information is missing, use an empty string (''). "
        "Return only valid JSON and nothing else."
    )

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation}]

    response = client.responses.create(model=model, input=messages)
    parsed_json = response.output_text.strip()

    try:
        order = json.loads(parsed_json)
    except json.JSONDecodeError:
        print("Model output not valid JSON, returning empty fields.")
        order = {"flavours": [], "size": "", "container": ""}

    return order

if __name__ == "__main__":
    conversation = """
    User: Hi! What flavours do you have today?
    Assistant: We have House Yoghurt, Coconut and Ube, and Dark Chocolate & Sea Salt.
    User: Great, can I get a double scoop of yoghurt and ube in a cone please?
    Assistant: Sure! One double scoop of House Yoghurt and Ube in a cone.
    """
    parsed = parse_gelato_order(conversation)
    print(json.dumps(parsed, indent=4))
    image_path = get_gelato(parsed)
    print(f"Image saved to {image_path}")