from openai import OpenAI
from openai.types.responses import ResponseInputParam
import os
import json
from gelato_api import get_gelato


def parse_gelato_order(conversation,
                       model="gpt-6-luna",
                       key_path="openai.key"):
    """
    Parse a conversation into a structured gelato order.
    Missing flavours are returned as an empty list.
    Missing size and container are returned as empty strings.
    """

    # Load API key
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"API key file not found at {key_path}")

    with open(key_path, "r", encoding="utf-8") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

    client = OpenAI()

    # Define the semantic parsing task
    system_prompt = (
        "You are a semantic parser for an ice cream shop called Jack's Gelato. "
        "Given a conversation between a user and assistant, extract the user's "
        "final order as a JSON object with the following fields: "
        "flavours (list of strings), size (string), and container (string). "
        "If flavours are missing, use an empty list ([]). "
        "If size or container is missing, use an empty string (\"\"). "
        "Return only valid JSON and nothing else."
    )

    messages: ResponseInputParam = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation}
    ]

    # Generate the structured order
    response = client.responses.create(
        model=model,
        input=messages,
        store=False
    )

    parsed_json = response.output_text.strip()

    # Convert JSON text into a Python dictionary
    try:
        order = json.loads(parsed_json)

        # Check the expected fields and their types
        if not (
            isinstance(order, dict)
            and isinstance(order.get("flavours"), list)
            and all(isinstance(f, str) for f in order["flavours"])
            and isinstance(order.get("size"), str)
            and isinstance(order.get("container"), str)
        ):
            raise ValueError("Unexpected order format.")

    except ValueError:
        print("Could not parse the order, returning empty fields.")
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

    if parsed["flavours"] and parsed["size"] and parsed["container"]:
        image_path = get_gelato(parsed)
        print(f"Image saved to {image_path}")
    else:
        print("The order is incomplete. Ask the user for more information.")