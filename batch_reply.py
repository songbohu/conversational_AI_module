import json
from datetime import datetime
import os
from parrot_bot import ParrotBot


def batch_replay(input_file: str, output_file: str = None):
    """
    Run a batch of multi-turn dialogues through the dialogue system.
    Keeps both system-generated responses and ground-truth assistant utterances.
    """

    # Load test dialogues
    with open(input_file, "r", encoding="utf-8") as f:
        test_dialogues = json.load(f)

    print(f"Running batch replay with {len(test_dialogues)} dialogues...\n")

    bot = ParrotBot()
    all_results = []

    for d_idx, dialogue in enumerate(test_dialogues, start=1):
        print(f"--- Start of Dialogue {d_idx} ---\n")
        bot.reset()
        dialogue_result = []

        for turn_idx, turn in enumerate(dialogue, start=1):
            speaker = turn["speaker"]
            utterance = turn["utterance"]

            if speaker.lower() == "user":
                # Record user utterance
                bot.append_turn("user", utterance)
                print(f"[{turn_idx}] User: {utterance}")

                # System response
                result = bot.chat(utterance)
                generated_reply = result["text"]

                # Find ground-truth reply (if exists next)
                gt_reply = None
                if turn_idx < len(dialogue) and dialogue[turn_idx]["speaker"] == "assistant":
                    gt_reply = dialogue[turn_idx]["utterance"]

                # Append both generated and ground-truth responses
                dialogue_result.append({
                    "user_utterance": utterance,
                    "ground_truth": gt_reply,
                    "system_response": generated_reply,
                    "meta": {k: v for k, v in result.items() if k != "text"},
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })

                bot.append_turn("assistant", generated_reply)
                print(f"     Bot: {generated_reply}\n")

        all_results.append({
            "dialogue_id": d_idx,
            "turns": dialogue_result
        })
        print(f"--- End of Dialogue {d_idx} ---\n")

    # Save to file
    os.makedirs("logs", exist_ok=True)
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join("logs", f"batch_output_{timestamp}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"Batch replay completed. Results saved to {output_file}\n")
    return output_file


if __name__ == "__main__":
    batch_replay("example_input.json")
