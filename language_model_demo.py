"""A small CPU demo: a language model assigns probabilities to token sequences.

Install: pip install torch transformers
The first run downloads DistilGPT2; no API key is required.
We calculate P(text tokens | BOS), without an end-of-text probability.
"""

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sentence_probability(sentence, tokenizer, model):
    """Display each next-token probability and return their product."""
    if not sentence.strip():
        raise ValueError("Please enter a non-empty sentence.")

    token_ids = tokenizer.encode(sentence, add_special_tokens=False)

    # Add a beginning-of-sequence token so the first word is scored too.
    input_ids = torch.tensor([[tokenizer.bos_token_id] + token_ids])
    if input_ids.shape[1] > model.config.n_positions:
        raise ValueError("This sentence is too long for the model.")

    with torch.no_grad():
        # Each position predicts the NEXT token; discard the last prediction.
        logits = model(input_ids).logits[0, :-1]
        log_probs = torch.log_softmax(logits.double(), dim=-1)
        token_log_probs = log_probs[torch.arange(len(token_ids)), token_ids]

    # Summing log probabilities is numerically more stable than multiplying.
    log_probability = token_log_probs.sum().item()
    probability = math.exp(log_probability)

    print(f"\nSentence: {sentence}")
    print(f"Number of tokens: {len(token_ids)}")
    print(f"{'Token':<20} P(token | previous tokens)")
    for token_id, log_p in zip(token_ids, token_log_probs.tolist()):
        token = repr(tokenizer.decode([token_id]))
        print(f"{token:<20} {math.exp(log_p):.6e}")

    print(f"Sequence probability: {probability:.6e}")
    print(f"Log probability (natural log): {log_probability:.4f}")
    return probability


if __name__ == "__main__":
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
    model.eval()  # Disable dropout for consistent scoring.

    # Edit these sentences to try your own examples.
    sentences = [
        "Colorless green ideas sleep furiously.",
        "How are you doing today?",
    ]

    for sentence in sentences:
        sentence_probability(sentence, tokenizer, model)
