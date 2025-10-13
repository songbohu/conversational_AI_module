from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json

def evaluate_bleu(output_file: str):
    with open(output_file, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    smoothie = SmoothingFunction().method4
    scores = []

    for d in dialogues:
        for turn in d["turns"]:
            gt = turn["ground_truth"]
            sys = turn["system_response"]
            if gt and sys:
                score = sentence_bleu([gt.split()], sys.split(), smoothing_function=smoothie)
                scores.append(score)

    avg_bleu = sum(scores) / len(scores) if scores else 0.0
    print(f"Average BLEU score: {avg_bleu:.4f}")

if __name__ == "__main__":
    evaluate_bleu("logs/batch_output_20251012_101233.json")