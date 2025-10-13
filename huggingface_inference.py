from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if __name__ == '__main__':

    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    prompt = "Translate to French: Cambridge students love gelato."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=40)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
