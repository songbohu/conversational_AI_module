from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    input_ids = tokenizer.encode("Hi there! How are you?", return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    print(tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True))
