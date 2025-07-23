from transformers import AutoTokenizer

def tokenize(example):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    return tokenizer(example["instruction"] + "\n" + example["output"], truncation=True, padding="max_length", max_length=512)
