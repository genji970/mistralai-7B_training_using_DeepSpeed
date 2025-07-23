from transformers import AutoTokenizer

def tokenize(example):
    return tokenizer(example["instruction"] + "\n" + example["output"], truncation=True, padding="max_length", max_length=512)
