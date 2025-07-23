from datasets import load_dataset

from tokenizer import *

dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func)
dataset = dataset.map(tokenize)