from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")