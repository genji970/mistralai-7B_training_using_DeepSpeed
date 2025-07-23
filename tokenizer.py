from transformers import AutoTokenizer

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
EOS_TOKEN = tokenizer.eos_token  # 미리 선언한 tokenizer 객체에서 가져옴

def formatting_prompts_func(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text.strip():
        text = alpaca_prompt.format(instruction, input_text, output)
    else:
        # input이 없는 경우 Input 섹션 제거
        text = alpaca_prompt.replace("### Input:\n{}\n\n", "").format(instruction, output)

    return { "text": text + EOS_TOKEN }

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)


