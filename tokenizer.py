from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
EOS_TOKEN = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = '[PAD]'

def tokenize(example):
    # prompt는 항상 문자열임
    model_inputs = tokenizer(
        example["prompt"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # completion도 문자열일 경우만 처리
    completion = example["completion"]
    if isinstance(completion, list):  # 이미 토큰화된 경우
        labels = completion
    else:  # 문자열이면 토큰화
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                completion + EOS_TOKEN,
                truncation=True,
                padding="max_length",
                max_length=512
            )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs
