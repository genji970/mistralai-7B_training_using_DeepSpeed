IGNORE_INDEX = -100

def tokenize(examples):
    EOS_TOKEN = tokenizer.eos_token or ""

    # 리스트 보장
    prompts = examples["prompt"] if isinstance(examples["prompt"], list) else [examples["prompt"]]
    completions = examples["completion"] if isinstance(examples["completion"], list) else [examples["completion"]]
    completions = [c + EOS_TOKEN for c in completions]

    # 각각 토크나이즈
    model_inputs = tokenizer(prompts, truncation=True, padding=False)
    label_outputs = tokenizer(completions, truncation=True, padding=False)

    input_ids = model_inputs["input_ids"]
    labels = label_outputs["input_ids"]

    # 🔧 label 비어 있거나 2차원 아닌 경우 처리
    for i in range(len(labels)):
        if len(labels[i]) == 0:
            labels[i] = [IGNORE_INDEX]

    # 🔧 정렬과 패딩
    padded = tokenizer.pad(
        {
            "input_ids": input_ids,
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
        },
        padding=True,
        return_tensors=None
    )

    return padded
