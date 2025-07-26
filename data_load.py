from datasets import Dataset
from typing import Dict

def format_prompt(example: Dict[str, str]) -> Dict[str, str]:
    if all(k in example for k in ("instruction", "output")):
        instruction = example["instruction"]
        input_text = example.get("input", "")
        prompt = f"### Instruction:\n{instruction}\n\n"
        if input_text.strip():
            prompt += f"### Input:\n{input_text}\n\n"
        prompt += f"### Response:\n"
        return {"prompt": prompt, "completion": example["output"]}

    elif all(k in example for k in ("prompt", "completion")):
        return {"prompt": example["prompt"], "completion": example["completion"]}

    else:
        raise ValueError(f"지원하지 않는 데이터 형식: {example}")

def preprocess(dataset):
    # 데이터셋 열 확인
    column_names = dataset.column_names
    if all(k in column_names for k in ("prompt", "completion")):
        return dataset  # 그대로 사용
    elif all(k in column_names for k in ("instruction", "output")):
        return dataset.map(format_prompt, remove_columns=column_names)
    else:
        raise ValueError(f"지원하지 않는 열 구성: {column_names}")


"""
# 출력 확인
print(processed_dataset[0]) # input_ids , attention_mask , labels
print("111")
print(tokenized_dataset[0])
"""