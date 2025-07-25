from datasets import load_dataset, Dataset
from typing import Dict
import json

from tokenizer import *

# ✅ 템플릿 정의
def format_prompt(example: Dict[str, str]) -> Dict[str, str]:
    # instruction-following 형식
    if all(k in example for k in ("instruction", "output")):
        instruction = example["instruction"]
        input_text = example.get("input", "")
        prompt = f"### Instruction:\n{instruction}\n\n"
        if input_text:
            prompt += f"### Input:\n{input_text}\n\n"
        prompt += f"### Response:\n"
        return {"prompt": prompt, "completion": example["output"]}

    # question-answering 형식
    elif all(k in example for k in ("question", "answer")):
        return {
            "prompt": f"### Question:\n{example['question']}\n\n### Answer:\n",
            "completion": example["answer"]
        }

    # text-label 분류 형식
    elif all(k in example for k in ("text", "label")):
        return {
            "prompt": f"### Classify the following text:\n{example['text']}\n\n### Label:\n",
            "completion": str(example["label"])
        }

    # 단일 text -> 출력 없음 (zero-shot generation 등)
    elif "text" in example:
        return {"prompt": example["text"], "completion": ""}

    else:
        raise ValueError(f"지원하지 않는 데이터 형식: {example}")

# ✅ 전처리 함수
def preprocess(dataset):
    return dataset.map(format_prompt, remove_columns=dataset.column_names)

"""
# 출력 확인
print(processed_dataset[0]) # input_ids , attention_mask , labels
print("111")
print(tokenized_dataset[0])
"""