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
    elif all(k in column_names for k in ("question", "expected_answer")):
        def map_to_prompts(example):
            prompts = []
        
            # 기본 preprocessed_output 먼저
            if "preprocessed_output" in example and example["preprocessed_output"] is not None:
                prompt = (
                f"### Question:\n{example['question']}\n\n"
                f"### Preprocessed Output:\n{example['preprocessed_output']}\n\n"
                f"### Response:\n"
                )
                prompts.append({"prompt": prompt, "completion": example["expected_answer"]})
        
            # preprocessed_output_1 ~ preprocessed_output_5
            for i in range(1, 6):
                key = f"preprocessed_output_{i}"
                if key in example and example[key] is not None:
                    prompt = (
                    f"### Question:\n{example['question']}\n\n"
                    f"### Preprocessed Output:\n{example[key]}\n\n"
                    f"### Response:\n"
                    )
                    prompts.append({"prompt": prompt, "completion": example["expected_answer"]})
            return prompts
        return dataset.flat_map(map_to_prompts, remove_columns=column_names)
    else:
        raise ValueError(f"지원하지 않는 열 구성: {column_names}")


"""
# 출력 확인
print(processed_dataset[0]) # input_ids , attention_mask , labels
print("111")
print(tokenized_dataset[0])
"""