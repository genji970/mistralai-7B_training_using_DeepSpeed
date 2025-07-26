from transformers import AutoTokenizer
# Hugging Face datasets 라이브러리를 사용하기 위한 임포트
from datasets import Dataset
import torch # PyTorch 텐서 연산을 위해 임포트

# Mistral-7B-v0.3 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

# 패딩 토큰이 없는 경우 [PAD] 토큰 추가
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize(examples, MAX_LEN=520):
    """
    주어진 예시 데이터를 토크나이징하고, 모델 입력 및 레이블을 생성합니다.
    'prompt'/'completion' 또는 'instruction'/'input'/'output' 구조를 지원합니다.
    패딩을 적용하고 레이블의 패딩 토큰을 -100으로 마스킹합니다.
    레이블 마스킹은 PyTorch 텐서 연산을 사용하여 직접 수행합니다.

    Args:
        examples (dict): 토크나이징할 데이터 예시 딕셔너리.
                         'prompt'와 'completion' 키 또는
                         'instruction', 'input', 'output' 키를 포함해야 합니다.
                         (map 함수에서 batched=True로 사용 시 각 키의 값은 리스트여야 함)
        MAX_LEN (int): 최대 시퀀스 길이. 이 길이에 맞춰 패딩 및 잘라내기가 수행됩니다.

    Returns:
        dict: 'input_ids', 'attention_mask', 'labels'를 포함하는 딕셔너리.
              각 값은 PyTorch 텐서 형태입니다.
    """
    EOS_TOKEN = tokenizer.eos_token # End-of-sequence 토큰

    prompts = []
    completions = []

    # 데이터 구조에 따라 prompt와 completion을 준비
    # case 1: prompt/completion 구조
    if "prompt" in examples and "completion" in examples:
        # 입력이 단일 문자열일 수도 있고, map 함수에서 batched=True 시 리스트일 수도 있음
        _prompts = examples["prompt"] if isinstance(examples["prompt"], list) else [examples["prompt"]]
        _completions = examples["completion"] if isinstance(examples["completion"], list) else [examples["completion"]]
        
        # completion 끝에 EOS 토큰 추가
        completions = [c + EOS_TOKEN for c in _completions]
        prompts = _prompts

    # case 2: instruction/input/output 구조 (Alpaca 스타일)
    elif "instruction" in examples and "output" in examples:
        _instructions = examples["instruction"] if isinstance(examples["instruction"], list) else [examples["instruction"]]
        # 'input' 키가 없을 경우 빈 문자열 리스트로 처리
        _inputs = examples["input"] if "input" in examples else ["" for _ in _instructions]
        _outputs = examples["output"] if isinstance(examples["output"], list) else [examples["output"]]

        for inst, inp in zip(_instructions, _inputs):
            prompt = f"### Instruction:\n{inst}\n\n"
            if inp and inp.strip(): # input이 비어있지 않은 경우에만 추가
                prompt += f"### Input:\n{inp}\n\n"
            prompt += "### Response:\n" # 응답 시작 부분
            prompts.append(prompt)
        
        # output 끝에 EOS 토큰 추가
        completions = [o + EOS_TOKEN for o in _outputs]

    else:
        # 지원하지 않는 키 구조일 경우 에러 발생
        raise ValueError(f"지원하지 않는 키 구조: {examples.keys()}")

    # 1. 프롬프트 토크나이징 (모델 입력)
    # return_tensors='pt'를 사용하여 PyTorch 텐서로 반환
    model_inputs = tokenizer(
        prompts,
        truncation=True,        # MAX_LEN보다 길면 잘라냄
        max_length=MAX_LEN,     # 최대 길이 설정
        padding='max_length',   # 모든 시퀀스를 MAX_LEN으로 패딩
        return_tensors='pt'     # PyTorch 텐서로 반환
    )

    # 2. 컴플리션 토크나이징 (레이블 생성용)
    # return_tensors='pt'를 사용하여 PyTorch 텐서로 반환
    label_outputs = tokenizer(
        completions,
        truncation=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_tensors='pt'
    )

    # 🔥 labels에서 pad_token_id를 -100으로 마스킹
    # PyTorch 텐서 연산을 사용하여 직접 마스킹 수행
    labels = label_outputs["input_ids"].clone() # 원본 텐서를 복사하여 수정
    labels[labels == tokenizer.pad_token_id] = -100 # 패딩 토큰 ID를 -100으로 변경

    # 패딩 및 마스킹 후 input_ids와 labels의 길이가 동일한지 확인 (디버깅 목적)
    # map 함수에서 batched=True를 사용할 때 모든 시퀀스의 길이가 동일해야 합니다.
    # 이제 input_ids와 labels 모두 텐서이므로 .shape[1]으로 길이 확인
    for i in range(model_inputs["input_ids"].shape[0]): # 배치 크기만큼 반복
        assert model_inputs["input_ids"].shape[1] == MAX_LEN, \
            f"Input IDs length mismatch at index {i}: Expected {MAX_LEN}, Got {model_inputs['input_ids'].shape[1]}"
        assert labels.shape[1] == MAX_LEN, \
            f"Labels length mismatch at index {i}: Expected {MAX_LEN}, Got {labels.shape[1]}"
        assert model_inputs["input_ids"].shape[1] == labels.shape[1], \
            f"Input IDs and Labels length mismatch at index {i}: Input IDs {model_inputs['input_ids'].shape[1]}, Labels {labels.shape[1]}"

    # 최종 레이블을 model_inputs 딕셔너리에 추가
    model_inputs["labels"] = labels
    return model_inputs