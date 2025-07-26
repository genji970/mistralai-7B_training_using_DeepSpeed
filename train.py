from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq
from datasets import load_dataset

from model_load import model
from tokenizer import tokenizer, tokenize   
from data_load import preprocess
from utils.utils import inspect_tokenized_dataset , print_label_lengths , print_field_lengths
from loss.trainer import MyTrainer

import torch

# ✅ 패딩 토큰 확인
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f"[디버깅] pad_token 추가됨 → {tokenizer.pad_token}")

# ✅ 모델 토크나이저 크기 조정
model.resize_token_embeddings(len(tokenizer))
print(f"[디버깅] 모델 임베딩 크기 재조정 완료 → {len(tokenizer)}")

# ✅ 데이터 로드
dataset_path = "yahma/alpaca-cleaned"  # 또는 "./my_dataset.json"
if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
    raw_data = load_dataset("json", data_files=dataset_path, split="train")
else:
    raw_data = load_dataset(dataset_path, split="train")

# ✅ 데이터 전처리
processed_dataset = preprocess(raw_data)
print_field_lengths(processed_dataset, stage="전처리 후")

# ✅ 토크나이징
tokenized_dataset = processed_dataset.map(tokenize, batched=True, remove_columns=processed_dataset.column_names)
print_field_lengths(tokenized_dataset, stage="토크나이징 후")

# ✅ 토치 텐서 형식으로 변환
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print(f"[디버깅] 텐서 포맷으로 설정 완료")
print_field_lengths(tokenized_dataset, stage="텐서 포맷 후")

# ✅ 각 필드 타입 확인
assert isinstance(tokenized_dataset[0]["input_ids"], torch.Tensor), "input_ids가 Tensor가 아닙니다"
assert isinstance(tokenized_dataset[0]["labels"], torch.Tensor), "labels가 Tensor가 아닙니다"

# ✅ inspect_tokenized_dataset 실행
# inspect_tokenized_dataset(tokenized_dataset)

# ✅ TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    report_to="none",
    deepspeed="ds_config.json",
    save_total_limit=1,
    save_strategy="epoch",
    fp16=True,
)
print("[디버깅] TrainingArguments 설정 완료")

# ✅ Trainer 설정
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    # data_collator=data_collator,  # 필요시 활성화
)
print("[디버깅] Trainer 인스턴스 생성 완료")

# ✅ 학습 시작
print("[디버깅] 학습 시작")
trainer.train()
