from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

from model_load import model , tokenizer

from data_load import dataset
train_data = dataset["train"]

from tokenizer import tokenize


tokenized_dataset = train_data.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    report_to="none",
    deepspeed="ds_config.json",  # Deepspeed 설정
    save_total_limit=1,
    save_strategy="epoch",
    fp16=True,
)
"""
    tensorboard 사용시
    logging_dir="./logs",           # 로그 디렉토리
    logging_strategy="steps",       # logging step 단위
    logging_steps=10,               # 몇 step마다 log할지
    report_to="tensorboard",        # 보고 대상
"""

# Trainer 생성 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()

"""
terminal 실행 용
deepspeed train.py \
  --deepspeed ds_config.json \
  --output_dir ./output
"""

