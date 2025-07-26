from transformers import AutoTokenizer, Trainer, TrainingArguments , DataCollatorWithPadding , DataCollatorForSeq2Seq
from datasets import load_dataset

from model_load import model, tokenizer
from tokenizer import tokenize  
from data_load import preprocess
from utils.utils import inspect_tokenized_dataset
from loss.trainer import MyTrainer


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 🔧 모델 토크나이저 크기 조정
model.resize_token_embeddings(len(tokenizer))

# 🔧 데이터 로드
dataset_path = "yahma/alpaca-cleaned"  # 또는 "./my_dataset.json"
if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
    raw_data = load_dataset("json", data_files=dataset_path, split="train")
else:
    raw_data = load_dataset(dataset_path, split="train")

processed_dataset = preprocess(raw_data)
tokenized_dataset = processed_dataset.map(tokenize, batched=True)

#inspect_tokenized_dataset(tokenized_dataset)

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

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
    #data_collator=data_collator,
)
trainer.train()
