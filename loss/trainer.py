from transformers import Trainer
import torch
import torch.nn as nn

"""

cross entropy loss는 [B,C] , [B] shape의 logits와 Labels를 받는다.

"""

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # ← **kwargs 추가
        # 1. inputs에서 라벨 추출
        labels = inputs.pop("labels")

        # 2. 모델 forward
        outputs = model(**inputs)
        logits = outputs.logits  # 예: language modeling일 경우

        # 3. Loss 함수 정의 및 계산
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # 일반적으로 -100은 패딩 토큰 무시
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),  # [batch*seq_len, vocab]
            labels.view(-1)                    # [batch*seq_len]
        )

        # 4. loss 반환
        return (loss, outputs) if return_outputs else loss
