import os
import json
import time
from datasets import load_dataset

from choose_relevant import extract_relevant_sentences
from eliminating_symbols import clean_git_answer
from filtering_danger import is_safe_content
from data_score_func import compute_reward
from topic_ratio_criteria import should_sample_regex, TOPIC_REGEX_PATTERNS

# 저장 경로 설정
save_dir = "file_path"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, "squad_qa_pairs_with_reward.ndjson")

# 데이터셋 로드
ds = load_dataset("rajpurkar/squad", split="train")

with open(output_path, "w", encoding="utf-8") as fout:
    count = 0
    for item in ds:
        try:
            context = item["context"].strip()
            question = item["question"].strip()
            answer_list = item["answers"]["text"]

            for i, answer_text in enumerate(answer_list):
                output_raw = answer_text.strip()

                # 주제 필터링
                if not should_sample_regex(question + " " + context, TOPIC_REGEX_PATTERNS):
                    print(f"✗ 주제 제외됨: {question[:60]}")
                    continue

                # 민감 콘텐츠 필터링
                if not is_safe_content(question) or not is_safe_content(context) or not is_safe_content(output_raw):
                    print(f"✗ 민감 제외됨: {question[:60]}")
                    continue

                # 전처리
                output_filtered = extract_relevant_sentences(output_raw, question)
                output_cleaned = clean_git_answer(output_filtered)
                reward = compute_reward(question, output_cleaned)

                qa_pair = {
                    "instruction": context,
                    "input": question,
                    "output": output_cleaned,
                    "reward": reward
                }

                fout.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"✓ {question[:50]} [ans#{i+1}] | Reward: {reward}")
                count += 1
                time.sleep(0.05)

        except Exception as e:
            print(f"오류 발생: {e} (질문: {item.get('question', '')[:60]})")
            continue

    print(f"총 수집된 QA 수: {count}")

