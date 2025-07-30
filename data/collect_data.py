import os
import json
from datasets import load_dataset

from choose_relevant import extract_relevant_sentences
from eliminating_symbols import clean_git_answer
from filtering_danger import is_safe_content
from data_score_func import compute_reward
from topic_ratio_criteria import should_sample_regex, TOPIC_REGEX_PATTERNS

# 저장 경로 설정
save_dir = "file_path"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, "qa_pairs_with_reward.ndjson")

# 데이터셋 로드
ds = load_dataset("nreimers/reddit_question_best_answers", split="train")

with open(output_path, "w", encoding="utf-8") as fout:
    count = 0
    for item in ds:
        try:
            title = item["title"].strip()
            body = item["selftext"].strip() if item["selftext"] else ""
            output_raw = item["best_answer"].strip()

            # 주제 기반 샘플링
            if not should_sample_regex(title + " " + body, TOPIC_REGEX_PATTERNS):
                print(f"✗ 주제 제외됨: {title[:60]}")
                continue

            # 민감 콘텐츠 제외
            if not is_safe_content(title) or not is_safe_content(body) or not is_safe_content(output_raw):
                print(f"✗ 민감 콘텐츠 제외됨: {title[:60]}")
                continue

            # 전처리
            output_filtered = extract_relevant_sentences(output_raw, title)
            output_cleaned = clean_git_answer(output_filtered)

            reward = compute_reward(title, output_cleaned)

            # SFT 포맷 저장
            qa_pair = {
                "instruction": body,
                "input": title,
                "output": output_cleaned,
                "reward": reward
            }

            fout.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"✓ {title[:60]} | Reward: {reward}")
            count += 1
            time.sleep(0.2)

        except Exception as e:
            print(f"오류 발생: {e} (Title: {item.get('title', '')[:60]})")
            continue

    print(f"총 수집된 QA 수: {count}")

