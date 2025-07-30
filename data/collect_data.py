import os
import json
import time
import praw

from choose_relevant import extract_relevant_sentences
from eliminating_symbols import clean_git_answer
from filtering_danger import is_safe_content
from data_score_func import compute_reward
from topic_ratio_criteria import should_sample_regex , TOPIC_REGEX_PATTERNS

# data 경로

# 저장 경로
save_dir = "file_path"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, "qa_pairs_with_reward.ndjson")

with open(output_path, "w", encoding="utf-8") as fout:
    count = 0
    for :
        try:
            if :
                continue

            title = submission.title.strip()
            body = submission.selftext.strip() if submission.selftext else ""

            if not should_sample_regex(title + " " + body, TOPIC_REGEX_PATTERNS):
                print(f"✗ 주제 샘플링 제외됨: {title[:60]}")
                continue

            submission.comments.replace_more(limit=0)
            comments = [c for c in submission.comments if hasattr(c, "body") and c.body.strip()]
            if not comments:
                continue

            top_comment = max(comments, key=lambda c: c.score)
            output_raw = top_comment.body.strip()

            if not is_safe_content(title) or not is_safe_content(body) or not is_safe_content(output_raw):
                print(f"✗ 민감 콘텐츠 제외됨: {title[:60]}")
                continue

            output_filtered = extract_relevant_sentences(output_raw, title)
            output_cleaned = clean_git_answer(output_filtered)

            reward = compute_reward(title, output_cleaned)

            # ✅ SFT 확장형 포맷 (Mistral-7B 학습용)
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
            print(f"오류 발생: {e} (Submission ID: {submission.id}, Title: {submission.title[:60]})")
            continue

    print(f"총 수집된 Reddit Q&A 수: {count}")
