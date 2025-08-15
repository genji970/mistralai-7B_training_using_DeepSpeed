import os
import sys
import time
import json
import orjson
from itertools import islice
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.functions import get, size
from concurrent.futures import ThreadPoolExecutor
from parser import parse_args
from pathlib import Path
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, FloatType, BooleanType, ArrayType

from choose_relevant import extract_relevant_sentences
from data_score_func import compute_reward
from eliminating_symbols import clean_git_answer
from filtering_danger import is_safe_content
from topic_ratio_criteria import should_sample_regex
from train_data_fit import extract_fields_squad
from erase_unnecessary import keep_qa_with_think
from preprocess import preprocess_messages

if __name__ == "__main__":
    args = parse_args()

    print(f"저장 경로: {args.save_path}")
    print(f"데이터셋 이름: {args.dataset_name}")

    safe_name = args.dataset_name.replace("/", "__")
    os.makedirs(args.save_path, exist_ok=True)
    output_path = os.path.join(args.save_path, safe_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"폴더 생성됨: {output_path}")
    else:
        print(f"이미 폴더 존재: {output_path}")

    dataset_name = args.dataset_name
    write_path = os.path.join(output_path, safe_name + ".jsonl")

    # HF Transfer 가속
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # 데이터셋 저장 및 로드
    """
    캐시 경로 예시 (POSIX 쉘):
    export HF_HOME="~/hf_home"
    export HF_DATASETS_CACHE="~/hf_datasets_cache"
    python collect_data.py --save_path /data --file_name nvidia --dataset_name nvidia/Nemotron-Post-Training-Dataset-v1 --sample_ratio 0.1
    """

    # streaming dataset load
    ds = load_dataset(dataset_name, split="math", streaming=True)

    # ===== 빠른 배치 저장 함수 =====
    def process_batch(batch):
        return [orjson.dumps(ex) + b"\n" for ex in batch]

    BATCH_SIZE = 4096
    MAX_WORKERS = 8

    if os.path.exists(write_path):
        print(f"이미 파일이 존재하여 저장 과정 건너뜀: {write_path}")
    else:
        start_dl = time.time()
        with open(write_path, "wb", buffering=1024 * 1024) as f:
            it = iter(ds)
            wrote_first = False

            # 진행률 집계 변수
            rows_written = 0
            report_every_batches = 1 
            batch_idx = 0
            last_report_t = start_dl

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                while True:
                    chunk = list(islice(it, BATCH_SIZE))
                    if not chunk:
                        break
                    fut = executor.submit(process_batch, chunk)
                    lines = fut.result()  # bytes 리스트여야 함
                    f.writelines(lines)

                    # 첫 기록 로그
                    if not wrote_first:
                        print(f"원본 jsonl 파일 저장 시작: {safe_name + '.jsonl'}")
                        wrote_first = True

                    # 진행률 집계/출력
                    batch_idx += 1
                    rows_written += len(chunk)
                    now = time.time()
                    elapsed = now - start_dl
                    rate = rows_written / elapsed if elapsed > 0 else 0.0
                    # 파일 크기는 tell()로 즉시 확인(버퍼링 중이어도 파일 오프셋은 증가)
                    bytes_written = f.tell()

                    if batch_idx % report_every_batches == 0 or (now - last_report_t) >= 2.0:
                        # 같은 줄 갱신
                        print(
                            f"\r[streaming] rows={rows_written:,} | size={bytes_written/1e6:.1f}MB | "
                            f"rate={rate:,.0f} rec/s | elapsed={elapsed:.1f}s",
                            end="",
                            flush=True,
                        )
                        last_report_t = now

            # 마지막 줄 개행
            print()

        end_dl = time.time()
        print(f"Streaming 저장 완료 (소요: {end_dl - start_dl:.2f}초)")

    # ===== Spark UDF 등록 =====
    udf_extract_relevant_sentences = udf(extract_relevant_sentences, StringType())
    udf_compute_reward = udf(compute_reward, FloatType())
    udf_clean_git_answer = udf(clean_git_answer, StringType())
    udf_is_safe_content = udf(is_safe_content, BooleanType())
    udf_should_sample_regex = udf(should_sample_regex, BooleanType())
    udf_keep_qa_with_think = F.udf(
    keep_qa_with_think,
    ArrayType(ArrayType(StringType()))
    )
    udf_preprocess_messages = udf(preprocess_messages, ArrayType(ArrayType(StringType())))
    udf_extract_context = udf(
        lambda context, question, answers_dict: extract_fields_squad(context, question, answers_dict)[0], StringType())
    udf_extract_question = udf(
        lambda context, question, answers_dict: extract_fields_squad(context, question, answers_dict)[1], StringType())
    udf_extract_answers = udf(
        lambda context, question, answers_dict: extract_fields_squad(context, question, answers_dict)[2],
        ArrayType(StringType()))

    # 경로 디버그 + URI 변환
    p = Path(write_path).resolve()
    print("exists?", p.exists(), "| path:", p)
    spark_uri = p.as_uri()  # 예: file:///.../nvidia__Nemotron-Post-Training-Dataset-v1.jsonl
    print("spark_uri:", spark_uri)  # <-- 경로 문제 디버깅을 위해 URI도 출력

    # 첫 줄 프리뷰 (텍스트 모드)
    with open(write_path, "r", encoding="utf-8", errors="replace") as f:
        print("샘플:", f.readline().strip()[:200])

    # Spark 세션 및 JSONL 로드 (중요: spark_uri 사용)
    if args.debug == 'False':
        spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(safe_name)
        .config("spark.executor.memory", "8g")
        .config("spark.driver.memory", "8g")
        .config("spark.python.worker.memory", "4g")  # Python worker 메모리 제한 완화
        .getOrCreate()
        )

    if args.debug == 'True':
        spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(safe_name)
        .config("spark.executor.memory", "8g")
        .config("spark.driver.memory", "8g")
        .config("spark.python.worker.memory", "4g")  # Python worker 메모리 제한 완화
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
        )
    spark.sparkContext.setLogLevel("INFO")

    # 로컬 파일은 URI로 읽는 게 안전
    df = (
        spark.read
        .option("multiLine", "false")  # JSON Lines
        .json(spark_uri)
    )
    print(f"Spark DataFrame 로드 완료, row 수: {df.count()}")
    
    # 카테고리별 샘플링
    sampled_dfs = []
    categories = [row["category"] for row in df.select("category").distinct().collect()]
    print(f"카테고리 목록: {categories}")
    for cat in categories:
        df_cat = df.filter(col("category") == cat)
        n = int(df_cat.count() * args.sample_ratio)
        print(f"카테고리 '{cat}' 샘플 수: {n}")
        df_cat_sample = df_cat.limit(n)
        sampled_dfs.append(df_cat_sample)
    df_sampled = sampled_dfs[0]
    for df_other in sampled_dfs[1:]:
        df_sampled = df_sampled.union(df_other)

    print("샘플링 데이터 row 수:", df_sampled.count())
    print("샘플링 데이터 컬럼:", df_sampled.columns)

    if "expected_answers" in df_sampled.columns:
        try:
            df_sampled = df_sampled.withColumn("output", col("answers.text")[0])
            print("answers.text 평탄화 성공")
        except Exception as e:
            print(" 평탄화 오류:", e)



# head -n 2 nvidia__Nemotron-Post-Training-Dataset-v1/spark_out/part-00000*.json | tail -n 1


"""
{
  "category": "math",
  "generator": "DeepSeek-R1-0528",
  "license": "CC BY 4.0",
  "messages": [
    {"role": "user", "content": "문제 텍스트"},
    {"role": "assistant", "content": "<think>... reasoning ...</think> 실제 답변"}
  ],
  "metadata": {
    "expected_answer": "$V=...",
    "problem_source": "aops_c6_high_school_olympiads"
  },
  "reasoning": "on",
  "uuid": "fb215f9b-9372-4a08-875e-94184764ad51",
  "version": "v1",
  "expected_answer": "$V=...",
  "problem_source": "aops_c6_high_school_olympiads"
}

"""