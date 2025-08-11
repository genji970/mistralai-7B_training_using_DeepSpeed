import os
import sys
import time
import json
import orjson
from itertools import islice
from concurrent.futures import ThreadPoolExecutor
from parser import parse_args
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

if __name__ == "__main__":
    args = parse_args()

    print(f"✅ 저장 경로: {args.save_path}")
    print(f"✅ 데이터셋 이름: {args.dataset_name}")

    safe_name = args.dataset_name.replace("/", "__")
    os.makedirs(args.save_path, exist_ok=True)
    output_path = os.path.join(args.save_path, safe_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"✅ 폴더 생성됨: {output_path}")
    else:
        print(f"✅ 이미 폴더 존재: {output_path}")

    dataset_name = args.dataset_name
    write_path = os.path.join(output_path, safe_name + ".jsonl")

    # HF Transfer 가속
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # 데이터셋 저장 및 로드
    """
    D:/에 cache저장하고 싶으면, windows powershell 기준,
    $env:HF_HOME = "D:/hf_home"
    $env:HF_DATASETS_CACHE = "D:/hf_datasets_cache"
    python collect_data.py --save_path D:/ --file_name nvidia --dataset_name nvidia/Nemotron-Post-Training-Dataset-v1 --sample_ratio 0.1
    """

    # streaming dataset load
    ds = load_dataset(dataset_name, split="math", streaming=True)

    # ===== 빠른 배치 저장 함수 =====
    def process_batch(batch):
        return [orjson.dumps(ex) + b"\n" for ex in batch]

    BATCH_SIZE = 2
    MAX_WORKERS = 1

    start_dl = time.time()
    with open(write_path, "wb", buffering=1024 * 1024) as f:
        it = iter(ds)
        wrote_first = False
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            while True:
                chunk = list(islice(it, BATCH_SIZE))
                if not chunk:
                    break
                futures = executor.submit(process_batch, chunk)
                lines = futures.result()
                f.writelines(lines)
                if not wrote_first:
                    print(f"✅ 원본 jsonl 파일 저장 시작: {safe_name + '.jsonl'}")
                    wrote_first = True
    end_dl = time.time()
    print(f"✅ Streaming 저장 완료 (소요: {end_dl - start_dl:.2f}초)")
    sys.stdout.flush()

    # ===== Spark UDF 등록 =====
    udf_extract_relevant_sentences = udf(extract_relevant_sentences, StringType())
    udf_compute_reward = udf(compute_reward, FloatType())
    udf_clean_git_answer = udf(clean_git_answer, StringType())
    udf_is_safe_content = udf(is_safe_content, BooleanType())
    udf_should_sample_regex = udf(should_sample_regex, BooleanType())
    udf_extract_context = udf(
        lambda context, question, answers_dict: extract_fields_squad(context, question, answers_dict)[0], StringType())
    udf_extract_question = udf(
        lambda context, question, answers_dict: extract_fields_squad(context, question, answers_dict)[1], StringType())
    udf_extract_answers = udf(
        lambda context, question, answers_dict: extract_fields_squad(context, question, answers_dict)[2],
        ArrayType(StringType()))

    with open(write_path, encoding="utf-8") as f:
        print("샘플:", f.readline().strip()[:200])
        sys.stdout.flush()

    spark = SparkSession.builder.appName(safe_name).getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    df = spark.read.json(write_path)
    print(f"✅ Spark DataFrame 로드 완료, row 수: {df.count()}")
    sys.stdout.flush()

    # 카테고리별 샘플링
    sampled_dfs = []
    categories = [row["category"] for row in df.select("category").distinct().collect()]
    print(f"카테고리 목록: {categories}")
    sys.stdout.flush()
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
    sys.stdout.flush()
    print("샘플링 데이터 컬럼:", df_sampled.columns)
    sys.stdout.flush()

    if "answers" in df_sampled.columns:
        try:
            df_sampled = df_sampled.withColumn("output", col("answers.text")[0])
            print("answers.text 평탄화 성공")
        except Exception as e:
            print(" 평탄화 오류:", e)

    start_time = time.time()

    df_proc = (
        df_sampled
        .withColumn("context", udf_extract_context(col("*")))
        .withColumn("question", udf_extract_question(col("*")))
        .withColumn("answers", udf_extract_answers(col("*")))
        .withColumn("answer", col("answers")[0])
        .withColumn("topic_ok", udf_should_sample_regex("context", "question"))
        .withColumn("safe_ok", udf_is_safe_content("context", "question", "answer"))
        .withColumn("output_filtered", udf_extract_relevant_sentences("context", "question", "answer"))
        .withColumn("output_cleaned", udf_clean_git_answer("output_filtered"))
    )

    print("UDF 변환 적용 후 row 수:", df_proc.count())
    print("변환 후 컬럼:", df_proc.columns)
    print("변환된 데이터 샘플:")
    df_proc.show(3, truncate=120)

    df_proc = df_proc.filter(col("topic_ok") & col("safe_ok"))
    print("필터링 적용 후 row 수:", df_proc.count())
    sys.stdout.flush()

    select_cols = ["question", "context", "output_cleaned"]
    if getattr(args, "rlhf_mode", False):
        df_proc = df_proc.withColumn("reward", udf_compute_reward("context", "question", "output_cleaned"))
        select_cols.append("reward")
        print("RLHF mode: reward 컬럼 추가")
        sys.stdout.flush()

    df_proc.select(*select_cols).write.mode("overwrite").json(output_path)
    print("Spark 결과 파일 저장 완료")
    sys.stdout.flush()
    print("저장된 파일 목록:", os.listdir(output_path))
    sys.stdout.flush()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Spark 처리 및 저장 완료: {output_path}")
    print(f"총 소요 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")

    spark.stop()
