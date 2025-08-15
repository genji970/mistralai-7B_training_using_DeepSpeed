import os
import sys
import time
import json
import orjson
from itertools import islice
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import get, size
from concurrent.futures import ThreadPoolExecutor
from parser import parse_args
from pathlib import Path
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StringType, FloatType, BooleanType, ArrayType

from data_score_func import compute_reward
from eliminating_symbols import clean_git_answer
from filtering_danger import is_safe_content
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
    udf_compute_reward = udf(compute_reward, FloatType())
    udf_clean_git_answer = udf(clean_git_answer, StringType())
    udf_is_safe_content = udf(is_safe_content, BooleanType())
    udf_keep_qa_with_think = F.udf(
    keep_qa_with_think,
    ArrayType(ArrayType(StringType()))
    )
    udf_preprocess_messages = F.udf(preprocess_messages, ArrayType(StringType()))

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

    metadata_schema = StructType([
    StructField("expected_answer", ArrayType(StringType()), True)
    ])

    if "metadata" in df_sampled.columns:
        df_sampled = df_sampled.withColumn(
        "metadata",
        from_json(col("metadata"), metadata_schema)
        )  # StringType -> StructType 변환

        if "expected_answer" in df_sampled.select("metadata.*").columns:
            try:
                field_type = df_sampled.schema["metadata"].dataType["expected_answer"].dataType
                if isinstance(field_type, ArrayType):
                    df_sampled = df_sampled.withColumn(
                    "expected_answer",
                    col("metadata.expected_answer").getItem(0)
                    )
                    print("expected_answer 평탄화 성공")
                else:
                    df_sampled = df_sampled.withColumn(
                    "expected_answer",
                    col("metadata.expected_answer")
                        )
                    print("expected_answer 이미 평탄화됨")
            except Exception as e:
                print("expected_answer 처리 오류:", e)

    if args.debug == 'True':
        df_sampled.printSchema()
        df_sampled.show(3, truncate=False)

    get_user_question = F.udf(
    lambda msgs: next(
        (m["content"] for m in msgs if isinstance(m, dict) and m.get("role") == "user"),
        None
    ) if msgs else None,
    StringType()
    )

    get_assistant_answer = F.udf(
    lambda msgs: [m["content"] for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"] if msgs else [],
    ArrayType(StringType())
    )

    get_expected_answer = F.udf(
    lambda metadata: metadata.get("expected_answer") if isinstance(metadata, dict) else None,
    StringType()
    )

    if args.debug == 'True':
        print("None?")
        df_sampled.select("messages").show(1, truncate=False)

    # context는 따로 없으니 None으로 초기화 (또는 messages 합쳐서 생성)
    df_sampled = (
    df_sampled
    .withColumn("question", get_user_question(F.col("messages")))
    .withColumn("answers", get_assistant_answer(F.col("messages")))
    .withColumn("expected_answers", get_expected_answer(F.col("metadata")))
    )

    if args.debug == 'True':
        print("None?")
        df_sampled.select("messages").show(1, truncate=False)

    if args.debug == 'True':
        print("None?")
        df_sampled.select("messages").show(1, truncate=False)

    # answers가 비어있는 row 제거
    df_sampled = df_sampled.filter(size(col("answers")) > 0)

    if args.debug == 'True':
        df_sampled.printSchema()
        df_sampled.show(3, truncate=False)

    start_time = time.time()

    df_proc = (
    df_sampled
    # .withColumn("question", udf_extract_question(col("context"), col("question"), col("answers")))
    # .withColumn("answers", udf_extract_answers(col("context"), col("question"), col("answers")))
    .withColumn("answer", get(col("answers"), 0))
    .withColumn("safe_ok", udf_is_safe_content("question", "answer"))
    .withColumn("output_cleaned", udf_clean_git_answer("answer"))
    .withColumn("answers", udf_preprocess_messages(F.col("answer")))
    )

    print("변환 후 컬럼:", df_proc.columns)
    print("변환된 데이터 샘플:")
    #df_proc.show(3, truncate=120) -> java heap space oom 발생함

    df_proc = df_proc.filter(col("safe_ok"))

    select_cols = ["question", "expected_answer", "output_cleaned"]
    if getattr(args, "rlhf_mode", False):
        df_proc = df_proc.withColumn("reward", udf_compute_reward("context", "question", "output_cleaned"))
        select_cols.append("reward")
        print("RLHF mode: reward 컬럼 추가")

    # ======= 경로 충돌 방지: 결과는 원본 폴더(output_path) 하위 'spark_out'에 저장 =======
    spark_out_dir = os.path.join(output_path, "spark_out")
    os.makedirs(spark_out_dir, exist_ok=True)

    df_proc.select(*select_cols).write.mode("overwrite").json(spark_out_dir)
    print("Spark 결과 파일 저장 완료")
    print("저장된 파일 목록:", os.listdir(spark_out_dir))

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Spark 처리 및 저장 완료: {spark_out_dir}")
    print(f"총 소요 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")

    spark.stop()
