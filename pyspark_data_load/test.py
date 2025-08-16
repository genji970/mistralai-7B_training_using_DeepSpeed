import os
import sys
import time
import json
import orjson
from itertools import islice
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import get, size, expr, element_at, coalesce
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

# streaming dataset load
args=parse_args()
dataset_name = args.dataset_name
ds = load_dataset(dataset_name, split="math", streaming=True)

# ===== 첫 번째 row만 저장 =====
first_row = next(iter(ds))  # 첫 번째 row만 가져오기

print(f"첫 번째 row만 저장 완료: {first_row}")



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