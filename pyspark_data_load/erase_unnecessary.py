import re
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType

def keep_qa_with_think(records):
    """
    현재 구조: [content, role, []]
    - user 질문은 그대로
    - assistant 답변은 그대로
    - assistant에 <think>...</think> 있으면 reasoning을 답변 앞에 붙임
    """
    if not records:
        return []
    
    result = []
    for rec in records:
        if len(rec) < 2:
            continue
        content, role = rec[0], rec[1]

        if role == "assistant":
            think_match = re.search(r"<think>.*?</think>", content, re.DOTALL)
            if think_match:
                content = f"{think_match.group(0)} {content}"

        result.append([content, role, []])

    return result
