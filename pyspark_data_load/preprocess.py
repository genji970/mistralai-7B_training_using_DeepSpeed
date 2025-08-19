import re
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, StructType, StructField

# ---- 1) 개별 문자열 클리너 ----
def clean_message_text(text: str) -> str:
    if not text:
        return text

    # 1) 줄 전체 주석 제거
    text = re.sub(r'//.*', '', text)  

    # 2) 남아있는 '//' 문자열 자체 제거 (URL 등도 날아감)
    text = text.replace('//', '')

    # 3) 공백 정리
    text = text.replace('\t', ' ')
    text = re.sub(r'[ \t]+\n', '\n', text)        
    text = re.sub(r'\n{3,}', '\n\n', text)        
    text = re.sub(r' {2,}', ' ', text)            
    text = text.strip()

    return text

# ---- 2) messages 배열 전체를 후처리하는 함수 ----
# text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
def preprocess_messages(answer: str):
    if not answer or len(answer) < 2:
        return ""
    answer = re.sub(r'//+', '', answer)
    answer = re.sub(r'\\+', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    answer = clean_message_text(answer)
    return answer

def make_think_variants(answer: str, max_versions: int = 1):
    if not answer:
        return [""] * max_versions

    # 모든 <think>...</think> 구간 추출
    thinks = re.findall(r'<think>.*?</think>', answer, flags=re.DOTALL)
    final_text = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

    if len(thinks) == 0:
        return [final_text] * max_versions

    # 마지막 <think>는 항상 답변으로 고정
    answer_think = thinks[-1]
    useful_thinks = thinks[:-1]

    variants = []
    n = len(useful_thinks)

    for i in range(max_versions):
        if i < n:
            # 끝에서 i+1번째 요소 제거
            temp = useful_thinks[:]
            del temp[-(i+1)]
            reduced = ''.join(temp + [answer_think]) + " " + final_text
        else:
            # 더 이상 뺄 게 없으면 그냥 최종 답변만
            reduced = answer_think + " " + final_text
        variants.append(reduced.strip())

    return variants


