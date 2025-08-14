import re
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType

# ---- 1) 개별 문자열 클리너 ----
def clean_message_text(text: str) -> str:
    if not text:
        return text

    # (a) C/C++ 스타일 한줄 주석 //... 제거 (멀티라인 대응)
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)

    # (b) 중복된 <think> 블록이 여러 번 붙어있다면, 첫 번째만 남기고 나머지 제거
    think_blocks = re.findall(r'<think>.*?</think>', text, flags=re.DOTALL)
    if len(think_blocks) > 1:
        # 본문에서 모든 think 제거 후 첫 번째 think + (나머지 본문) 형태로 재구성
        text_wo_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        first_think = think_blocks[0]
        # 맨 앞에 첫 think 배치
        text = f"{first_think} {text_wo_think}"

    # (c) 공백 정리: 탭 -> 공백, 줄 끝 공백 제거, 연속 빈 줄을 한 줄로 축소
    text = text.replace('\t', ' ')
    text = re.sub(r'[ \t]+\n', '\n', text)        # 줄 끝 공백 제거
    text = re.sub(r'\n{3,}', '\n\n', text)        # 3줄 이상 연속 개행 -> 2줄
    text = re.sub(r' {2,}', ' ', text)            # 여러 공백 -> 한 칸
    text = text.strip()

    return text

# ---- 2) messages 배열 전체를 후처리하는 함수 ----
# text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
def preprocess_messages(records):
    """
    입력: records = [[content, role, []], ...]
    동작:
      - 각 content에 clean_message_text 적용
      - 구조는 그대로 유지
    """
    if not records:
        return []

    out = []
    for rec in records:
        if not rec or len(rec) < 2:
            continue
        content, role = rec[0], rec[1]
        tool_calls = rec[2] if len(rec) > 2 else []

        content = clean_message_text(content)
        out.append([content, role, tool_calls if isinstance(tool_calls, list) else []])

    return out


