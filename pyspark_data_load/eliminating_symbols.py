import re


def clean_git_answer(text: str) -> str:
    if not text:
        return ""
    # 주석 (# ...) 제거
    text = re.sub(r'#.*', '', text)
    # 특수문자 \n 제거 및 줄바꿈 정리
    text = text.replace('\\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    # 코드 블록 구분 기호나 이상한 기호 제거 (예: ===, ---)
    text = re.sub(r'[-=*]{2,}', '', text)
    # 명령어만 있는 줄 제거
    lines = [line.strip() for line in text.split('\n') if not re.match(r'^\$[\w\s\.\-\~]+$', line.strip())]
    return ' '.join(lines).strip()