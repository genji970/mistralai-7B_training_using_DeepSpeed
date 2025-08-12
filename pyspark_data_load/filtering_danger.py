def is_safe_content(context: str, question: str, answer: str) -> bool:
    blacklist = [
        "suicide", "kill", "death", "dying", "murder", "self-harm",
        "drugs", "marijuana", "cocaine", "meth", "LSD", "stimulant",
        "gun", "shooting", "homicide", "violence", "illegal", "rape",
        "sexual assault", "sex crime", "kidnapping", "terror", "terrorism",
        "bombing", "crime", "sexual exploitation", "abuse", "suicide attempt",
        "sex", "sexiest"
    ]
    # None 값 방지 후 하나의 문자열로 합치기
    full_text = " ".join(filter(None, [context, question, answer]))
    
    if not full_text.strip():
        return True  # 빈 텍스트는 안전하다고 간주
    
    lowered = full_text.lower()
    return not any(keyword in lowered for keyword in blacklist)
