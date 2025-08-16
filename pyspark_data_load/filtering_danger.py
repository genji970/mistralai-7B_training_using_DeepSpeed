def is_safe_content(question:list[str], answer:str) -> bool:
    blacklist = [
        "suicide", "kill", "death", "dying", "murder", "self-harm",
        "drugs", "marijuana", "cocaine", "meth", "LSD", "stimulant",
        "gun", "shooting", "homicide", "violence", "illegal", "rape",
        "sexual assault", "sex crime", "kidnapping", "terror", "terrorism",
        "bombing", "crime", "sexual exploitation", "abuse", "suicide attempt",
        "sex", "sexiest"
    ]

    # list[str] → str 변환
    if isinstance(question, list):
        question = " ".join(str(q) for q in question if q)
    if isinstance(answer, list):
        answer = " ".join(str(a) for a in answer if a)

    # None 방지 후 합치기
    full_text = " ".join(filter(None, [question, answer]))

    if not full_text.strip():
        return True

    lowered = full_text.lower()
    return not any(keyword in lowered for keyword in blacklist)
