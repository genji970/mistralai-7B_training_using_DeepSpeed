def is_safe_content(text: str) -> bool:
    blacklist = [
        "suicide", "kill", "death", "dying", "murder", "self-harm",
        "drugs", "marijuana", "cocaine", "meth", "LSD", "stimulant",
        "gun", "shooting", "homicide", "violence", "illegal", "rape",
        "sexual assault", "sex crime", "kidnapping", "terror", "terrorism",
        "bombing", "crime", "sexual exploitation", "abuse", "suicide attempt",
        "sex", "sexiest"
    ]
    if not text:
        return True  # 빈 텍스트는 안전하다고 간주
    lowered = text.lower()
    return not any(keyword in lowered for keyword in blacklist)