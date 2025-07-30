def is_safe_content(text: str) -> bool:
    """
    민감한 키워드를 포함한 콘텐츠인지 여부를 판단.
    포함되어 있으면 False 반환 (제외 대상)
    """
    blacklist = [
        "suicide", "kill", "death", "dying", "murder", "self-harm",
        "drugs", "marijuana", "cocaine", "meth", "LSD", "stimulant",
        "gun", "shooting", "homicide", "violence", "illegal", "rape",
        "sexual assault", "sex crime", "kidnapping", "terror", "terrorism",
        "bombing", "crime", "sexual exploitation", "abuse", "suicide attempt",
        "sex","sexiest"
    ]
    lowered = text.lower()
    return not any(keyword.lower() in lowered for keyword in blacklist)
