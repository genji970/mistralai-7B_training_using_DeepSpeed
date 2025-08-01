def extract_fields_squad(context, question, answers_dict):
    # context, question은 문자열, answers_dict는 {"text": [...]}
    try:
        context = context.strip()
        question = question.strip()
        answers = answers_dict.get("text", [])
        return (context, question, answers)
    except Exception:
        return ("", "", [])