import re


TOPIC_REGEX_PATTERNS = {
    "money": {
        "pattern": re.compile(r"\b(money|salary|income|rich|poor|finance|broke|debt)\b", re.IGNORECASE),
        "ratio": 0.5
    },
    "relationship": {
        "pattern": re.compile(r"\b(relationship|dating|boyfriend|girlfriend|partner|marriage|spouse)\b", re.IGNORECASE),
        "ratio": 1.0
    },
    "career": {
        "pattern": re.compile(r"\b(job|career|work|promotion|fired|boss|internship|resume|interview)\b", re.IGNORECASE),
        "ratio": 0.8
    },
    "parenting": {
        "pattern": re.compile(r"\b(parent|baby|child|kids|raise|mother|father|dad|mom|pregnant)\b", re.IGNORECASE),
        "ratio": 0.6
    },
    "food": {
        "pattern": re.compile(r"\b(food|eat|meal|cooking|snack|restaurant|dinner|lunch|breakfast)\b", re.IGNORECASE),
        "ratio": 0.4
    },

    # ✅ 수학 / CS 지식 관련 주제 10개
    "mathematics": {
        "pattern": re.compile(r"\b(math|algebra|geometry|calculus|equation|integral|derivative|proof|theorem)\b", re.IGNORECASE),
        "ratio": 1.0
    },
    "programming": {
        "pattern": re.compile(r"\b(code|coding|programming|developer|bug|function|variable|compile|runtime)\b", re.IGNORECASE),
        "ratio": 1.0
    },
    "computer science": {
        "pattern": re.compile(r"\b(algorithm|data structure|complexity|binary tree|graph theory|recursion|big-?o)\b", re.IGNORECASE),
        "ratio": 1.0
    },
    "machine learning": {
        "pattern": re.compile(r"\b(machine learning|neural network|training|loss function|backprop|gradient descent|model)\b", re.IGNORECASE),
        "ratio": 1.0
    },
    "ai ethics": {
        "pattern": re.compile(r"\b(ai ethics|alignment|bias in ai|responsible ai|explainable ai)\b", re.IGNORECASE),
        "ratio": 1.0
    },
    "hardware": {
        "pattern": re.compile(r"\b(cpu|gpu|ram|motherboard|ssd|hardware|architecture)\b", re.IGNORECASE),
        "ratio": 0.9
    },
    "networking": {
        "pattern": re.compile(r"\b(network|tcp/ip|latency|bandwidth|dns|packet|protocol|router)\b", re.IGNORECASE),
        "ratio": 0.9
    },
    "cybersecurity": {
        "pattern": re.compile(r"\b(security|hacking|malware|phishing|firewall|cyberattack|vpn)\b", re.IGNORECASE),
        "ratio": 0.9
    },
    "databases": {
        "pattern": re.compile(r"\b(database|sql|nosql|query|indexing|relational|schema|join)\b", re.IGNORECASE),
        "ratio": 1.0
    },
    "cloud computing": {
        "pattern": re.compile(r"\b(cloud|aws|gcp|azure|container|docker|kubernetes|devops)\b", re.IGNORECASE),
        "ratio": 1.0
    }
}


import random

def should_sample_regex(context: str, question: str) -> bool:
    """
    context와 question을 합쳐서 TOPIC_REGEX_PATTERNS에 따라 샘플링 여부를 결정
    """
    # None 방지
    context = context or ""
    question = question or ""
    
    # 두 텍스트를 합치고 소문자로 변환
    full_text = f"{context} {question}".lower()

    for topic, config in TOPIC_REGEX_PATTERNS.items():
        pattern = config["pattern"]
        ratio = config["ratio"]
        if pattern.search(full_text):
            return random.random() < ratio
    return True
