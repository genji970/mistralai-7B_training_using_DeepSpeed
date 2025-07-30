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

def should_sample_regex(title: str, body: str, topic_ratios: dict) -> bool:
    full_text = (title + " " + body).lower()
    for keyword, ratio in topic_ratios.items():
        if keyword in full_text:
            return random.random() < ratio  # 확률에 따라 통과 여부 결정
    return True  # 키워드가 없으면 기본적으로 포함