import nltk

nltk.download('punkt')  # for word_tokenize, sent_tokenize
nltk.download('averaged_perceptron_tagger')  # for pos_tag
nltk.download('stopwords')  # if you're using stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_relevant_sentences(text: str, question: str, drop_rate=0.8) -> str:
    sentences = sent_tokenize(text)
    total = len(sentences)
    if total == 0:
        return ""

    # 임베딩
    question_emb = model.encode(question, convert_to_tensor=True)
    sentence_embs = model.encode(sentences, convert_to_tensor=True)

    # 유사도 계산
    sims = util.cos_sim(question_emb, sentence_embs)[0]  # shape: [num_sentences]
    top_idx = sims.argmax().item()  # 가장 유사한 문장 인덱스

    # 추출할 문장 수
    keep_count = max(1, int(total * (1 - drop_rate)))  # 최소 1문장 유지
    half_window = keep_count // 2

    # 윈도우 기반 전후 인덱스 계산
    start = max(0, top_idx - half_window)
    end = min(total, top_idx + half_window + 1)

    selected_sentences = sentences[start:end]
    return " ".join(selected_sentences)