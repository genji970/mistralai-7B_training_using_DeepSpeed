import nltk

nltk.download('punkt')  # for word_tokenize, sent_tokenize
nltk.download('averaged_perceptron_tagger')  # for pos_tag
nltk.download('stopwords')  # if you're using stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_relevant_sentences(text: str, question: str, answer: str, top_k: int = 2) -> str:
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return answer.strip()

    # 임베딩
    question_emb = model.encode(question, convert_to_tensor=True)
    answer_emb = model.encode(answer, convert_to_tensor=True)
    sentence_embs = model.encode(sentences, convert_to_tensor=True)

    # 유사도 계산
    sim_q = util.cos_sim(question_emb, sentence_embs)[0]  # [num_sentences]
    sim_a = util.cos_sim(answer_emb, sentence_embs)[0]    # [num_sentences]

    # 평균 유사도 기준 상위 K개 선택
    combined_score = (sim_q + sim_a) / 2  # [num_sentences]
    top_indices = combined_score.topk(k=min(top_k, len(sentences))).indices.tolist()

    # 선택된 문장 정렬 및 연결
    selected = [sentences[i] for i in sorted(top_indices)]
    combined_answer = answer.strip() + " " + " ".join(selected).strip()

    return combined_answer.strip()