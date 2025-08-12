import nltk
nltk.download('punkt')  # for word_tokenize, sent_tokenize
nltk.download('averaged_perceptron_tagger')  # for pos_tag
nltk.download('stopwords')  # if you're using stopwords
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
from nltk.tokenize import sent_tokenize

def extract_relevant_sentences(text, question, answer, top_k=2):
    try:
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return answer.strip()
        question_emb = model.encode(question, convert_to_tensor=True)
        answer_emb = model.encode(answer, convert_to_tensor=True)
        sentence_embs = model.encode(sentences, convert_to_tensor=True)
        sim_q = util.cos_sim(question_emb, sentence_embs)[0]
        sim_a = util.cos_sim(answer_emb, sentence_embs)[0]
        combined_score = (sim_q + sim_a) / 2
        top_indices = combined_score.topk(k=min(top_k, len(sentences))).indices.tolist()
        selected = [sentences[i] for i in sorted(top_indices)]
        combined_answer = answer.strip() + " " + " ".join(selected).strip()
        return combined_answer.strip()
    except Exception:
        return answer if answer else ""