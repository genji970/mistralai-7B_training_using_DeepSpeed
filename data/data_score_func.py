from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_reward(instruction, output):
    emb_q = model.encode(instruction, convert_to_tensor=True)
    emb_a = model.encode(output, convert_to_tensor=True)
    sim = util.cos_sim(emb_q, emb_a).item()
    return round(min(max(sim, 0.0), 1.0), 3)
