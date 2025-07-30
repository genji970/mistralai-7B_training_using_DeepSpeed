from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_reward(context, instruction, output):
    emb_ctx = model.encode(context, convert_to_tensor=True)
    emb_ins = model.encode(instruction, convert_to_tensor=True)
    emb_out = model.encode(output, convert_to_tensor=True)

    sim_ctx = util.cos_sim(emb_ctx, emb_out).item()
    sim_ins = util.cos_sim(emb_ins, emb_out).item()

    sim = (sim_ctx + sim_ins) / 2
    return round(min(max(sim, 0.0), 1.0), 3)

