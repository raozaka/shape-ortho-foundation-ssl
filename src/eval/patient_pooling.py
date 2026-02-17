import numpy as np

def mean_pool(embeddings, patient_ids):
    pooled = {}

    for emb, pid in zip(embeddings, patient_ids):
        pooled.setdefault(pid, []).append(emb)

    return np.array([np.mean(v, axis=0) for v in pooled.values()])
