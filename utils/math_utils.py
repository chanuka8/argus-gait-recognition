import numpy as np


def l2_normalize(vector, eps: float = 1e-8):
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    return array / (norm + eps)


def batch_l2_normalize(matrix, eps: float = 1e-8):
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / (norms + eps)


def cosine_similarity(vector_a, vector_b) -> float:
    a = l2_normalize(vector_a)
    b = l2_normalize(vector_b)
    return float(np.dot(a, b))


def cosine_similarity_batch(query, gallery):
    query = l2_normalize(query)
    gallery = batch_l2_normalize(gallery)
    return np.dot(gallery, query)