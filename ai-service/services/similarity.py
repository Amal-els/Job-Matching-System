import numpy as np


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def cosine_top_k(query_vector, candidate_vectors, top_k):
    """
    Vectorized cosine similarity for many candidates.
    Returns (indices, scores) sorted by descending similarity.
    """
    if candidate_vectors is None or len(candidate_vectors) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    query = np.asarray(query_vector, dtype=np.float32)
    matrix = np.asarray(candidate_vectors, dtype=np.float32)
    if matrix.ndim != 2 or query.ndim != 1:
        return np.array([], dtype=int), np.array([], dtype=float)

    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    safe_denominator = matrix_norms * (query_norm if query_norm > 0 else 1.0)
    safe_denominator[safe_denominator == 0] = 1.0
    scores = np.dot(matrix, query) / safe_denominator

    k = min(max(0, int(top_k)), len(scores))
    if k == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if k == len(scores):
        indices = np.argsort(scores)[::-1]
    else:
        partial = np.argpartition(scores, -k)[-k:]
        indices = partial[np.argsort(scores[partial])[::-1]]
    return indices.astype(int), scores[indices].astype(float)
