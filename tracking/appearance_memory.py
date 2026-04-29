from collections import deque

import numpy as np


def _l2_normalize(vec):
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec
    return vec / norm


class AppearanceMemory:
    def __init__(self, max_entries=8):
        self.max_entries = max_entries
        self.reset()

    def reset(self):
        self.embeddings = deque(maxlen=self.max_entries)
        self.points = deque(maxlen=self.max_entries)
        self.boxes = deque(maxlen=self.max_entries)

    def has_reference(self):
        return len(self.embeddings) > 0

    def observe(self, signature, confidence=1.0):
        if signature is None or confidence < 0.65:
            return

        embedding = signature.get("embedding")
        if embedding is None:
            return

        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.size == 0:
            return

        self.embeddings.append(_l2_normalize(embedding))
        self.points.append(signature.get("centroid") or signature.get("point"))
        self.boxes.append(signature.get("crop_box"))

    def similarity(self, embedding):
        if embedding is None or not self.embeddings:
            return 0.0

        candidate = _l2_normalize(np.asarray(embedding, dtype=np.float32))
        scores = [float(np.dot(candidate, ref)) for ref in self.embeddings]
        if not scores:
            return 0.0
        return max(scores)

    def recent_point(self):
        if not self.points:
            return None
        return self.points[-1]

    def recent_box(self):
        if not self.boxes:
            return None
        return self.boxes[-1]
