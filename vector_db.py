import faiss
import numpy as np

class VectorDB:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, embeddings, chunks):

        self.index.add(
            np.array(embeddings).astype("float32")
        )

        self.texts.extend(chunks)

    def search(self, query_embedding, k=5):

        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"),
            k
        )

        results = []

        for idx in indices[0]:
            results.append(self.texts[idx])

        return results