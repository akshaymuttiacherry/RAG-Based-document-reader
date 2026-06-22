from rank_bm25 import BM25Okapi

class HybridSearch:

    def __init__(self, chunks):

        self.chunks = chunks

        tokenized = [
            chunk.split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

    def keyword_search(self, query):

        scores = self.bm25.get_scores(
            query.split()
        )

        top_indices = scores.argsort()[-5:]

        return [
            self.chunks[i]
            for i in top_indices
        ]