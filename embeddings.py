from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def get_embeddings(texts):

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings