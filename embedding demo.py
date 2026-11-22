from sentence_transformers import SentenceTransformer

if __name__ == '__main__':

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    sentences = [
        "Colorless green ideas sleep furiously.."
    ]

    embeddings = model.encode(sentences)

    print("Embedding shape:", embeddings.shape)
    print("Values:", embeddings.round(3))
