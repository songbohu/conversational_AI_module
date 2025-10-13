from dialogue_system import DialogueSystem
from openai import OpenAI
import numpy as np
import json
import os
import sys


class RAGBot(DialogueSystem):
    """
    A retrieval-augmented dialogue agent.
    It retrieves relevant knowledge snippets from a local knowledge base
    using OpenAI's embedding API before generating responses.
    """

    def __init__(self,
                 model: str = "gpt-5-nano-2025-08-07",
                 embedding_model: str = "text-embedding-3-small",
                 key_path: str = "openai.key",
                 kb_path: str = "cambridge_knowledge_list.json"):
        super().__init__()
        self.model = model
        self.embedding_model = embedding_model
        self.kb_path = kb_path
        self._load_openai_key(key_path)
        self.client = OpenAI()

        # --- Load Knowledge Base ---
        if not os.path.exists(kb_path):
            sys.exit(f"Error: The knowledge base '{kb_path}' was not found.")
        with open(kb_path, "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)
        self.doc_texts = [d["text"] for d in self.knowledge_base]
        print(f"Knowledge base loaded with {len(self.doc_texts)} entries.\n")

        # --- Embedding Cache Path ---
        self.embedding_cache_path = os.path.splitext(kb_path)[0] + "_embeddings.npy"

        # --- Load or Create Embeddings ---
        if os.path.exists(self.embedding_cache_path):
            print(f"Loading cached embeddings from {self.embedding_cache_path} ...")
            self.doc_embeddings = np.load(self.embedding_cache_path)
        else:
            print("Computing embeddings for knowledge base ...")
            self.doc_embeddings = self._embed_texts(self.doc_texts)
            np.save(self.embedding_cache_path, self.doc_embeddings)
            print(f"Embeddings saved to cache: {self.embedding_cache_path}\n")

    def _load_openai_key(self, key_path: str):
        if not os.path.exists(key_path):
            sys.exit(f"Error: The API key file '{key_path}' was not found.")
        with open(key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
        print("OpenAI API key loaded successfully.\n")

    def _embed_texts(self, texts):
        """Generate embeddings for a list of texts using OpenAI embedding API."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def _embed_query(self, query):
        """Generate embedding for a single query."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        return np.array(response.data[0].embedding)

    def retrieve_context(self, query, top_k: int = 3):
        """Retrieve top-k most relevant snippets using cosine similarity."""
        query_emb = self._embed_query(query)
        scores = np.dot(self.doc_embeddings, query_emb.T) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(scores)[::-1][:top_k]
        retrieved_texts = [self.doc_texts[i] for i in top_indices]

        print("\nRetrieved Knowledge Snippets:")
        for i, text in enumerate(retrieved_texts, 1):
            print(f"[{i}] {text}")
        print("------------------------------------------------------------\n")

        return "\n".join(retrieved_texts)

    def chat(self, utterance: str) -> dict:
        """Main chat logic: retrieve context → generate answer."""
        # Step 1: Retrieve top relevant snippets
        retrieved_context = self.retrieve_context(utterance)

        # Step 2: Construct conversation with system prompt
        system_prompt = (
            "You are a friendly and knowledgeable Cambridge student who helps "
            "others learn about university life. "
            "Use the retrieved context below to answer accurately and naturally. "
            "If you don't know, say so politely.\n\n"
            f"--- Retrieved context ---\n{retrieved_context}\n--- End context ---"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.conversation_history:
            role = "assistant" if turn["speaker"] == "assistant" else "user"
            messages.append({"role": role, "content": turn["utterance"]})
        messages.append({"role": "user", "content": utterance})

        response = self.client.responses.create(
            model=self.model,
            input=messages
        )

        reply_text = response.output_text.strip()

        return {
            "text": reply_text,
            "retrieved_context": retrieved_context
        }


if __name__ == "__main__":
    bot = RAGBot()
    bot.start_a_chat()