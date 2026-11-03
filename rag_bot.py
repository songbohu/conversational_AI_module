from dialogue_system import DialogueSystem
from openai import OpenAI
from openai.types.responses import ResponseInputParam
from typing import Literal
import numpy as np
import json
import os
import sys

class RAGBot(DialogueSystem):
    """
    A retrieval-augmented dialogue system.
    It retrieves relevant knowledge snippets using embeddings,
    then generates responses using GPT-6 Luna.
    """

    def __init__(self,
                 model: str = "gpt-6-luna",
                 embedding_model: str = "text-embedding-3-small",
                 key_path: str = "openai.key",
                 kb_path: str = "cambridge_knowledge_list.json"):
        super().__init__()
        self.model = model
        self.embedding_model = embedding_model
        self._load_openai_key(key_path)
        self.client = OpenAI()

        # Load knowledge base
        if not os.path.exists(kb_path):
            sys.exit(f"Error: The knowledge base '{kb_path}' was not found.")

        with open(kb_path, "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)

        self.doc_texts = [d["text"] for d in self.knowledge_base]
        print(f"Knowledge base loaded with {len(self.doc_texts)} entries.\n")

        # Load or create embeddings
        self.embedding_cache_path = (
            os.path.splitext(kb_path)[0] + "_embeddings.npy"
        )

        if os.path.exists(self.embedding_cache_path):
            print("Loading cached embeddings...")
            self.doc_embeddings = np.load(self.embedding_cache_path)
        else:
            print("Computing embeddings for knowledge base...")
            self.doc_embeddings = self._embed_texts(self.doc_texts)
            np.save(self.embedding_cache_path, self.doc_embeddings)
            print("Embeddings saved to cache.\n")

    def _load_openai_key(self, key_path: str):
        if not os.path.exists(key_path):
            sys.exit(f"Error: The API key file '{key_path}' was not found.")

        with open(key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()

        os.environ["OPENAI_API_KEY"] = api_key
        print("OpenAI API key loaded successfully.\n")

    def _embed_texts(self, texts):
        """Generate embeddings for a list of texts."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def _embed_query(self, query):
        """Generate an embedding for a single query."""
        return self._embed_texts([query])[0]

    def retrieve_context(self, query, top_k: int = 3):
        """Retrieve the top-k snippets using cosine similarity."""
        query_emb = self._embed_query(query)

        scores = np.dot(self.doc_embeddings, query_emb) / (
            np.linalg.norm(self.doc_embeddings, axis=1)
            * np.linalg.norm(query_emb)
        )

        top_indices = np.argsort(scores)[::-1][:top_k]
        retrieved_texts = [self.doc_texts[i] for i in top_indices]

        print("\nRetrieved Knowledge Snippets:")
        for i, text in enumerate(retrieved_texts, 1):
            print(f"[{i}] {text}")
        print("------------------------------------------------------------\n")

        return "\n".join(retrieved_texts)

    def chat(self, utterance: str) -> dict:
        """Retrieve relevant context, then generate an answer."""

        # Step 1: Retrieve relevant snippets
        retrieved_context = self.retrieve_context(utterance)

        # Step 2: Construct the prompt and dialogue history
        system_prompt = (
            "You are a friendly and knowledgeable Cambridge student "
            "who helps others learn about university life. "
            "Use the retrieved context below to answer accurately "
            "and naturally. If you don't know, say so politely.\n\n"
            f"--- Retrieved context ---\n{retrieved_context}\n"
            "--- End context ---"
        )

        messages: ResponseInputParam = [
            {"role": "system", "content": system_prompt}
        ]

        for turn in self.conversation_history:
            role: Literal["assistant", "user"] = (
                "assistant" if turn["speaker"] == "assistant" else "user"
            )
            messages.append({"role": role, "content": turn["utterance"]})

        messages.append({"role": "user", "content": utterance})

        # Step 3: Generate a response
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            store=False
        )

        reply_text = response.output_text.strip()

        return {
            "text": reply_text,
            "retrieved_context": retrieved_context
        }

if __name__ == "__main__":
    bot = RAGBot()
    bot.start_a_chat()