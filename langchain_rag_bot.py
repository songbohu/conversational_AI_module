from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import os, json, sys


class LangChainRAGBot:

    def __init__(self,
                 model_name="gpt-4o-mini",
                 embedding_model="text-embedding-3-small",
                 key_path="openai.key",
                 kb_path="cambridge_knowledge_list.json"):
        self._load_openai_key(key_path)

        if not os.path.exists(kb_path):
            sys.exit(f"Error: The knowledge base '{kb_path}' was not found.")
        with open(kb_path, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
        texts = [doc["text"] for doc in knowledge_base]
        print(f"Knowledge base loaded with {len(texts)} entries.\n")

        embeddings = OpenAIEmbeddings(model=embedding_model)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents(texts)
        self.vectorstore = Chroma.from_documents(docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        self.llm = ChatOpenAI(model=model_name)
        self.prompt = PromptTemplate.from_template(
            "You are a helpful Cambridge student. "
            "Answer the question using the context below. "
            "If unsure, say you don’t know.\n\n"
            "Conversation History:\n{history}\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}"
        )

        self.history = []

        print("LangChain RAGBot initialised successfully.\n")

    def _load_openai_key(self, key_path: str):
        if not os.path.exists(key_path):
            sys.exit(f"Error: The API key file '{key_path}' was not found.")
        with open(key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
        print("OpenAI API key loaded successfully.\n")

    def chat(self, user_input: str):
        docs = self.retriever.get_relevant_documents(user_input)
        context = "\n".join([d.page_content for d in docs])

        history_text = ""
        for user, assistant in self.history[-3:]:
            history_text += f"User: {user}\nAssistant: {assistant}\n"
        history_text += f"User: {user_input}\n"

        final_prompt = self.prompt.format(context=context, history=history_text, question=user_input)

        response = self.llm.invoke(final_prompt)
        reply = response.content.strip()

        print(f"\nAssistant: {reply}\n")

        self.history.append((user_input, reply))

    def start(self):
        print("Welcome to LangChain RAGBot! Type 'bye' or 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "bye"}:
                print("Goodbye!")
                break
            self.chat(user_input)


if __name__ == "__main__":
    bot = LangChainRAGBot()
    bot.start()
