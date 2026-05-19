from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def create_rag_prompt(context, question):

    prompt = f"""
You are an intelligent AI assistant.

Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def load_embedding_model():

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


def create_text_splitter():

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    return splitter