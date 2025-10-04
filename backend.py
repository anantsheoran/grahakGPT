# backend.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

# -----------------------------
# Load knowledge base
# -----------------------------
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # keep chunks small enough for flan-t5-base
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)

# -----------------------------
# Embeddings (fast & accurate)
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# -----------------------------
# Vector Store (persistent & cached)
# -----------------------------
persist_dir = "db"

if os.path.exists(persist_dir):
    vector_store = Chroma(
        collection_name="consumer_protection_kb",
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
else:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="consumer_protection_kb",
        persist_directory=persist_dir
    )

# -----------------------------
# Retriever for RAG
# -----------------------------
# 🔁 Changed from k=1 → k=3 for richer context
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Prompt Template
# -----------------------------
prompt_template = """
You are a helpful assistant that answers questions based on the provided context.
Only use the information from the context. If the answer is not contained in the context, say you cannot answer.

Context:
{context}

Question: {question}

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -----------------------------
# Free Local LLM (flan-t5-base)
# -----------------------------
# 🔁 Upgraded from t5-small → flan-t5-base
# 🔁 Added better generation parameters
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device_map="auto",   # uses MPS on Apple Silicon if available
    max_new_tokens=512,  # allow longer answers
    temperature=0.7,     # balance factual vs natural
    top_p=0.9            # nucleus sampling for diversity
)
llm = HuggingFacePipeline(pipeline=generator)

# -----------------------------
# RAG Chain
# -----------------------------
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# -----------------------------
# Function to answer queries
# -----------------------------
def get_answer(query: str) -> str:
    """Answer user queries using the RAG pipeline."""
    return rag_chain.invoke(query)

# -----------------------------
# Optional: test CLI
# -----------------------------
if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is a consumer dispute?"
    print(f"\n🔎 Query: {query}\n")

    # Show top retrieved docs (now 3 instead of 1)
    retrieved_docs = vector_store.similarity_search(query, k=3)
    print("📄 Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n--- Doc {i} ---\n{doc.page_content[:500]}...\n")

    # Generate final answer
    answer = get_answer(query)
    print("💡 Answer:\n")
    print(answer)
