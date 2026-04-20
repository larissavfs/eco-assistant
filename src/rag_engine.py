"""
rag_engine.py
=============
Motor principal do RAG: ingestão, indexação, retrieval e geração de respostas.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

load_dotenv()


# ─────────────────────────────────────────────
# 1. EMBEDDINGS
# ─────────────────────────────────────────────
def get_embeddings():
    """Retorna o modelo de embeddings configurado no .env."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = os.getenv(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Padrão: OpenAI
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )


# ─────────────────────────────────────────────
# 2. LLM
# ─────────────────────────────────────────────
def get_llm(temperature: float = 0.2):
    """Retorna o modelo de linguagem configurado."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        streaming=True,
    )


# ─────────────────────────────────────────────
# 3. CARREGAMENTO DE DOCUMENTOS
# ─────────────────────────────────────────────
def load_documents(knowledge_dir: str) -> list[Document]:
    """
    Carrega todos os documentos da pasta de conhecimento.
    Suporta: .txt, .pdf
    """
    docs = []
    path = Path(knowledge_dir)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return docs

    loaders = {
        "**/*.txt": TextLoader,
        "**/*.pdf": PyPDFLoader,
    }

    for glob_pattern, LoaderClass in loaders.items():
        for file_path in path.glob(glob_pattern):
            try:
                if LoaderClass == TextLoader:
                    loader = LoaderClass(str(file_path), encoding="utf-8")
                else:
                    loader = LoaderClass(str(file_path))
                file_docs = loader.load()
                # Adiciona metadados de origem
                for doc in file_docs:
                    doc.metadata["source_file"] = file_path.name
                docs.extend(file_docs)
            except Exception as e:
                print(f"[AVISO] Erro ao carregar {file_path.name}: {e}")

    return docs


# ─────────────────────────────────────────────
# 4. CHUNKING
# ─────────────────────────────────────────────
def split_documents(docs: list[Document]) -> list[Document]:
    """Divide documentos em chunks para indexação."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    return chunks


# ─────────────────────────────────────────────
# 5. BANCO VETORIAL (ChromaDB)
# ─────────────────────────────────────────────
def get_vectorstore(chunks: Optional[list[Document]] = None, reset: bool = False):
    """
    Cria ou carrega o banco vetorial ChromaDB.
    Se chunks for fornecido, indexa os documentos.
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection = os.getenv("CHROMA_COLLECTION", "rag_knowledge_base")
    embeddings = get_embeddings()

    if reset and Path(persist_dir).exists():
        import shutil
        shutil.rmtree(persist_dir)

    if chunks:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=collection,
        )
    else:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection,
        )

    return vectorstore


# ─────────────────────────────────────────────
# 6. RETRIEVER
# ─────────────────────────────────────────────
def get_retriever(vectorstore):
    """Configura o retriever com busca por similaridade."""
    k = int(os.getenv("RETRIEVER_K", 4))
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ─────────────────────────────────────────────
# 7. PROMPT PERSONALIZADO
# ─────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""Você é um assistente especialista que responde perguntas com base exclusivamente no contexto fornecido.

REGRAS:
- Responda APENAS com informações do contexto abaixo
- Se a informação não estiver no contexto, diga claramente: "Não encontrei essa informação na base de conhecimento."
- Seja objetivo, claro e preciso
- Responda em português do Brasil
- Cite a fonte quando relevante (nome do arquivo)

HISTÓRICO DA CONVERSA:
{chat_history}

CONTEXTO RECUPERADO:
{context}

PERGUNTA: {question}

RESPOSTA:""",
)


# ─────────────────────────────────────────────
# 8. CHAIN CONVERSACIONAL
# ─────────────────────────────────────────────
def build_rag_chain(vectorstore, temperature: float = 0.2):
    """Monta a chain RAG com memória conversacional."""
    llm = get_llm(temperature=temperature)
    retriever = get_retriever(vectorstore)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
        verbose=False,
    )

    return chain


# ─────────────────────────────────────────────
# 9. INGESTÃO COMPLETA
# ─────────────────────────────────────────────
def ingest_knowledge_base(knowledge_dir: str, reset: bool = False) -> dict:
    """
    Pipeline completo de ingestão:
    1. Carrega documentos
    2. Divide em chunks
    3. Indexa no ChromaDB
    Retorna estatísticas do processo.
    """
    docs = load_documents(knowledge_dir)
    if not docs:
        return {"status": "empty", "docs": 0, "chunks": 0}

    chunks = split_documents(docs)
    vectorstore = get_vectorstore(chunks=chunks, reset=reset)

    return {
        "status": "success",
        "docs": len(docs),
        "chunks": len(chunks),
        "vectorstore": vectorstore,
    }


# ─────────────────────────────────────────────
# 10. TESTE RÁPIDO (modo CLI)
# ─────────────────────────────────────────────
def quick_test(question: str, knowledge_dir: str = "./knowledge_base"):
    """Executa um teste rápido via linha de comando."""
    print(f"\n{'='*60}")
    print("🔍 RAG Engine - Teste Rápido")
    print(f"{'='*60}\n")

    print("📂 Carregando e indexando documentos...")
    result = ingest_knowledge_base(knowledge_dir)

    if result["status"] == "empty":
        print("❌ Nenhum documento encontrado na base de conhecimento.")
        return

    print(f"✅ {result['docs']} documento(s) | {result['chunks']} chunks indexados\n")

    chain = build_rag_chain(result["vectorstore"])

    print(f"❓ Pergunta: {question}\n")
    response = chain.invoke({"question": question})

    print(f"🤖 Resposta:\n{response['answer']}\n")
    print("📎 Fontes utilizadas:")
    for doc in response.get("source_documents", []):
        source = doc.metadata.get("source_file", "desconhecida")
        print(f"  • {source}: {doc.page_content[:100]}...")


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Quais produtos vocês oferecem?"
    quick_test(q)