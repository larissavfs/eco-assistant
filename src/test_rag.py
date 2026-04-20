"""
test_rag.py
===========
Testes e validação do pipeline RAG.
Execute com: python test_rag.py
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Cores para terminal ───
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def header(title: str):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")


def ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str):
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str):
    print(f"  {YELLOW}⚠{RESET} {msg}")


def info(msg: str):
    print(f"  {CYAN}ℹ{RESET} {msg}")


# ─────────────────────────────────────────────
# TESTES
# ─────────────────────────────────────────────

def test_imports():
    header("1. Verificando Dependências")
    packages = [
        ("langchain", "LangChain"),
        ("langchain_community", "LangChain Community"),
        ("langchain_openai", "LangChain OpenAI"),
        ("chromadb", "ChromaDB"),
        ("streamlit", "Streamlit"),
        ("dotenv", "Python-dotenv"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    all_ok = True
    for module, name in packages:
        try:
            __import__(module)
            ok(f"{name}")
        except ImportError:
            fail(f"{name} não instalado → pip install {module.replace('_', '-')}")
            all_ok = False
    return all_ok


def test_env():
    header("2. Verificando Configurações (.env)")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key.startswith("sk-"):
        ok(f"OPENAI_API_KEY configurada (***{api_key[-6:]})")
    else:
        fail("OPENAI_API_KEY não configurada ou inválida")
        info("Copie .env.example para .env e adicione sua chave")
        return False

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ok(f"Modelo: {model}")

    embed = os.getenv("EMBEDDING_PROVIDER", "openai")
    ok(f"Embeddings: {embed}")

    chunk_size = os.getenv("CHUNK_SIZE", "1000")
    chunk_overlap = os.getenv("CHUNK_OVERLAP", "200")
    ok(f"Chunk size: {chunk_size} | Overlap: {chunk_overlap}")

    retriever_k = os.getenv("RETRIEVER_K", "4")
    ok(f"Retriever K: {retriever_k} (chunks por consulta)")

    return True


def test_knowledge_base():
    header("3. Verificando Base de Conhecimento")
    kb_dir = Path(os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base"))

    if not kb_dir.exists():
        warn(f"Pasta '{kb_dir}' não encontrada. Criando...")
        kb_dir.mkdir(parents=True)
        fail("Base de conhecimento vazia. Adicione arquivos .txt ou .pdf.")
        return False

    files = list(kb_dir.glob("*.txt")) + list(kb_dir.glob("*.pdf"))
    if not files:
        fail("Nenhum arquivo encontrado na base de conhecimento")
        info(f"Coloque arquivos .txt ou .pdf em: {kb_dir.resolve()}")
        return False

    for f in files:
        size = f.stat().st_size / 1024
        ok(f"{f.name} ({size:.1f} KB)")

    info(f"Total: {len(files)} arquivo(s) encontrado(s)")
    return True


def test_embeddings():
    header("4. Testando Embeddings")
    try:
        from rag_engine import get_embeddings
        info("Carregando modelo de embeddings...")
        t0 = time.time()
        embeddings = get_embeddings()
        test_text = "Teste de embedding do sistema RAG"
        result = embeddings.embed_query(test_text)
        elapsed = time.time() - t0
        ok(f"Embedding gerado em {elapsed:.2f}s")
        ok(f"Dimensão do vetor: {len(result)}")
        return True
    except Exception as e:
        fail(f"Erro nos embeddings: {e}")
        return False


def test_document_loading():
    header("5. Testando Carregamento de Documentos")
    try:
        from rag_engine import load_documents, split_documents
        kb_dir = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")

        info("Carregando documentos...")
        docs = load_documents(kb_dir)
        if not docs:
            warn("Nenhum documento carregado")
            return False
        ok(f"{len(docs)} documento(s) carregado(s)")

        info("Dividindo em chunks...")
        chunks = split_documents(docs)
        ok(f"{len(chunks)} chunks gerados")

        # Mostra exemplo
        if chunks:
            sample = chunks[0]
            preview = sample.page_content[:120].replace("\n", " ") + "..."
            info(f"Exemplo de chunk:\n    '{preview}'")
            info(f"Metadados: {sample.metadata}")

        return True
    except Exception as e:
        fail(f"Erro no carregamento: {e}")
        return False


def test_vectorstore():
    header("6. Testando Banco Vetorial (ChromaDB)")
    try:
        from rag_engine import load_documents, split_documents, get_vectorstore

        kb_dir = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
        docs = load_documents(kb_dir)
        if not docs:
            warn("Sem documentos para indexar")
            return False

        chunks = split_documents(docs)

        info("Indexando no ChromaDB...")
        t0 = time.time()
        vs = get_vectorstore(chunks=chunks, reset=True)
        elapsed = time.time() - t0
        ok(f"Indexado em {elapsed:.2f}s")

        info("Testando busca por similaridade...")
        results = vs.similarity_search("produtos e serviços", k=2)
        ok(f"{len(results)} resultado(s) encontrado(s)")
        for i, r in enumerate(results, 1):
            preview = r.page_content[:80].replace("\n", " ")
            info(f"  [{i}] {preview}...")

        return True, vs
    except Exception as e:
        fail(f"Erro no banco vetorial: {e}")
        return False, None


def test_rag_chain(vectorstore):
    header("7. Testando Chain RAG Completa")

    test_questions = [
        "Quais produtos vocês oferecem?",
        "Como funciona o suporte?",
        "Qual é o preço do plano básico?",
    ]

    try:
        from rag_engine import build_rag_chain
        chain = build_rag_chain(vectorstore, temperature=0.0)
        ok("Chain RAG inicializada com sucesso")

        for question in test_questions:
            info(f"\n  Pergunta: '{question}'")
            t0 = time.time()
            try:
                response = chain.invoke({"question": question})
                elapsed = time.time() - t0
                answer = response.get("answer", "")
                sources = response.get("source_documents", [])

                preview = answer[:120].replace("\n", " ") + ("..." if len(answer) > 120 else "")
                ok(f"Resposta ({elapsed:.2f}s): {preview}")
                ok(f"Fontes: {len(sources)} chunk(s) recuperado(s)")
            except Exception as e:
                fail(f"Erro na pergunta '{question}': {e}")

        return True
    except Exception as e:
        fail(f"Erro na chain: {e}")
        return False


def test_memory():
    header("8. Testando Memória Conversacional")
    try:
        from rag_engine import load_documents, split_documents, get_vectorstore, build_rag_chain

        kb_dir = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
        docs = load_documents(kb_dir)
        chunks = split_documents(docs)
        vs = get_vectorstore(chunks=chunks, reset=False)
        chain = build_rag_chain(vs)

        conversation = [
            "Quais são os planos disponíveis?",
            "E o plano mais barato, o que inclui?",
            "Tem suporte 24h nesse plano?",
        ]

        info("Simulando conversa multi-turno:")
        for i, q in enumerate(conversation, 1):
            print(f"\n  [{i}] 👤 {q}")
            response = chain.invoke({"question": q})
            answer = response.get("answer", "")[:100].replace("\n", " ")
            print(f"      🤖 {answer}...")

        ok("Memória conversacional funcionando!")
        return True
    except Exception as e:
        fail(f"Erro na memória: {e}")
        return False


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────
def run_all_tests():
    print(f"\n{BOLD}{'='*60}")
    print("  🧪 RAG Pipeline — Suite de Testes Completa")
    print(f"{'='*60}{RESET}")

    results = {}

    # Testes que não dependem da API
    results["imports"] = test_imports()
    results["env"] = test_env()
    results["knowledge_base"] = test_knowledge_base()

    # Testes que dependem da API
    if results["env"] and results["knowledge_base"]:
        results["embeddings"] = test_embeddings()
        results["documents"] = test_document_loading()

        if results.get("embeddings") and results.get("documents"):
            vs_result = test_vectorstore()
            results["vectorstore"] = vs_result[0] if isinstance(vs_result, tuple) else vs_result
            vectorstore = vs_result[1] if isinstance(vs_result, tuple) else None

            if results["vectorstore"] and vectorstore:
                results["rag_chain"] = test_rag_chain(vectorstore)
                results["memory"] = test_memory()

    # Sumário
    header("Sumário dos Testes")
    passed = sum(1 for v in results.values() if v is True)
    total  = len(results)

    for test, result in results.items():
        if result is True:
            ok(f"{test}")
        else:
            fail(f"{test}")

    print(f"\n  {'='*40}")
    color = GREEN if passed == total else (YELLOW if passed > total // 2 else RED)
    print(f"  {color}{BOLD}Resultado: {passed}/{total} testes passaram{RESET}")

    if passed == total:
        print(f"\n  {GREEN}{BOLD}🎉 Sistema RAG pronto para uso!{RESET}")
        print(f"  {CYAN}Execute: streamlit run app.py{RESET}\n")
    else:
        print(f"\n  {YELLOW}Corrija os erros acima antes de usar o sistema.{RESET}\n")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)