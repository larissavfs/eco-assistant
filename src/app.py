"""
app.py
======
Interface Streamlit para o sistema RAG.
Execute com: streamlit run app.py
"""

import os
import time
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

    :root {
        --bg-primary: #0d0f14;
        --bg-card: #141720;
        --bg-input: #1a1e2a;
        --accent: #6ee7b7;
        --accent-2: #818cf8;
        --text-primary: #e2e8f0;
        --text-muted: #64748b;
        --border: #1e2535;
        --user-bubble: #1e2d3d;
        --ai-bubble: #141c2b;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
    }

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Hide streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }

    /* ── Header ── */
    .rag-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 1.2rem 1.6rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        margin-bottom: 1.5rem;
    }
    .rag-logo {
        font-size: 2rem;
        filter: drop-shadow(0 0 12px #6ee7b766);
    }
    .rag-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.5px;
        margin: 0;
    }
    .rag-subtitle {
        font-size: 0.78rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        margin: 0;
    }

    /* ── Chat Messages ── */
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 0.5rem 0;
    }
    .msg-user, .msg-ai {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        animation: fadeUp 0.3s ease;
    }
    .msg-user { flex-direction: row-reverse; }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .avatar {
        width: 36px; height: 36px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    .avatar-user { background: var(--accent-2); }
    .avatar-ai   { background: linear-gradient(135deg, var(--accent), var(--accent-2)); }
    .bubble {
        max-width: 78%;
        padding: 0.9rem 1.1rem;
        border-radius: 14px;
        font-size: 0.93rem;
        line-height: 1.65;
        border: 1px solid var(--border);
    }
    .bubble-user { background: var(--user-bubble); border-radius: 14px 4px 14px 14px; }
    .bubble-ai   { background: var(--ai-bubble);   border-radius: 4px 14px 14px 14px; }
    .bubble-time {
        font-size: 0.68rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        margin-top: 6px;
    }

    /* ── Sources ── */
    .sources-box {
        margin-top: 0.6rem;
        padding: 0.7rem 1rem;
        background: #0f1420;
        border: 1px solid #1e2d3d;
        border-radius: 10px;
        font-size: 0.8rem;
        color: var(--text-muted);
    }
    .sources-title {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .source-item {
        padding: 4px 0;
        border-bottom: 1px solid #1a2030;
        font-size: 0.78rem;
    }
    .source-item:last-child { border-bottom: none; }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    .status-ok    { background: #063b2f; color: var(--success); border: 1px solid #065f46; }
    .status-warn  { background: #3b2906; color: var(--warning); border: 1px solid #92400e; }
    .status-error { background: #3b0606; color: var(--error);   border: 1px solid #991b1b; }

    /* ── Metrics ── */
    .metric-grid { display: flex; gap: 10px; margin: 1rem 0; }
    .metric-card {
        flex: 1;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--accent);
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label { font-size: 0.7rem; color: var(--text-muted); margin-top: 2px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
    }
    .sidebar-section {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-title {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.6rem;
    }

    /* Streamlit widget overrides */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
        color: #0d0f14 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
    .stSlider > div { color: var(--text-primary) !important; }
    .stSelectbox > div > div {
        background: var(--bg-input) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stFileUploader"] {
        background: var(--bg-input) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 10px !important;
    }
    .stSpinner > div { border-top-color: var(--accent) !important; }
    div[data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* Chat container scroll */
    .chat-scroll {
        max-height: 520px;
        overflow-y: auto;
        padding: 1rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        margin-bottom: 1rem;
    }

    /* Welcome message */
    .welcome-box {
        text-align: center;
        padding: 3rem 2rem;
        color: var(--text-muted);
    }
    .welcome-icon { font-size: 3.5rem; margin-bottom: 1rem; }
    .welcome-text { font-size: 0.95rem; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "chain": None,
        "vectorstore": None,
        "indexed": False,
        "index_stats": {},
        "temperature": 0.2,
        "show_sources": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
Path(KNOWLEDGE_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_chain(temperature: float, _reset_key: str):
    """Inicializa e caches a chain RAG."""
    from rag_engine import ingest_knowledge_base, build_rag_chain
    result = ingest_knowledge_base(KNOWLEDGE_DIR, reset=(_reset_key == "reset"))
    if result["status"] == "empty":
        return None, result
    chain = build_rag_chain(result["vectorstore"], temperature=temperature)
    return chain, result


def check_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").startswith("sk-"))


def list_knowledge_files() -> list:
    p = Path(KNOWLEDGE_DIR)
    return sorted([f for f in p.glob("*") if f.suffix in (".txt", ".pdf")])


def format_timestamp() -> str:
    return time.strftime("%H:%M")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:2.2rem">🧠</div>
        <div style="font-size:1rem; font-weight:700; color:#6ee7b7; letter-spacing:-0.5px;">RAG Assistant</div>
        <div style="font-size:0.7rem; color:#64748b; font-family:'JetBrains Mono',monospace;">LangChain + ChromaDB + Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

    # Status API
    api_ok = check_api_key()
    if api_ok:
        st.markdown('<div class="status-badge status-ok">● API conectada</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-error">● API key não configurada</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem; color:#64748b; margin-top:8px; line-height:1.6;">
        Crie um arquivo <code>.env</code> com sua <code>OPENAI_API_KEY</code>.
        Veja <code>.env.example</code>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Upload de documentos
    st.markdown('<div class="sidebar-title">📁 Base de Conhecimento</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Envie arquivos (.txt ou .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        for f in uploaded:
            dest = Path(KNOWLEDGE_DIR) / f.name
            dest.write_bytes(f.read())
        st.success(f"✅ {len(uploaded)} arquivo(s) adicionado(s)")

    # Lista de arquivos
    files = list_knowledge_files()
    if files:
        with st.expander(f"📄 {len(files)} arquivo(s) na base", expanded=False):
            for f in files:
                size = f.stat().st_size / 1024
                st.markdown(
                    f'<div style="font-size:0.78rem; padding:3px 0; color:#94a3b8;">'
                    f'{"📄" if f.suffix == ".pdf" else "📝"} {f.name} '
                    f'<span style="color:#475569">({size:.1f} KB)</span></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div style="font-size:0.78rem; color:#475569; margin:8px 0;">'
            'Nenhum arquivo na base ainda. Envie ou use o arquivo de exemplo.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Configurações
    st.markdown('<div class="sidebar-title">⚙️ Configurações</div>', unsafe_allow_html=True)

    temperature = st.slider(
        "Temperatura (criatividade)",
        min_value=0.0, max_value=1.0,
        value=st.session_state.temperature,
        step=0.05,
        help="0 = respostas precisas | 1 = mais criativo"
    )
    st.session_state.temperature = temperature

    show_sources = st.toggle("Mostrar fontes", value=st.session_state.show_sources)
    st.session_state.show_sources = show_sources

    st.markdown("---")

    # Botões de ação
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Indexar", use_container_width=True):
            get_chain.clear()
            st.session_state.indexed = False
            st.session_state.chain = None
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Limpar chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chain = None
            st.session_state.indexed = False
            st.rerun()

    # Stats
    if st.session_state.indexed and st.session_state.index_stats:
        stats = st.session_state.index_stats
        st.markdown("---")
        st.markdown('<div class="sidebar-title">📊 Índice</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.8rem; color:#94a3b8; line-height:2;">
        📂 Documentos: <b style="color:#6ee7b7">{stats.get("docs", 0)}</b><br>
        🔪 Chunks: <b style="color:#6ee7b7">{stats.get("chunks", 0)}</b><br>
        🤖 Modelo: <b style="color:#6ee7b7">{os.getenv("OPENAI_MODEL", "gpt-4o-mini")}</b>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <div class="rag-logo">🧠</div>
    <div>
        <div class="rag-title">RAG Assistant</div>
        <div class="rag-subtitle">Retrieval-Augmented Generation · LangChain + ChromaDB + Streamlit</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INICIALIZAR CHAIN
# ─────────────────────────────────────────────
if not st.session_state.indexed and api_ok and list_knowledge_files():
    with st.spinner("⚙️ Indexando base de conhecimento..."):
        try:
            reset_key = "normal"
            chain, stats = get_chain(temperature, reset_key)
            if chain:
                st.session_state.chain = chain
                st.session_state.indexed = True
                st.session_state.index_stats = stats
            else:
                st.warning("⚠️ Nenhum documento encontrado. Envie arquivos na sidebar.")
        except Exception as e:
            st.error(f"❌ Erro ao indexar: {e}")


# ─────────────────────────────────────────────
# CHAT — EXIBIÇÃO
# ─────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-box">
            <div class="welcome-icon">💬</div>
            <div class="welcome-text">
                Sua base de conhecimento está pronta.<br>
                Faça uma pergunta para começar.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            ts = msg.get("time", "")

            if role == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="avatar avatar-user">👤</div>
                    <div>
                        <div class="bubble bubble-user">{content}</div>
                        <div class="bubble-time" style="text-align:right">{ts}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = ""
                if st.session_state.show_sources and msg.get("sources"):
                    items = "".join(
                        f'<div class="source-item">📎 {s["file"]} — <span style="color:#94a3b8">{s["preview"]}</span></div>'
                        for s in msg["sources"]
                    )
                    sources_html = f"""
                    <div class="sources-box">
                        <div class="sources-title">Fontes utilizadas</div>
                        {items}
                    </div>
                    """

                st.markdown(f"""
                <div class="msg-ai">
                    <div class="avatar avatar-ai">🧠</div>
                    <div style="max-width:78%">
                        <div class="bubble bubble-ai">{content}</div>
                        <div class="bubble-time">{ts}</div>
                        {sources_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT — CHAT
# ─────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([6, 1])
    with cols[0]:
        user_input = st.text_input(
            "Mensagem",
            placeholder="Digite sua pergunta sobre a base de conhecimento...",
            label_visibility="collapsed",
        )
    with cols[1]:
        submitted = st.form_submit_button("Enviar →", use_container_width=True)

# Sugestões de perguntas
if not st.session_state.messages:
    st.markdown("""
    <div style="margin-top: 0.5rem;">
    <span style="font-size:0.75rem; color:#475569; font-family:'JetBrains Mono',monospace;">Sugestões:</span>
    </div>
    """, unsafe_allow_html=True)

    sugs = [
        "Quais produtos vocês oferecem?",
        "Como funciona o suporte?",
        "Qual é a política de cancelamento?",
        "Quais são os preços dos planos?",
    ]
    cols = st.columns(len(sugs))
    for i, sug in enumerate(sugs):
        with cols[i]:
            if st.button(sug, use_container_width=True, key=f"sug_{i}"):
                user_input = sug
                submitted = True


# ─────────────────────────────────────────────
# PROCESSAR PERGUNTA
# ─────────────────────────────────────────────
if submitted and user_input.strip():
    # Verificações
    if not api_ok:
        st.error("❌ Configure sua OPENAI_API_KEY no arquivo .env")
        st.stop()

    if not list_knowledge_files():
        st.warning("⚠️ Adicione documentos na base de conhecimento antes de perguntar.")
        st.stop()

    if not st.session_state.chain:
        st.warning("⚠️ Base ainda não indexada. Clique em 'Indexar' na sidebar.")
        st.stop()

    # Adiciona mensagem do usuário
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "time": format_timestamp(),
    })

    # Gera resposta
    with st.spinner("🔍 Buscando contexto e gerando resposta..."):
        try:
            t0 = time.time()
            response = st.session_state.chain.invoke({"question": user_input})
            elapsed = time.time() - t0

            answer = response.get("answer", "Não consegui gerar uma resposta.")
            source_docs = response.get("source_documents", [])

            sources = []
            seen = set()
            for doc in source_docs:
                file_name = doc.metadata.get("source_file", "desconhecida")
                preview = doc.page_content[:90].replace("\n", " ").strip() + "..."
                key = file_name + preview[:30]
                if key not in seen:
                    seen.add(key)
                    sources.append({"file": file_name, "preview": preview})

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "time": f"{format_timestamp()} · {elapsed:.1f}s",
            })

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Erro ao processar sua pergunta: {str(e)}",
                "sources": [],
                "time": format_timestamp(),
            })

    st.rerun()


<<<<<<< HEAD
# ─────────────────────────────────────────────
# ESTADO — SEM API / SEM ARQUIVOS
# ─────────────────────────────────────────────
=======
>>>>>>> 5e9c592 (feat: implement v1)
if not api_ok:
    st.info("""
    **Para começar:**
    1. Copie `.env.example` para `.env`
    2. Adicione sua `OPENAI_API_KEY`
    3. Reinicie a aplicação com `streamlit run app.py`
    """)

elif not list_knowledge_files():
    st.info("""
    **Base de conhecimento vazia!**
    Adicione arquivos `.txt` ou `.pdf` na sidebar (ou na pasta `knowledge_base/`).
    Depois clique em **Indexar**.
    """)