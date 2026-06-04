import os, time, re, uuid, tempfile, requests, chromadb, streamlit as st
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import bs4, google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_mic_recorder import mic_recorder

# 1. SYSTEM INITIALIZATION & CORE CONFIGS
# 1. SYSTEM INITIALIZATION & CORE CONFIGS
load_dotenv()

# Streamlit Cloud passes secrets via st.secrets, while local uses os.getenv
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("API Key missing! Please configure GOOGLE_API_KEY in your settings.")
    st.stop()

CHROMA_PATH, EXPORT_DIR = "./chroma_db", "saved_chats"
for d in [EXPORT_DIR, CHROMA_PATH]: Path(d).mkdir(parents=True, exist_ok=True)

# FIX: Switched to the live supported model identifiers
LLM_MODEL = "gemini-1.5-flash"
EMBED_MODEL = "models/gemini-embedding-001" 
FALLBACK_ERROR = "I could not find that information in the uploaded context."

# Initialize the generative model instance once globally
AI_MODEL_INSTANCE = genai.GenerativeModel(LLM_MODEL)

st.set_page_config(page_title="AI Research Engine", layout="wide")
st.title("⚡ Optimized Voice-Enabled Knowledge Engine")

for key, val in [("answer_cache", {}), ("db_chunk_count", 0), ("voice_text", "")]:
    if key not in st.session_state: 
        st.session_state[key] = val

@st.cache_resource
def get_vector_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=chromadb.Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(
        name="unified_rag_store", metadata={"hnsw:space": "cosine"}
    )
    st.session_state.db_chunk_count = collection.count()
    return collection

collection_instance = get_vector_collection()


def embed_io(texts, task="retrieval_document"):
    # If it's a single string, wrap it in a list to normalize processing
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = []
    batch_size = 30  # Safe batch size to stay well under token-per-minute (TPM) limits
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        retries = 3
        delay = 2.0
        
        while retries > 0:
            try:
                # CRITICAL LEGACY SDK FIX: 
                # For a list of multiple items, we must use contents=batch.
                # For a single item, we use content=batch[0].
                if len(batch) > 1:
                    response = genai.embed_content(
                        model=EMBED_MODEL,
                        contents=batch,  # Notice the plural 'contents'
                        task_type=task
                    )
                else:
                    response = genai.embed_content(
                        model=EMBED_MODEL,
                        content=batch[0], # Notice the singular 'content'
                        task_type=task
                    )
                
                # Extract the vectors based on response structure
                if "embedding" in response:
                    # If it's a batch response, it returns a list of dicts/lists
                    if isinstance(response["embedding"], list) and len(response["embedding"]) > 0:
                        # Check if it's a nested list of embeddings or a single vector
                        if isinstance(response["embedding"][0], (list, float, int)):
                            # If it's a single string wrapped in a list, wrap the response to extend cleanly
                            if len(batch) == 1 and not isinstance(response["embedding"][0], list):
                                embeddings.append(response["embedding"])
                            else:
                                embeddings.extend(response["embedding"])
                        else:
                            embeddings.append(response["embedding"])
                
                # Introduce a solid baseline pause between batches to let the RPM quota rest
                time.sleep(1.0)
                break
                
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    retries -= 1
                    if retries == 0:
                        st.error("🚨 Gemini API Free Tier Quota Exhausted. Please wait 60 seconds and click Build again.")
                        raise e
                    import random
                    time.sleep(delay + random.uniform(0, 1))
                    delay *= 2
                else:
                    raise e
                    
    return embeddings
# Helper function to convert text to speech using the HTML5 Web Speech API
def text_to_speech_autoplay(text_content):
    """Injects a clean browser-native JavaScript snippet to instantly read out answers."""
    clean_text = text_content.replace('"', '\\"').replace('\n', ' ')
    tts_script = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{clean_text}");
        window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(tts_script, height=0, width=0)

# 2. DOCUMENT PROCESSING & CRAWLER LAYERS
def load_single_pdf(file_bytes, file_name):
    pages = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        for idx, page in enumerate(PdfReader(tmp_path).pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append((f"[Page {idx+1}]\n{text.strip()}", {"source_name": f"📄 PDF: {file_name}", "page": idx+1}))
        return pages
    finally:
        if os.path.exists(tmp_path): 
            os.remove(tmp_path)

def crawl_site_recursive(start_url, max_depth=2):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    visited, queue, extracted = set(), [(start_url, 0)], []
    
    start_parsed = urlparse(start_url)
    start_domain = start_parsed.netloc.replace("www.", "")
    base_prefix = f"{start_parsed.scheme}://{start_parsed.netloc}"
    
    while queue and len(visited) < 25:
        url, depth = queue.pop(0)
        norm_url = urlparse(url)._replace(fragment="").geturl().rstrip('/')
        if norm_url in visited or depth > max_depth: 
            continue
        visited.add(norm_url)
        
        try:
            res = requests.get(norm_url, timeout=5, headers=headers, verify=False)
            if res.status_code != 200 or "text/html" not in res.headers.get("Content-Type", ""): 
                continue
            
            soup = bs4.BeautifulSoup(res.content, "html.parser")
            for tag in soup(["script", "style", "noscript", "nav", "footer"]): 
                tag.decompose()
            cleaned_text = "\n".join([p.strip() for p in soup.get_text().splitlines() if p.strip()])
            
            if cleaned_text: 
                extracted.append((cleaned_text, {"source_name": f"🕸️ Crawled: {norm_url}", "page": "Web"}))
            
            if depth < max_depth:
                for a in soup.find_all('a', href=True):
                    href = a['href'].strip()
                    if href.startswith(('javascript:', 'mailto:', 'tel:')) or href.startswith('#'): 
                        continue
                    
                    child = href if urlparse(href).netloc else f"{base_prefix.rstrip('/')}/{href.lstrip('/')}"
                    child_domain = urlparse(child).netloc.replace("www.", "")
                    
                    if start_domain in child_domain and child not in visited:
                        queue.append((child, depth + 1))
        except: 
            continue
    return extracted

# 3. CACHE INTERACTION UTILITIES
def get_file_path(question_text):
    slug = re.sub(r'[^a-zA-Z0-9_]', '', question_text.strip().lower().replace(" ", "_"))[:40]
    return os.path.join(EXPORT_DIR, f"cached_{slug}.txt")

def check_disk_cache(question_text):
    path = get_file_path(question_text)
    if os.path.exists(path):
        try:
            parts = Path(path).read_text(encoding="utf-8").split("--- ANSWER ---\n")
            ans_part, src_part = parts[1].split("\n--- SOURCES ---")
            return {"answer": ans_part.strip(), "sources": [s.strip("- ") for s in src_part.strip().splitlines() if s]}
        except: 
            return None
    return None

# 4. STREAMLIT WORKFLOW EXECUTION
with st.sidebar:
    st.subheader("Ingestion Control Panel")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    url_input = st.text_input("Crawl Website URL:", placeholder="https://example.com")
    process_button = st.button("Build Knowledge Base", type="primary")

if process_button and (uploaded_files or url_input.strip()):
    status = st.sidebar.empty()
    progress = st.sidebar.progress(10)
    docs_pool = []
    
    if uploaded_files:
        status.write("🚀 Processing PDFs via Background Parallel Workers...")
        with ThreadPoolExecutor(max_workers=4) as exe:
            futures = [exe.submit(load_single_pdf, f.getvalue(), f.name) for f in uploaded_files]
            for f in as_completed(futures): 
                docs_pool.extend(f.result())
            
    if url_input.strip():
        status.write("🕸️ Executing Network Target Domain Discoveries...")
        try: 
            docs_pool.extend(crawl_site_recursive(url_input.strip()))
        except Exception as e: 
            st.sidebar.error(f"Crawler failure: {e}")
        
    if docs_pool:
        status.write("⚡ Chunking & Injecting Vector Embeddings...")
        progress.progress(60)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        
        final_chunks, final_metas, final_ids = [], [], []
        for text, meta in docs_pool:
            for i, split_txt in enumerate(splitter.split_text(text)):
                final_chunks.append(split_txt)
                final_metas.append({**meta, "chunk_idx": i})
                final_ids.append(f"chk_{uuid.uuid4().hex[:8]}")
                
        collection_instance.add(ids=final_ids, embeddings=embed_io(final_chunks), documents=final_chunks, metadatas=final_metas)
        st.session_state.db_chunk_count = collection_instance.count()
        st.session_state.answer_cache.clear()
        
        progress.progress(100)
        status.write("🎉 **Knowledge Base Live!**")

# 5. USER INTERFACE GENERATION LOGIC WITH AUDIO MODULES
st.subheader("Interact with Knowledge Base")

audio_col, clear_col = st.columns([1, 4])
with audio_col:
    st.write("🎙️ Voice Input:")
    audio_data = mic_recorder(start_prompt="Record Question", stop_prompt="Stop & Process", key="mic")

if audio_data and 'bytes' in audio_data:
    with st.spinner("Translating speech to text..."):
        try:
            voice_file_bytes = audio_data['bytes']
            audio_part = {"mime_type": "audio/wav", "data": voice_file_bytes}
            st.session_state.voice_text = AI_MODEL_INSTANCE.generate_content(
                ["Examine this audio clip and type exactly what was spoken in plain text without any introductory commentary.", audio_part]
            ).text.strip()
        except Exception as e:
            st.error(f"Voice Recognition Error: {e}")

default_query = st.session_state.voice_text if st.session_state.voice_text else ""
question = st.text_input("Ask a question (or use the voice recorder button above):", value=default_query)

if question:
    st.session_state.voice_text = ""
    cache_key = f"{question.strip().lower()}_{st.session_state.db_chunk_count}"
    start_time = time.perf_counter()
    
    if cache_key in st.session_state.answer_cache and st.session_state.answer_cache[cache_key]["answer"] != FALLBACK_ERROR:
        res = st.session_state.answer_cache[cache_key]
        st.write(res["answer"])
        st.caption(f"⏱️ RAM Cache Lookup: {time.perf_counter() - start_time:.4f}s")
        for s in res["sources"]: 
            st.markdown(f"- {s}")
        text_to_speech_autoplay(res["answer"])
    else:
        d_cache = check_disk_cache(question)
        if d_cache and d_cache["answer"] != FALLBACK_ERROR:
            st.write(d_cache["answer"])
            st.caption(f"⏱️ Disk Cache Lookup: {time.perf_counter() - start_time:.4f}s")
            for s in d_cache["sources"]: 
                st.markdown(f"- {s}")
            st.session_state.answer_cache[cache_key] = d_cache
            text_to_speech_autoplay(d_cache["answer"])
        elif st.session_state.db_chunk_count == 0:
            st.warning("⚠️ Vector Base is empty. Build via sidebar first.")
        else:
            # FIX: Unified query lookup using the corrected custom embed_io layout
            q_emb = embed_io(question, task="retrieval_query")[0]
            hits = collection_instance.query(query_embeddings=[q_emb], n_results=3, include=["documents", "metadatas"])
            
            if not hits["documents"] or not hits["documents"][0]:
                st.write(FALLBACK_ERROR)
            else:
                ctx = "\n\n".join([f"[Source: {m['source_name']}]\n{d}" for d, m in zip(hits["documents"][0], hits["metadatas"][0])])
                prompt = f"Concisely answer query solely using context. End completely. No markdown blocks.\nIf missing, reply exactly with: {FALLBACK_ERROR}\n\nContext:\n{ctx}\n\nQuery:\n{question}"
                
                ans = st.write_stream(
                    chunk.text for chunk in AI_MODEL_INSTANCE.generate_content(prompt, stream=True) if chunk.text
                )
                
                srcs = {f"{m['source_name']}" + (f" (Page {m['page']})" if m['page'] != "Web" else "") for m in hits["metadatas"][0]}
                st.caption(f"⏱️ Vector Execution Time: {time.perf_counter() - start_time:.3f}s")
                
                st.subheader("Sources Used")
                for s in srcs: 
                    st.markdown(f"- {s}")
                
                st.session_state.answer_cache[cache_key] = {"answer": ans, "sources": list(srcs)}
                Path(get_file_path(question)).write_text(f"QUESTION:\n{question}\n--- ANSWER ---\n{ans}\n--- SOURCES ---\n" + "\n".join([f"- {s}" for s in srcs]), encoding="utf-8")
                
                text_to_speech_autoplay(ans)