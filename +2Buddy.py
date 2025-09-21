# cbse_study_app_plus.py — CBSE +2 Study Helper (Topics + Formulae + Past-paper Mode)
# pip install streamlit google-generativeai gTTS pandas pdfplumber pypdf pillow

import io, os, math, datetime as dt
from textwrap import dedent

import streamlit as st
import pandas as pd
import google.generativeai as genai

import pdfplumber
from pypdf import PdfReader
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="BoardReady XII", page_icon="🎓", layout="wide")
st.title("🎓 +2 Buddy")
st.caption("BoardReady XII is a Streamlit-based learning assistant for CBSE Class 12. It converts any chosen topic into a structured study pack—clear definitions, analogies, step-by-step explanations, text-diagrams, and formula sheets—so students build understanding before memorising. A dedicated Translate & Audio mode renders the same content in major Indian languages with natural speech, widening access for multilingual learners. Ask a Teacher simulates past-paper practice with guided hints and model answers. The interface is simple: pick a subject, choose a topic (or type your own), and generate focused, exam-aligned notes and practice.")

# ------------------------- API KEY -------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
 # put your key here for local runs
if not API_KEY:
    API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if not API_KEY:
    st.warning("Add your Gemini API key (hardcode here or via .streamlit/secrets.toml).")
genai.configure(api_key=API_KEY)

def llm(model_id: str):
    return genai.GenerativeModel(model_id)

# -------------------- SUBJECTS & TOPICS --------------------
SUBJECTS = ["English", "Mathematics", "Physics", "Chemistry", "Botany", "Zoology"]

# Quick, commonly-taught slices of CBSE +2 topics (you can expand anytime)
TOPICS = {
    "English": [
        "Reading Comprehension", "Note Making & Summary", "Invitation & Replies",
        "Article Writing", "Report Writing", "Debate", "Speech"
    ],
    "Mathematics": [
        "Relations & Functions", "Inverse Trigonometric Functions", "Matrices",
        "Determinants", "Continuity & Differentiability", "Application of Derivatives",
        "Integrals", "Differential Equations", "Vectors", "Three-Dimensional Geometry",
        "Linear Programming", "Probability"
    ],
    "Physics": [
        "Electrostatics", "Current Electricity", "Magnetism", "EM Induction",
        "Alternating Current", "Optics", "Dual Nature of Radiation and Matter",
        "Atoms and Nuclei", "Semiconductors & Electronics", "Communication Systems"
    ],
    "Chemistry": [
        "Solutions", "Electrochemistry", "Chemical Kinetics", "Surface Chemistry",
        "d- and f-Block Elements", "Coordination Compounds",
        "Haloalkanes & Haloarenes", "Alcohols Phenols and Ethers",
        "Aldehydes Ketones and Carboxylic Acids", "Amines", "Biomolecules", "Polymers"
    ],
    "Botany": [
        "Reproduction in Organisms", "Sexual Reproduction in Flowering Plants",
        "Principles of Inheritance and Variation", "Molecular Basis of Inheritance",
        "Evolution", "Human Health and Disease", "Microbes in Human Welfare",
        "Biotechnology: Principles and Processes", "Biotechnology and its Applications"
    ],
    "Zoology": [
        "Human Reproduction", "Reproductive Health", "Human Physiology—Neural Control",
        "Human Physiology—Chemical Coordination", "Immune System Basics",
        "Evolutionary Biology", "Strategies for Enhancement in Food Production",
        "Organisms and Populations", "Ecosystem"
    ],
}

# ----------------------- SESSION STATE -----------------------
if "translated" not in st.session_state:
    st.session_state.translated = ""
if "concept_text" not in st.session_state:
    st.session_state.concept_text = {"definition": "", "analogy": "", "formulae": ""}
if "concept_keywords" not in st.session_state:
    st.session_state.concept_keywords = []

# --------------------------- HELPERS ---------------------------
def concept_prompt(subject: str, topic: str, need_formulae: bool) -> str:
    # Ask for definition, analogy, keywords AND LaTeX formulae when relevant
    formula_clause = (
        "4) A list of 3–6 KEY FORMULAE in LaTeX (one per line, like: $$F=ma$$) relevant to the topic."
        if need_formulae else
        "4) If formulae are not central (e.g., English), skip formulae."
    )
    return dedent(f"""
        You are an expert CBSE +2 {subject} teacher. For the topic "{topic}", write:

        1) A crisp, exam-ready DEFINITION (2–4 sentences).
        2) A friendly ANALOGY to build intuition (2–3 sentences).
        3) A bullet list of 4–6 KEY TERMS/COMPONENTS for a concept map diagram (1–2 words each).
        {formula_clause}

        Format strictly:
        Definition:
        <text>

        Analogy:
        <text>

        Keywords:
        - <word>
        - <word>
        - <word>
        - <word>
        - <word>
        - <word>

        Formulae:
        $$...$$
        $$...$$
        $$...$$
    """).strip()

def parse_concept_response(text: str):
    definition, analogy, keywords, formulae = "", "", [], []
    sec = None
    for line in text.splitlines():
        l = line.strip()
        tag = l.lower()
        if tag.startswith("definition:"):
            sec = "def"; continue
        if tag.startswith("analogy:"):
            sec = "ana"; continue
        if tag.startswith("keywords:"):
            sec = "keys"; continue
        if tag.startswith("formulae:"):
            sec = "for"; continue
        if sec == "def":
            definition += (l + " ")
        elif sec == "ana":
            analogy += (l + " ")
        elif sec == "keys":
            if l.startswith("-"):
                kw = l.lstrip("-").strip()
                if kw: keywords.append(kw)
        elif sec == "for":
            if "$$" in l:
                # tolerate lines like $$E=mc^2$$ or text $$...$$ text
                start = l.find("$$"); end = l.rfind("$$")
                if end > start: formulae.append(l[start+2:end])
    return definition.strip(), analogy.strip(), keywords, formulae

def draw_concept_map(topic: str, keywords: list[str], size=(1000, 600)) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("arial.ttf", 36)
        font_kw = ImageFont.truetype("arial.ttf", 24)
    except:
        font_title = ImageFont.load_default()
        font_kw = ImageFont.load_default()

    cx, cy, r = W // 2, H // 2, 120
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#E7F0FF", outline="#2D6CDF", width=3)
    tw, th = draw.textbbox((0, 0), topic, font=font_title)[2:]
    draw.text((cx - tw/2, cy - th/2), topic, fill="#0B2A66", font=font_title)

    n = max(1, min(len(keywords), 8))
    radius = 230
    for i in range(n):
        angle = 2 * math.pi * i / n
        kx = int(cx + radius * math.cos(angle))
        ky = int(cy + radius * math.sin(angle))
        rw, rh = 150, 60
        box = (kx - rw//2, ky - rh//2, kx + rw//2, ky + rh//2)
        draw.rounded_rectangle(box, radius=20, fill="#FFF7E6", outline="#F59E0B", width=3)
        draw.line((cx, cy, kx, ky), fill="#999", width=2)
        kw = keywords[i][:28]
        tw, th = draw.textbbox((0, 0), kw, font=font_kw)[2:]
        draw.text((kx - tw/2, ky - th/2), kw, fill="#7C2D12", font=font_kw)
    return img

# --------------------- TRANSLATION (unchanged core) ---------------------
INDIC_LANGS = ["Hindi", "Tamil", "Telugu", "Malayalam", "Kannada", "Marathi",
               "Gujarati", "Bengali", "Punjabi", "Urdu", "Odia", "English"]
GTTs_MAP = {"English":"en","Hindi":"hi","Tamil":"ta","Telugu":"te","Malayalam":"ml",
            "Kannada":"kn","Marathi":"mr","Gujarati":"gu","Bengali":"bn","Punjabi":"pa",
            "Urdu":"ur","Odia":"or"}

def build_translation_prompt(text: str, language: str) -> str:
    return dedent(f"""
        Translate the following CBSE +2 study material into **{language}**.
        Preserve formulae and terminology. Return only the translation.
        Text:
        ---
        {text}
        ---
    """).strip()

def translate_with_gemini(text: str, model_id: str, target_lang: str) -> str:
    model = llm(model_id)
    resp = model.generate_content(build_translation_prompt(text, target_lang))
    return (getattr(resp, "text", None) or "").strip() or "No translation was returned."

# --------------- File Parsing for Translation Tab -----------------
def read_txt(file) -> str: return file.read().decode("utf-8", errors="ignore")
def read_csv(file) -> str: return pd.read_csv(file).to_string(index=False)
def read_xlsx(file) -> str: return pd.read_excel(file).to_string(index=False)
def read_pdf(file) -> str:
    text = []
    try:
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages: text.append(p.extract_text() or "")
        t = "\n".join(text).strip()
        if t: return t
    except Exception: pass
    file.seek(0)
    pdf = PdfReader(file)
    for p in pdf.pages: text.append(p.extract_text() or "")
    return "\n".join(text).strip()

def extract_text_from_upload(upload) -> str:
    n = upload.name.lower()
    if n.endswith(".txt"): return read_txt(upload)
    if n.endswith(".csv"): return read_csv(upload)
    if n.endswith(".xlsx") or n.endswith(".xls"): return read_xlsx(upload)
    if n.endswith(".pdf"): return read_pdf(upload)
    raise ValueError("Unsupported file type. Upload PDF, TXT, CSV, or XLSX.")

# ==================================================================
#                                   TABS
# ==================================================================
tab1, tab2, tab3 = st.tabs(["📘 Concept Builder", "🌐 Translate & Audio", "❓ Ask a Teacher"])

# ============================== TAB 1 ==============================
with tab1:
    st.subheader("Build Understanding")

    # Subject + Topic selector
    colA, colB = st.columns([1, 1])
    with colA:
        subject = st.selectbox("Subject", SUBJECTS, index=0, key="sub1")
    with colB:
        options = TOPICS.get(subject, [])
        topic_choice = st.selectbox("Pick a topic", options + ["Other (type below)"], index=0)
    custom_topic = st.text_input("If Other, type your topic")
    topic = custom_topic.strip() if "Other" in topic_choice else topic_choice

    # Formulae relevant for STEM subjects
    needs_formulae = subject in {"Mathematics", "Physics", "Chemistry"}

    go = st.button("Generate Definition, Analogy, Diagram & Formulae", type="primary")
    if go:
        if not topic:
            st.error("Please pick or type a topic.")
        elif not API_KEY:
            st.error("Missing API key.")
        else:
            with st.spinner("Asking your CBSE teacher LLM…"):
                try:
                    resp = llm("gemini-1.5-pro").generate_content(
                        concept_prompt(subject, topic, needs_formulae)
                    )
                    text = getattr(resp, "text", "") or ""
                    definition, analogy, keywords, formulae = parse_concept_response(text)
                    st.session_state.concept_text = {
                        "definition": definition or "—",
                        "analogy": analogy or "—",
                        "formulae": "\n".join(formulae) if formulae else ""
                    }
                    st.session_state.concept_keywords = keywords[:6] or [subject, "Basics", "Examples", "Formulae"]
                except Exception as e:
                    st.error(f"LLM error: {e}")

    # Display
    if st.session_state.concept_text["definition"]:
        st.markdown("#### Definition")
        st.write(st.session_state.concept_text["definition"])
    if st.session_state.concept_text["analogy"]:
        st.markdown("#### Analogy")
        st.write(st.session_state.concept_text["analogy"])

    if st.session_state.concept_keywords:
        st.markdown("#### Concept Diagram")
        try:
            img = draw_concept_map(topic or subject, st.session_state.concept_keywords)
            buf = io.BytesIO(); img.save(buf, format="PNG"); data = buf.getvalue()
            st.image(data, caption="Auto-generated concept map", use_column_width=True)
            st.download_button("Download Diagram (PNG)", data=data,
                               file_name=f"concept_{(topic or subject).replace(' ', '_')}.png",
                               mime="image/png")
        except Exception as e:
            st.info(f"Could not render diagram ({e}).")

    # Formulae block (render LaTeX)
    if st.session_state.concept_text["formulae"]:
        st.markdown("#### Key Formulae (LaTeX)")
        for line in st.session_state.concept_text["formulae"].splitlines():
            l = line.strip()
            if not l: continue
            # Render with st.latex; wrap with \displaystyle for readability
            st.latex(r"\displaystyle " + l)

# ============================== TAB 2 ==============================
with tab2:
    st.subheader("Translate study material & download MP3")
    target_lang = st.selectbox("Target language", INDIC_LANGS, index=0)

    src_text = st.text_area("Paste text (or) upload a document below",
                            height=180,
                            placeholder="Paste notes, answers, explanations…")
    uploaded = st.file_uploader("Upload document", type=["pdf", "txt", "csv", "xlsx"])

    if uploaded:
        try:
            with st.spinner("Extracting text from document…"):
                doc_text = extract_text_from_upload(uploaded)
            if doc_text:
                src_text = (src_text.rstrip()+"\n\n"+doc_text) if src_text.strip() else doc_text
                st.success("Document text added.")
        except Exception as e:
            st.error(f"Could not extract text: {e}")

    c1, c2 = st.columns(2)
    with c1:
        translate_btn = st.button("Translate", type="primary", use_container_width=True)
    with c2:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.translated = ""
        st.experimental_rerun()

    if translate_btn:
        if not API_KEY:
            st.error("Missing API key.")
        elif not src_text.strip():
            st.error("Please paste text or upload a document first.")
        else:
            with st.spinner("Translating…"):
                try:
                    out = translate_with_gemini(src_text, "gemini-1.5-flash", target_lang)
                    st.session_state.translated = out
                except Exception as e:
                    st.error(f"Gemini API error: {e}")

    if st.session_state.translated:
        st.markdown("#### Translated text")
        st.write(st.session_state.translated)
        fname_txt = f"translation_{target_lang.replace(' ', '_')}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        st.download_button("Download translation (.txt)",
                           data=st.session_state.translated.encode("utf-8"),
                           file_name=fname_txt, mime="text/plain")

    st.markdown("#### Convert to speech (gTTS)")
    code = GTTs_MAP.get(target_lang)
    tts_enabled = bool(code) and bool(st.session_state.translated)
    if not code:
        st.info("gTTS voice may not be available for this language.")
    if st.button("Generate Speech (gTTS)", disabled=not tts_enabled, type="secondary"):
        with st.spinner("Synthesizing speech…"):
            try:
                tts = gTTS(text=st.session_state.translated, lang=code)
                buf = io.BytesIO(); tts.write_to_fp(buf); mp3_bytes = buf.getvalue()
                st.audio(mp3_bytes, format="audio/mp3")
                fname_mp3 = f"speech_{target_lang.replace(' ', '_')}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.mp3"
                st.download_button("Download MP3", data=mp3_bytes, file_name=fname_mp3, mime="audio/mpeg")
            except Exception as e:
                st.error(f"gTTS error: {e}")

# ============================== TAB 3 ==============================
with tab3:
    st.subheader("Ask a Teacher (Past-paper Mode)")
    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        subject2 = st.selectbox("Subject", SUBJECTS, index=0, key="sub3")
    with col2:
        mark_weight = st.selectbox("Answer length (marks)", [2, 3, 5, 10], index=2)
    with col3:
        style = st.selectbox("Style", ["Exam-focused", "Step-by-step tutorial"], index=0)

    q = st.text_area("Your question (e.g., 'State and prove Gauss’s law', 'Derive rate law for first order reaction')",
                     height=140)
    ask = st.button("Get Answer", type="primary")

    if ask:
        if not q.strip():
            st.error("Please type a question.")
        elif not API_KEY:
            st.error("Missing API key.")
        else:
            rubric = {
                2: "2-mark: brief definition/statement + one key point/example.",
                3: "3-mark: short explanation with 2–3 steps or points; include one formula/diagram cue if relevant.",
                5: "5-mark: structured answer with headings, derivation/steps (4–5 points), and a concise example.",
                10:"10-mark: full derivation/explanation with subheadings, labelled diagram if relevant, and a concluding summary."
            }[mark_weight]

            style_clause = "Concise, bullet points allowed." if style=="Exam-focused" else "Teach step-by-step, then condense into key points."

            prompt = dedent(f"""
                You are a CBSE +2 {subject2} teacher. Answer the student's question below in {mark_weight}-mark depth.

                Marking guidance: {rubric}
                Style: {style_clause}
                Must include (when relevant): definitions, formulae (LaTeX), labelled steps, and one short example.
                End with:
                - Quick Recap (2–4 bullets)
                - ONE practice question of the same mark weight.

                Question: "{q}"
            """).strip()

            with st.spinner("Thinking like a teacher…"):
                try:
                    resp = llm("gemini-1.5-pro").generate_content(prompt)
                    ans = getattr(resp, "text", "") or "No answer returned."
                    st.markdown(ans)
                except Exception as e:
                    st.error(f"LLM error: {e}")

st.caption("Made for CBSE +2 learners • Expand TOPICS dict to mirror your exact syllabus.")

