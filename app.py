import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from difflib import SequenceMatcher
import re
import time

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="UDOM Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1520;
    color: #c8d0e0;
}
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 6rem;
    max-width: 820px;
    margin: 0 auto;
}

/*  Sidebar  */
[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid #1e2a40 !important;
    min-width: 230px !important;
}
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/*  Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.15rem 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: #1a2540;
    border: 1px solid #2a3a58;
    border-radius: 16px 16px 4px 16px;
    padding: 0.7rem 1rem;
    max-width: 82%;
    margin-left: auto;
    color: #d8e0f0;
    font-size: 0.91rem;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: #111827;
    border: 1px solid #1e2a40;
    border-left: 3px solid #e8c87a;
    border-radius: 4px 16px 16px 16px;
    padding: 0.85rem 1.1rem;
    color: #c8d0e0;
    font-size: 0.91rem;
    line-height: 1.72;
}
[data-testid="chatAvatarIcon-user"]      { background: #2a3a58 !important; }
[data-testid="chatAvatarIcon-assistant"] { background: #1a2030 !important; border: 1px solid #e8c87a44; }

/* Chat input  */
[data-testid="stChatInput"] { max-width: 720px; margin: 0 auto; }
[data-testid="stChatInput"] textarea {
    background: #141e2f !important;
    border: 1px solid #3a4f72 !important;
    border-radius: 14px !important;
    color: #d8e0f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.91rem !important;
    padding: 0.8rem 1rem !important;
    box-shadow: 0 0 0 3px #2a3a5820 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #8a9ab8 !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #e8c87a !important;
    box-shadow: 0 0 0 3px #e8c87a22 !important;
}
[data-testid="stChatInputSubmitButton"] svg { fill: #e8c87a !important; }

/* Sidebar action buttons */
[data-testid="stSidebar"] .stButton button {
    background: #0f1825 !important;
    border: 1px solid #1e2e48 !important;
    border-radius: 8px !important;
    color: #8a9ab8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.79rem !important;
    padding: 0.42rem 0.75rem !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: #e8c87a55 !important;
    color: #e8c87a !important;
    background: #141f30 !important;
}

/*  General buttons (welcome chips etc.)  */
.stButton button {
    background: #111827 !important;
    border: 1px solid #2a3a58 !important;
    border-radius: 10px !important;
    color: #8a9ab8 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 0.6rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
.stButton button:hover { border-color: #e8c87a99 !important; color: #e8c87a !important; background: #1a2540 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: #0a0f1a !important;
    border: 1px solid #1e2a40 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #3a4a68 !important;
    font-size: 0.73rem !important;
    letter-spacing: 0.06em !important;
}

/* Memory badges  */
.mem-badge {
    display: block; background: #0f1825; border: 1px solid #1e2e48;
    border-radius: 6px; padding: 4px 10px; font-size: 0.71rem;
    color: #4a5a78; margin-bottom: 4px; font-family: 'DM Sans', sans-serif;
}
.mem-badge.active { background: #0f1f0f; border-color: #2a4a2a; color: #6ab87a; }




/* Thinking dots */
.thinking-wrap {
    display: flex; align-items: center; gap: 6px; padding: 0.6rem 0.2rem;
}
.thinking-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #e8c87a; opacity: 0.3;
    animation: pulse 1.2s ease-in-out infinite;
}
.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%,100%{opacity:0.25;transform:scale(0.85)}
    50%{opacity:1;transform:scale(1.1)}
}

/*  Timestamp + copy */
.msg-time {
    font-size: 0.65rem; color: #2a3a58;
    font-family: 'DM Sans', sans-serif;
}
.copy-btn {
    display: inline-block; cursor: pointer;
    font-size: 0.68rem; color: #3a4a68;
    padding: 2px 8px; border-radius: 4px;
    border: 1px solid #1e2a40; background: #0a0f1a;
    margin-top: 0; transition: all 0.15s;
    font-family: 'DM Sans', sans-serif;
}
.copy-btn:hover { color: #e8c87a; border-color: #e8c87a44; }

/* Clarify card */
.clarify-card {
    background: #0d1a2e;
    border: 1px solid #2a3a58;
    border-left: 3px solid #e8c87a;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #8a9ab8;
    line-height: 1.6;
}
.clarify-title {
    font-size: 0.78rem;
    color: #e8c87a;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

/* ── Welcome ── */
.welcome-wrap {
    display: flex; flex-direction: column;
    align-items: center; text-align: center;
    padding: 3rem 1rem 1.5rem 1rem;
}
.welcome-logo { font-size: 2.6rem; margin-bottom: 0.9rem; filter: drop-shadow(0 0 18px #e8c87a44); }
.welcome-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem; font-weight: 700; color: #e8c87a;
    margin-bottom: 0.35rem; letter-spacing: -0.01em;
}
.welcome-sub {
    font-size: 0.82rem; color: #4a5a78;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 1.4rem;
}
.welcome-hint {
    font-size: 0.86rem; color: #3a4a68;
    margin-bottom: 1.6rem; line-height: 1.6;
}

/* Mobile  */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
        max-width: 100% !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
        max-width: 96%;
        font-size: 0.88rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
        font-size: 0.88rem;
        padding: 0.7rem 0.85rem;
    }
    .welcome-title { font-size: 1.3rem; }
    .welcome-logo  { font-size: 2rem; }
    .welcome-hint  { font-size: 0.82rem; }
    [data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 240px !important;
    }
    [data-testid="stChatInput"] { max-width: 100%; }
}

/* Scrollbar  */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0f1a; }
::-webkit-scrollbar-thumb { background: #2a3a58; border-radius: 4px; }

.sidebar-label {
    font-size: 0.61rem; color: #2a3a58; text-transform: uppercase;
    letter-spacing: 0.12em; margin: 12px 0 6px 2px;
    font-family: 'DM Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = {
        "college": None, "program": None,
        "year": None, "semester": None, "topic": None
    }
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "msg_times" not in st.session_state:
    st.session_state.msg_times = []
# Stores a list of program names when ambiguity is detected
if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = []
if "clarification_original_query" not in st.session_state:
    st.session_state.clarification_original_query = None
# Rolling summary of old conversation turns (used when history > 6 turns)
if "history_summary" not in st.session_state:
    st.session_state.history_summary = ""


# =====================================================
# MEMORY HELPERS
# =====================================================

def get_memory(key):
    return st.session_state.conversation_memory.get(key)

def update_memory(entities, query_type=None):
    """
    Update memory with new entities.
    - Clears curriculum keys when switching to regulation/bylaw.
    - Clears year/semester (but not college/program) when switching topics.
    """
    if query_type in ("regulation", "bylaw"):
        for key in ("college", "program", "year", "semester"):
            st.session_state.conversation_memory[key] = None

    # If a new college is explicitly detected, clear the old program
    # (user switched to a different college)
    new_college = entities.get("college")
    old_college = get_memory("college")
    if new_college and old_college and new_college.lower() != old_college.lower():
        st.session_state.conversation_memory["program"] = None
        st.session_state.conversation_memory["year"] = None
        st.session_state.conversation_memory["semester"] = None

    # If a new program is explicitly detected, clear old year/semester
    new_program = entities.get("program")
    old_program = get_memory("program")
    if new_program and old_program and new_program.lower() != old_program.lower():
        st.session_state.conversation_memory["year"] = None
        st.session_state.conversation_memory["semester"] = None

    for key, value in entities.items():
        if value:
            st.session_state.conversation_memory[key] = value

def clear_memory_keys(*keys):
    for key in keys:
        st.session_state.conversation_memory[key] = None

def get_last_topic():
    return get_memory("topic")


# =====================================================
# LOAD VECTOR DB
# =====================================================

@st.cache_resource
def load_vector_store():
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        embedding_function=emb,
        persist_directory="./chroma_db",
        collection_name="university_docs"
    )

vector_store = load_vector_store()


# =====================================================
# LOAD CATALOGS
# =====================================================

@st.cache_data
def load_catalogs():
    all_data = vector_store.get()
    programs_set = set()
    colleges_set = set()
    for meta in all_data["metadatas"]:
        prog = meta.get("program")
        college = meta.get("college")
        if prog and isinstance(prog, str):
            programs_set.add(prog.strip())
        if college and isinstance(college, str):
            colleges_set.add(college.strip())
    return list(programs_set), list(colleges_set), all_data

all_programs, all_colleges, all_data = load_catalogs()


# =====================================================
# ENTITY DETECTION
# =====================================================

def detect_college(query):
    q = query.lower()
    best_match = None
    best_score = 0.0
    BLOCKED_COMPOUNDS = {"law": ["bylaw", "bylaws"], "school": ["preschool"]}
    for college in all_colleges:
        c = college.lower()
        if c in q:
            return college
        short_name = re.sub(r"^(college of|school of|institute of)\s*", "", c).strip()
        score = SequenceMatcher(None, q, c).ratio()
        if short_name and len(short_name) >= 5:
            pattern = r"\b" + re.escape(short_name) + r"\b"
            if re.search(pattern, q):
                blocked = BLOCKED_COMPOUNDS.get(short_name, [])
                if not any(b in q for b in blocked):
                    score += 0.4
        if score > best_score:
            best_score = score
            best_match = college
    if best_score >= 0.72:
        return best_match
    return None


PROGRAM_MIN_WORDS = 4

def detect_program(query):
    q = query.lower()
    # Pass 1 — longest exact substring
    exact_matches = []
    for prog in all_programs:
        prog_lower = prog.lower()
        if len(prog_lower.split()) < PROGRAM_MIN_WORDS:
            continue
        if prog_lower in q:
            exact_matches.append(prog)
    if exact_matches:
        return max(exact_matches, key=lambda p: len(p))
    # Pass 2 — fuzzy + short-name
    best_match = None
    best_score = 0.0
    for prog in all_programs:
        prog_lower = prog.lower()
        if len(prog_lower.split()) < PROGRAM_MIN_WORDS:
            continue
        short_name = re.sub(
            r"^(bachelor of science in|bachelor of arts in|bachelor of education in"
            r"|bachelor of|b\.sc\.\s*in|b\.sc\s*in|diploma in|doctor of|doctor of medicine)\s*",
            "", prog_lower
        ).strip()
        short_name = re.sub(r"\(.*?\)", "", short_name).strip()
        score = SequenceMatcher(None, q, prog_lower).ratio()
        if short_name and short_name in q:
            score += 0.5
        if short_name:
            sn_words = short_name.split()
            hits = sum(1 for w in sn_words if w in q and len(w) > 3)
            if sn_words:
                score += (hits / len(sn_words)) * 0.3
        if score > best_score:
            best_score = score
            best_match = prog
    if best_score >= 0.60:
        return best_match
    return None


def detect_ambiguous_programs(query):
    """
    Returns a list of candidate programs when the query is ambiguous —
    i.e. when multiple programs score close to each other.
    Returns [] when the top match is clearly better than the rest.
    """
    q = query.lower()
    scored = []
    for prog in all_programs:
        prog_lower = prog.lower()
        if len(prog_lower.split()) < PROGRAM_MIN_WORDS:
            continue
        short_name = re.sub(
            r"^(bachelor of science in|bachelor of arts in|bachelor of education in"
            r"|bachelor of|diploma in|doctor of)\s*", "", prog_lower
        ).strip()
        short_name = re.sub(r"\(.*?\)", "", short_name).strip()
        score = SequenceMatcher(None, q, prog_lower).ratio()
        if short_name:
            sn_words = short_name.split()
            hits = sum(1 for w in sn_words if w in q and len(w) > 3)
            if sn_words:
                score += (hits / len(sn_words)) * 0.3
        scored.append((score, prog))
    scored.sort(reverse=True)
    top = [(s, p) for s, p in scored[:5] if s >= 0.50]
    if len(top) >= 2 and (top[0][0] - top[1][0]) < 0.07:
        return [p for _, p in top[:4]]
    return []


def extract_entities(query):
    q = query.lower()
    entities = {
        "college": None, "program": None,
        "year": None, "semester": None, "topic": None
    }
    college = detect_college(query)
    if college:
        entities["college"] = college
    program = detect_program(query)
    if program:
        entities["program"] = program
    if any(x in q for x in ["year one", "first year", "1st year"]):
        entities["year"] = "year one"
    elif any(x in q for x in ["year two", "second year", "2nd year"]):
        entities["year"] = "year two"
    elif any(x in q for x in ["year three", "third year", "3rd year"]):
        entities["year"] = "year three"
    elif any(x in q for x in ["year four", "fourth year", "4th year"]):
        entities["year"] = "year four"
    if "semester one" in q or "first semester" in q:
        entities["semester"] = "semester one"
    elif "semester two" in q or "second semester" in q:
        entities["semester"] = "semester two"
    if any(x in q for x in ["gpa", "regulation", "carry", "probation",
                              "classification", "discontinue", "academic standing"]):
        entities["topic"] = "regulation"
    elif any(x in q for x in ["misconduct", "discipline", "penalty", "offence",
                                "bylaw", "bylaws", "accommodation", "cafeteria",
                                "rent fee", "hostel", "rusticate"]):
        entities["topic"] = "bylaw"
    elif any(x in q for x in ["college", "colleges", "school", "schools",
                                "institute", "institutes"]):
        entities["topic"] = "college"
    elif any(x in q for x in ["program", "programs", "degree", "degrees"]):
        entities["topic"] = "program"
    elif any(x in q for x in ["course", "courses", "year one", "year two",
                                "year three", "year four", "first year",
                                "second year", "third year", "fourth year"]):
        entities["topic"] = "course"
    elif "semester" in q and any(x in q for x in ["year", "program", "course"]):
        entities["topic"] = "course"
    return entities


# =====================================================
# QUERY CLASSIFIER
# =====================================================

def classify_query(query):
    q = query.lower()

    # Explain / clarify → inherit memory topic
    if re.search(r"^(explain|clarify|describe|elaborate|what does|what do you mean|tell me more)", q):
        topic = get_last_topic()
        if topic:
            return topic

    # Follow-up phrases
    followup_phrases = [
        "what about", "how about", "those", "them", "the others",
        "others", "only those", "and for", "what of",
        "what are they", "list them", "show them"
    ]
    if any(x in q for x in followup_phrases):
        topic = get_last_topic()
        if topic:
            return topic

    # Pronoun follow-ups — only when memory has context
    if get_memory("program") or get_memory("college"):
        pronoun_signals = [
            r"\bits\s+(course|year|semester|subject|curriculum)",
            r"\btheir\s+(course|year|semester|subject|curriculum)",
            r"\bfor\s+(it|them|this|that)\b",
            r"^(what are|list|show)\s+(its|their|the)\s+",
            r"^(and|also)\s+(what about|list|show)\s+",
        ]
        for pattern in pronoun_signals:
            if re.search(pattern, q):
                topic = get_last_topic()
                if topic:
                    return topic
                if any(x in q for x in ["year", "course", "courses", "semester"]):
                    return "course"
                break

    if re.search(r"\b(hello|hi|hey|helo|howdy|good morning|good evening)\b", q):
        return "greeting"

    if any(x in q for x in [
        "gpa", "carry", "probation", "supplementary",
        "classification", "discontinue", "regulation",
        "academic standing", "credit hour", "grading policy"
    ]):
        return "regulation"

    if any(x in q for x in [
        "misconduct", "discipline", "penalty", "offence",
        "bylaw", "bylaws", "student rule", "student conduct",
        "rusticate", "accommodation", "cafeteria", "rent fee",
        "hostel", "student by-law"
    ]):
        return "bylaw"

    if any(x in q for x in [
        "course", "courses",
        "year one", "year two", "year three", "year four",
        "first year", "second year", "third year", "fourth year"
    ]):
        return "course"

    if "semester" in q and any(x in q for x in [
        "year", "program", "course", "engineering", "science",
        "arts", "education", "commerce", "medicine", "nursing", "law"
    ]):
        return "course"

    if any(x in q for x in ["program", "programs", "degree", "degrees",
                              "offered", "available"]):
        if re.search(r"\b(what|which)\s+(college|school|institute)\b", q):
            return "semantic"
        return "program_list"

    if any(x in q for x in ["college", "colleges", "school", "schools",
                              "institute", "institutes"]):
        if re.search(r"\b(what|which)\s+(college|school|institute)\b", q):
            return "semantic"
        if any(x in q for x in ["program", "programs", "degree", "degrees",
                                  "offered", "available", "under", "in"]):
            return "program_list"
        return "college_list"

    return "semantic"


# =====================================================
# FILTER BUILDER
# =====================================================

def extract_filters(query, query_type):
    q = query.lower()
    entities = extract_entities(query)
    update_memory(entities, query_type=query_type)
    memory_college  = get_memory("college")
    memory_program  = get_memory("program")
    memory_year     = get_memory("year")
    memory_semester = get_memory("semester")
    detected_college  = entities.get("college")  or memory_college
    detected_program  = entities.get("program")  or memory_program
    detected_year     = entities.get("year")     or memory_year
    detected_semester = entities.get("semester") or memory_semester
    if detected_college:
        detected_college = next(
            (c for c in all_colleges if c.lower() == detected_college.lower()),
            detected_college
        )
    if detected_program:
        detected_program = next(
            (p for p in all_programs if p.lower() == detected_program.lower()),
            detected_program
        )
    conditions = []
    if query_type == "course":
        conditions.append({"type": "course"})
    elif query_type == "regulation":
        conditions.append({"type": "regulation"})
        if "gpa" in q:        conditions.append({"topic": "gpa"})
        elif "carry" in q:    conditions.append({"topic": "carry"})
        elif "credit" in q:   conditions.append({"topic": "credit"})
        elif "classification" in q: conditions.append({"topic": "degree_classification"})
        elif "discontinue" in q:    conditions.append({"topic": "discontinuous"})
        return _build_filter(conditions)
    elif query_type == "bylaw":
        conditions.append({"type": "bylaw"})
        return _build_filter(conditions)
    if detected_program:
        conditions.append({"program": detected_program})
        if not detected_college:
            for meta in all_data["metadatas"]:
                if (meta.get("type") == "college_program_map"
                        and meta.get("program", "").lower() == detected_program.lower()):
                    detected_college = meta.get("college")
                    break
    if detected_college:
        conditions.append({"college": detected_college})
    if detected_year and query_type == "course":
        conditions.append({"year": detected_year})
    if detected_semester and query_type == "course":
        conditions.append({"semester": detected_semester})
    return _build_filter(conditions)


def _build_filter(conditions):
    if not conditions:
        return None
    seen_keys = set()
    deduped = []
    for cond in conditions:
        key = list(cond.keys())[0]
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(cond)
    if len(deduped) == 1:
        return deduped[0]
    return {"$and": deduped}


# =====================================================
# HYBRID RERANKER
# =====================================================

def hybrid_rerank(query, docs, query_type):
    memory_program = get_memory("program")
    memory_college = get_memory("college")
    q = query.lower()
    scored = []
    for d in docs:
        score = 0.0
        content = d.page_content.lower()
        meta = d.metadata
        words = [w for w in q.split() if len(w) > 3]
        for word in words:
            if word in content:
                score += 2
        for yr in ["year one", "year two", "year three", "year four"]:
            if yr in q and meta.get("year") == yr:
                score += 15
        for sem in ["semester one", "semester two"]:
            if sem in q and meta.get("semester") == sem:
                score += 10
        college = detect_college(query) or memory_college
        if college:
            doc_college = meta.get("college", "").lower()
            if doc_college:
                sim = SequenceMatcher(None, college.lower(), doc_college).ratio()
                score += sim * 20
        program = detect_program(query) or memory_program
        if program:
            doc_program = meta.get("program", "").lower()
            if doc_program:
                sim = SequenceMatcher(None, program.lower(), doc_program).ratio()
                score += sim * 25
        if "gpa" in q and meta.get("topic") == "gpa":           score += 20
        if "carry" in q and meta.get("topic") == "carry":       score += 20
        if "classification" in q and meta.get("topic") == "degree_classification": score += 20
        scored.append((score, d))
    scored.sort(reverse=True, key=lambda x: x[0])
    seen = set()
    unique_docs = []
    for score, d in scored:
        meta = d.metadata
        if meta.get("type") == "course":
            key = (meta.get("program"), meta.get("year"),
                   meta.get("semester"), meta.get("course_code"))
        else:
            key = (meta.get("type"), meta.get("college"),
                   meta.get("program"), meta.get("section"))
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)
    is_listing = any(x in q for x in ["list", "all", "courses", "programs", "colleges"])
    if is_listing or query_type in ["program_list", "college_list", "course"]:
        return unique_docs[:100]
    return unique_docs[:15]


# =====================================================
# FORMAT DOCS
# =====================================================

def format_docs(docs):
    output = []
    for i, d in enumerate(docs):
        meta = d.metadata
        text = (
            f"DOCUMENT {i+1}\n"
            f"TYPE     : {meta.get('type','N/A')}\n"
            f"COLLEGE  : {meta.get('college','N/A')}\n"
            f"PROGRAM  : {meta.get('program','N/A')}\n"
            f"YEAR     : {meta.get('year','N/A')}\n"
            f"SEMESTER : {meta.get('semester','N/A')}\n"
            f"SECTION  : {meta.get('section','N/A')}\n\n"
            f"CONTENT:\n{d.page_content}"
        )
        output.append(text)
    return "\n\n===================\n\n".join(output)


# =====================================================
# HISTORY BUILDER WITH SUMMARIZATION
# =====================================================

def build_history(max_recent=4):
    """
    Returns conversation context for the LLM.
    - Last max_recent turns are shown verbatim (most important for follow-ups).
    - Older turns are represented by a rolling summary stored in session state.
    - When history grows past max_recent, the oldest turn is compressed into
      the rolling summary using the LLM.
    """
    history = st.session_state.chat_history
    summary = st.session_state.history_summary

    if not history:
        return "No previous conversation."

    recent = history[-max_recent:]
    lines = []

    if summary:
        lines.append(f"[Earlier conversation summary: {summary}]")

    for turn in recent:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['bot']}")

    return "\n".join(lines)


def maybe_summarize_history():
    """
    When chat history exceeds 8 turns, summarize the oldest 4 into
    a rolling summary and drop them. This keeps the prompt compact
    while preserving context.
    """
    history = st.session_state.chat_history
    if len(history) <= 8:
        return

    to_summarize = history[:4]
    st.session_state.chat_history = history[4:]

    turns_text = "\n".join(
        f"User: {t['user']}\nAssistant: {t['bot']}" for t in to_summarize
    )
    existing = st.session_state.history_summary
    prefix = f"Previous summary: {existing}\n\n" if existing else ""

    try:
        summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        result = summary_llm.invoke(
            f"{prefix}Summarize this conversation in 2-3 sentences, "
            f"focusing on what topics, programs, or colleges were discussed:\n\n{turns_text}"
        )
        st.session_state.history_summary = result.content.strip()
    except Exception:
        pass  # Summarization is best-effort


# =====================================================
# DYNAMIC K
# =====================================================

def determine_k(query, query_type):
    q = query.lower()
    if any(x in q for x in ["list", "all", "courses", "programs"]):
        return 80
    if query_type == "course":
        return 50
    return 20


# =====================================================
# LLM
# =====================================================

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

llm = load_llm()







# =====================================================
# HELPERS
# =====================================================

def fmt_time(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M")

def copy_row(content, copy_id, ts=""):
    """
    Render timestamp + copy button.
    Content is stored in a hidden <span data-copy> attribute to avoid
    JS escaping issues with backticks, backslashes, and markdown characters.
    """
    import base64
    # Base64-encode the content so it survives HTML attribute embedding safely
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
        <span class="msg-time">{ts}</span>
        <span class="copy-btn" id="{copy_id}" data-b64="{b64}"
              onclick="(function(){{
                  var b=document.getElementById('{copy_id}').getAttribute('data-b64');
                  navigator.clipboard.writeText(atob(b)).then(function(){{
                      document.getElementById('{copy_id}').innerText='✓ Copied';
                      setTimeout(function(){{document.getElementById('{copy_id}').innerText='⎘ Copy';}},1800);
                  }});
              }})()">⎘ Copy</span>
    </div>""", unsafe_allow_html=True)

def save_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    st.session_state.msg_times.append(time.time())


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div style='padding:0.9rem 0 0.4rem 0;'>
        <div style='font-family:"Playfair Display",serif;font-size:1.05rem;
                    color:#e8c87a;font-weight:700;margin-bottom:3px;'>UDOM Assistant</div>
        <div style='font-size:0.64rem;color:#2a3a58;letter-spacing:0.12em;text-transform:uppercase;'>
            University of Dodoma
        </div>
    </div>
    <hr style='border-color:#1a2540;margin:0.5rem 0 0.7rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Quick Actions</div>', unsafe_allow_html=True)
    QUICK_ACTIONS = [
        ("🏛️ Colleges",  "List all colleges"),
        ("🎓 Programs",   "Show all degree programs"),
        ("📋 GPA Rules",  "What are the GPA regulations?"),
        ("📜 Bylaws",     "Give me an overview of the student bylaws"),
        ("📚 Help",       "hello"),
    ]
    for label, qtext in QUICK_ACTIONS:
        if st.button(label, key=f"qa_{label}"):
            st.session_state.pending_query = qtext
            st.session_state.awaiting_clarification = []
            st.rerun()

    st.markdown('<hr style="border-color:#1a2540;margin:0.7rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Context Memory</div>', unsafe_allow_html=True)

    mem = st.session_state.conversation_memory
    mem_items = {
        "College": mem.get("college"), "Program": mem.get("program"),
        "Year": mem.get("year"), "Semester": mem.get("semester"),
        "Topic": mem.get("topic")
    }
    mem_html = ""
    for label, val in mem_items.items():
        if val:
            display = val.title()
            if len(display) > 28:
                display = display[:26] + "…"
            mem_html += f'<div class="mem-badge active">{label}: {display}</div>'
        else:
            mem_html += f'<div class="mem-badge">{label}: —</div>'
    st.markdown(mem_html, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1a2540;margin:0.7rem 0;">', unsafe_allow_html=True)
    if st.button("🗑  Clear conversation", key="clear_btn"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.msg_times = []
        st.session_state.history_summary = ""
        st.session_state.conversation_memory = {
            "college": None, "program": None,
            "year": None, "semester": None, "topic": None
        }
        st.session_state.pending_query = None
        st.session_state.awaiting_clarification = []
        st.session_state.clarification_original_query = None
        st.rerun()


# =====================================================
# MAIN AREA
# =====================================================

# Header or welcome screen
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-logo">🎓</div>
        <div class="welcome-title">UDOM Academic Assistant</div>
        <div class="welcome-sub">University of Dodoma</div>
        <div class="welcome-hint">
            Ask me anything about your academic journey —<br>
            courses, programs, regulations, bylaws, or which college offers what.
        </div>
    </div>
    """, unsafe_allow_html=True)

    WELCOME_ACTIONS = [
        ("🏛️ Colleges",  "List all colleges"),
        ("🎓 Programs",   "Show all degree programs"),
        ("📋 GPA Rules",  "What are the GPA regulations?"),
        ("📜 Bylaws",     "Give me an overview of the student bylaws"),
    ]
    cols = st.columns(len(WELCOME_ACTIONS))
    for i, (label, qtext) in enumerate(WELCOME_ACTIONS):
        with cols[i]:
            if st.button(label, key=f"wq_{i}"):
                st.session_state.pending_query = qtext
else:
    st.markdown("""
    <div style='margin-bottom:0.8rem;'>
        <span style='font-family:"Playfair Display",serif;font-size:1.05rem;
                     color:#e8c87a;font-weight:700;'>UDOM Assistant</span>
        <span style='font-size:0.7rem;color:#2a3a58;margin-left:10px;
                     letter-spacing:0.1em;text-transform:uppercase;'>
            University of Dodoma</span>
    </div>
    """, unsafe_allow_html=True)

#Replay full chat history

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            ts = fmt_time(st.session_state.msg_times[idx]) if idx < len(st.session_state.msg_times) else ""
            copy_row(msg["content"], f"copy_{idx}", ts)

#Clarification UI 
if st.session_state.awaiting_clarification:
    original = st.session_state.clarification_original_query or "your question"
    st.markdown(f"""
    <div class="clarify-card">
        <div class="clarify-title">🔍 Which program did you mean?</div>
        I found a few programs that could match <em>"{original}"</em>.
        Please pick one and I'll answer right away:
    </div>
    """, unsafe_allow_html=True)
    clarify_cols = st.columns(len(st.session_state.awaiting_clarification))
    for i, prog in enumerate(st.session_state.awaiting_clarification):
        with clarify_cols[i]:
            label = prog.title()
            if len(label) > 38:
                label = label[:36] + "…"
            if st.button(label, key=f"clarify_{i}"):
                resolved = f"{st.session_state.clarification_original_query} for {prog}"
                update_memory({"program": prog})
                st.session_state.awaiting_clarification = []
                st.session_state.clarification_original_query = None
                st.session_state.pending_query = resolved

#Chat input 
chat_input = st.chat_input("Ask about courses, programs, regulations…")
_pending = st.session_state.pending_query
if _pending:
    st.session_state.pending_query = None
query = _pending or chat_input


# =====================================================
# PROCESS QUERY
# =====================================================

def run_query(query):

    save_message("user", query)
    with st.chat_message("user"):
        st.markdown(query)

    query_type = classify_query(query)

    #Greeting 
    if query_type == "greeting":
        answer = (
            "Hey there 👋 I'm your UDOM Academic Assistant.\n\n"
            "You can ask me about:\n"
            "- 🏛️ Colleges and Schools\n"
            "- 🎓 Degree Programs\n"
            "- 📚 Courses and Curriculum\n"
            "- 📋 GPA & Academic Regulations\n"
            "- 📜 Student Bylaws\n\n"
            "Try the quick-action buttons on the left, or just type your question!"
        )
        save_message("assistant", answer)
        with st.chat_message("assistant"):
            st.markdown(answer)
        return

    # Ambiguity check 
    if query_type == "course" and not get_memory("program"):
        candidates = detect_ambiguous_programs(query)
        if len(candidates) >= 2:
            st.session_state.awaiting_clarification = candidates
            st.session_state.clarification_original_query = query
            _amsg = ("I want to make sure I get this right — your question could relate "
                     "to a few different programs. Which one are you asking about?")
            save_message("assistant", _amsg)
            with st.chat_message("assistant"):
                st.markdown(_amsg)
            return

    # College list shortcut 
    if query_type == "college_list":
        results = vector_store.get(where={"type": "college"})
        college_names = sorted(set(
            m.get("college") for m in results["metadatas"] if m.get("college")
        ))
        answer = "## 🏛️ Colleges & Schools\n\n"
        for c in college_names:
            answer += f"- {c.title()}\n"
        st.session_state.conversation_memory["topic"] = "college"
        save_message("assistant", answer)
        with st.chat_message("assistant"):
            st.markdown(answer)
        return

    # Program list shortcut 
    if query_type == "program_list":
        entities = extract_entities(query)
        detected_college = entities.get("college") or get_memory("college")
        if detected_college:
            detected_college = next(
                (c for c in all_colleges if c.lower() == detected_college.lower()),
                detected_college
            )
            results = vector_store.get(where={
                "$and": [{"type": "college_program_map"}, {"college": detected_college}]
            })
            prog_names = sorted(set(
                m.get("program") for m in results["metadatas"] if m.get("program")
            ))
            answer = f"## 🎓 Programs under {detected_college.title()}\n\n"
            answer += "\n".join(f"- {p.title()}" for p in prog_names) if prog_names else "_No programs found._"
        else:
            results = vector_store.get(where={"type": "college_program_map"})
            grouped = {}
            for m in results["metadatas"]:
                col = m.get("college", "Unknown")
                prog = m.get("program")
                if prog:
                    grouped.setdefault(col, set()).add(prog)
            answer = "## 🎓 All Degree Programs\n\n"
            for col, progs in sorted(grouped.items()):
                answer += f"\n**{col.title()}**\n\n"
                for p in sorted(progs):
                    answer += f"- {p.title()}\n"
        update_memory({"topic": "program", "college": detected_college})
        save_message("assistant", answer)
        with st.chat_message("assistant"):
            st.markdown(answer)
        return

    #Retrieval + LLM 
    entities_now = extract_entities(query)
    update_memory(entities_now, query_type=query_type)
    search_filter = extract_filters(query, query_type)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": determine_k(query, query_type),
            "fetch_k": 200,
            "lambda_mult": 0.7,
            **({"filter": search_filter} if search_filter else {})
        }
    )
    docs = retriever.invoke(query)

    # Fallback: relax filter if nothing returned
    if len(docs) == 0 and search_filter:
        detected_college = get_memory("college")
        fallback_filter = {"type": "course"} if query_type == "course" else None
        if fallback_filter and detected_college:
            fallback_filter = {"$and": [{"type": "course"}, {"college": detected_college}]}
        if fallback_filter:
            docs = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": determine_k(query, query_type),
                    "fetch_k": 200, "lambda_mult": 0.7,
                    "filter": fallback_filter
                }
            ).invoke(query)

    docs = hybrid_rerank(query, docs, query_type)
    context = format_docs(docs)
    history = build_history(max_recent=4)

    prompt = f"""You are a friendly, knowledgeable academic assistant for the University of Dodoma (UDOM).

PERSONALITY: Warm, helpful, and conversational. Talk like a knowledgeable senior student helping a friend —
not like a formal document. Use natural language, not robotic phrases like "Based on the provided context...".

STRICT RULES:
1. Only use information from the Context below. Never invent course names, program names, or policies.
2. If the answer is genuinely not in the Context, say: "I couldn't find that in the university documents. Try rephrasing or ask me something else!"
3. For listing questions (courses, programs): list ALL relevant items with bullet points. Include course codes.
4. For detail questions: give a clear, direct answer. One or two sentences if possible.
5. For follow-up questions: use Chat History to understand what was being discussed, then answer from Context.
6. Never repeat the question. Never start with "Certainly!" or "Of course!".
7. When listing courses, include: course code, course title, Core/Elective. Skip any line without a course code.
8. If the question is vague, answer for the most likely match and mention which program you answered for.

--- CHAT HISTORY ---
{history}

--- CONTEXT ---
{context}

--- QUESTION ---
{query}

Answer:"""

    with st.chat_message("assistant"):
        # Thinking indicator
        thinking_ph = st.empty()
        thinking_ph.markdown("""
        <div class="thinking-wrap">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
        </div>""", unsafe_allow_html=True)

        placeholder = st.empty()
        full_answer = ""
        first_chunk = True
        for chunk in llm.stream(prompt):
            if first_chunk:
                thinking_ph.empty()
                first_chunk = False
            full_answer += chunk.content
            placeholder.markdown(full_answer + "▌")
        placeholder.markdown(full_answer)
        copy_row(full_answer, f"copy_ans_{len(st.session_state.messages)}", fmt_time(time.time()))

    # Save to state AFTER streaming completes
    st.session_state.chat_history.append({"user": query, "bot": full_answer})
    save_message("assistant", full_answer)
    maybe_summarize_history()


if query:
    run_query(query)