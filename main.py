from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import re
import os
import shutil
from difflib import SequenceMatcher


# =====================================
# LOAD PDF FILES
# =====================================

loader = DirectoryLoader(
    "./data",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
raw_documents = loader.load()

type_counts = {}

for doc in raw_documents:
    source = doc.metadata.get("source", "").lower()

    if "curriculum" in source:
        doc.metadata["type"] = "course"
    elif "student" in source:
        doc.metadata["type"] = "bylaw"
    elif "regulation" in source:
        doc.metadata["type"] = "regulation"
    else:
        doc.metadata["type"] = "general"

    t = doc.metadata["type"]
    type_counts[t] = type_counts.get(t, 0) + 1

print("\nSUMMARY:")
print(type_counts)


# =====================================
# ENTITY REGISTRIES
# =====================================

colleges = {}
programs = {}
courses = {}


# =====================================
# NORMALIZATION UTILITIES
# =====================================

def normalize_college(name):
    name = name.lower()
    name = name.replace("&", "and")
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def normalize_program(name):
    name = name.lower()

    # Abbreviation expansions
    replacements = {
        "bsc.": "bachelor of science",
        "b.sc": "bachelor of science",
        "ba ": "bachelor of arts ",
        "bed ": "bachelor of education ",
        "&": "and"
    }
    for old, new in replacements.items():
        name = name.replace(old, new)

    # Collapse OCR-split words e.g. "educ ation" -> "education"
    name = re.sub(r"\b([a-z]{2,})\s([a-z]{2,})\b", lambda m:
        m.group(0) if len(m.group(1)) + len(m.group(2)) > 12
        else m.group(1) + m.group(2), name)

    # Normalize plural endings so variants resolve to the same key
    name = re.sub(r"\bsciences\b", "science", name)
    name = re.sub(r"\bstudies\b", "study", name)
    name = re.sub(r"\bengineerings\b", "engineering", name)
    name = re.sub(r"\bforensics\b", "forensic", name)

    # Remove abbreviations in brackets
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip()

    # Word-order normalization for the subject portion only.
    # "administration and management" == "management and administration"
    # Split off the degree prefix, sort the subject words, rejoin.
    prefix_match = re.match(
        r"^(bachelor of science in|bachelor of arts in|bachelor of education in"
        r"|bachelor of business administration in|bachelor of|diploma in|doctor of)\s*",
        name
    )
    if prefix_match:
        prefix = prefix_match.group(0).strip()
        subject = name[prefix_match.end():].strip()
        # Only sort if subject has 2+ words and contains "and"
        # (word-order swaps almost always involve "and" conjunctions)
        if " and " in subject:
            parts = [p.strip() for p in subject.split(" and ")]
            parts_sorted = sorted(parts)
            subject = " and ".join(parts_sorted)
        name = prefix + " " + subject

    return name.strip()


def normalize_course(name):
    name = name.upper()
    name = re.sub(r"\s+", " ", name)
    return name.strip()


# =====================================
# SIMILARITY MATCHING
# =====================================

def is_similar(a, b, threshold=0.96):
    """
    Only treat strings as duplicates if they are near-identical (>= 0.96).
    This catches genuine OCR variants like "forensic science" / "forensic sciences"
    while rejecting similar-but-different programs like
    "software engineering" / "computer engineering" (score ~0.88).
    """
    return SequenceMatcher(None, a, b).ratio() >= threshold


# =====================================
# ENTITY RESOLVER
# =====================================

def resolve_entity(entity, registry, normalizer, college=None):
    """
    Resolves an entity against the registry with two strategies:

    1. Exact normalized match — return existing canonical name.
       For programs, the key is (college, normalized_name) so that the
       same program name in different colleges is stored separately.
    2. Fuzzy similarity >= 0.85 within the same college — treat as
       duplicate.

    NOTE: Prefix/substring containment was intentionally removed.
    It caused cross-college contamination.
    """
    normalized = normalizer(entity)

    # Use college-scoped key for programs to prevent cross-college merging
    registry_key = (college, normalized) if college is not None else normalized

    # 1. Exact match
    if registry_key in registry:
        return registry[registry_key]

    # 2. Fuzzy match — only within the same college
    for existing_key, existing_original in registry.items():
        existing_college = existing_key[0] if isinstance(existing_key, tuple) else None
        existing_norm   = existing_key[1] if isinstance(existing_key, tuple) else existing_key
        if existing_college == college and is_similar(normalized, existing_norm):
            return existing_original

    # New entity — register it
    registry[registry_key] = entity
    return entity


# =====================================
# COURSE CODE EXTRACTOR
# =====================================

def extract_course_code(text):
    match = re.search(r"([A-Z]{2,}\s*\d{2,4})", text)
    if match:
        return match.group(1).replace(" ", "")
    return text


# =====================================
# PROGRAM NAME CLEANER
# =====================================

# ── OCR garble fixes applied before any pattern matching ──
OCR_FIXES = [
    (r"\bphys ics\b", "physics"),
    (r"\bdegr ee\b", "degree"),
    (r"\beducat ion\b", "education"),
    (r"\beduc ation\b", "education"),   # catches "educ ation" variant
    (r"\benv ironment\b", "environment"),
    (r"\btech nology\b", "technology"),
    (r"\bscien ce\b", "science"),
    (r"\bmanage ment\b", "management"),
    (r"\badminist ration\b", "administration"),
    (r"\bcomm unication\b", "communication"),
    (r"\binformat ion\b", "information"),
    (r"\bstat istics\b", "statistics"),
    (r"\bmath ematics\b", "mathematics"),
    (r"\bsoft ware\b", "software"),
    (r"\bhard ware\b", "hardware"),
    (r"\bnet work\b", "network"),
    (r"\bdata base\b", "database"),
    (r"\barch aeology\b", "archaeology"),
    (r"\barch itecture\b", "architecture"),
    (r"\bgeo informatics\b", "geo-informatics"),
    (r"\bmineral ogy\b", "mineralogy"),
    (r"\bbioin formatics\b", "bioinformatics"),
    (r"\baquacult ure\b", "aquaculture"),
    (r"\bpla nning\b", "planning"),
    (r"\bplan ning\b", "planning"),
    (r"\bacco mmodation\b", "accommodation"),
    (r"\bdepart ment\b", "department"),
    (r"\bdevel opment\b", "development"),
    (r"\bcomm unity\b", "community"),
    # Remove extra spaces around commas (OCR artifact: "management , and")
    (r"\s+,\s+", ", "),
    # "degree in" is OCR noise inserted mid-name, collapse it back to "in"
    (r"\bdegree in\b", "in"),
]

# ── Sentence boundary triggers ──
# When any of these appear, everything from that point onward is description text.
# All patterns are word-boundary anchored to avoid false matches inside words.
SENTENCE_TRIGGERS = [
    r"\b(is\s+a|is\s+an)\b",
    r"\b(is\s+to)\b",
    r"\b(is\s+designed)\b",
    r"\b(is\s+offered)\b",
    r"\b(is\s+geared)\b",
    r"\b(is\s+expected)\b",
    r"\b(are\s+core)\b",
    r"\b(are\s+a|are\s+an)\b",
    r"\boffered\s+by\b",
    r"\bprepares\s+[a-z]",
    r"\bfocuses\s+on\b",
    r"\baims\s+to\b",
    r"\bprovides\s+[a-z]",
    r"\bdesigned\s+to\b",
    r"\btakes\s+advantage\b",
    r"\bgeared\s+towards\b",
    r"\bexpected\s+to\b",
    r"\bat\s+the\s+university\b",
    r"\bof\s+the\s+university\b",
    r"\bwa\s+chuo\b",
    r"\bkikuu\b",
    r"\bstudents\s+will\b",
    r"\ba\s+cadre\b",
    r"\bhigh\s+quality\b",
    r"\bcourse\s+of\s+study\b",
    r"\borganized\s+in\b",
    r"\bfull\s*-?\s*time\b",
    r"\bfour\s*-?\s*(year|years)\b",
    r"\bthree\s*-?\s*(year|years)\b",
    r"\btwo\s*-?\s*(year|years)\b",
    r"\bfive\s*-?\s*(year|years)\b",
    # Description sentence starters not caught by other patterns
    r"\bshall\s+(conduct|be|have|include|produce|provide|require)\b",
    r",\s*in\s+the\s+(first|second|third|fourth|final)\s+year\b",
    r"\bin\s+the\s+(first|second|third|fourth|final)\s+year\b",
    r",\s*students\s+(in|are|will|shall)\b",
    r"\bgraduates\s+(will|shall|are|of)\b",
    r"\baim\s+to\b",
    r"\bintended\s+to\b",
    r"\bequip\s+(students|graduates)\b",
]

PROGRAMME_PATTERN = r"\b(programme|program)\b"

# Words that are never valid as the last word in a program name
TRAILING_JUNK = {
    "is", "are", "a", "an", "the", "and", "of", "in", "to", "its",
    "this", "these", "for", "on", "at", "by", "from", "there",
    "cha", "wa", "ya", "na", "it", "with", "or", "as", "be", "not",
    "also", "that", "which", "where", "will", "has", "have", "degree",
    "offered",
}

# Single-word subjects that are specific enough to be a valid program alone
VALID_SINGLE_SUBJECTS = {
    # Sciences & technical
    "sociology", "economics", "mathematics", "physics", "chemistry",
    "biology", "nursing", "medicine", "dentistry", "pharmacy",
    "geology", "geography", "statistics", "biotechnology",
    "aquaculture", "mineralogy",
    # Commerce & management
    "accounting", "marketing", "finance", "management", "administration",
    "commerce",
    # Law
    "law", "laws",
    # Languages & humanities
    "kiswahili", "french", "arabic", "english", "history",
    "archaeology", "philosophy", "linguistics", "literature",
    "translation", "interpretation",
    # Education
    "psychology", "education", "arts", "science",
    # Technology & computing
    "technology", "engineering",
    # Planning
    "planning",
}

# Subject words that on their own are too vague — reject multi-word names
# ending with these (e.g. "diploma in medical laboratories")
INVALID_SUBJECT_ENDINGS = {
    "laboratories", "courses", "study",
}

GENERIC_STUBS = {
    "bachelor of science",
    "bachelor of arts",
    "bachelor of education",
    "bachelor of commerce",
    "bachelor of laws",
    "diploma",
    "doctor of medicine",
    "bachelor of business administration",
}

# Canonical program name map.
# Keys are known truncated/variant names that appear in the PDF
# (from TOC entries, abbreviation lines, etc.).
# Values are the correct full canonical name to use instead.
# This prevents storing both a short and long variant of the same program.
CANONICAL_PROGRAM_MAP = {
    # CIVE truncations
    "bachelor of science in cyber security and digital forensics":
        "bachelor of science in cyber security and digital forensics engineering",
    "bachelor of science in digital content and broadcasting":
        "bachelor of science in digital content and broadcasting engineering",
    "bachelor of science in instructional design and information":
        "bachelor of science in instructional design and information technology",
    "bachelor of science in health information systems":
        "bachelor of science in health information systems",
    # CESE truncations
    "bachelor of science in metallurgy and mineral processing":
        "bachelor of science in metallurgy and mineral processing engineering",
    # COE truncations
    "bachelor of education in adult education and community":
        "bachelor of education in adult education and community development",
    # IDS dirty name
    "bachelor of arts in development studies is one":
        "bachelor of arts in development studies",
    "bachelor of arts in development studies is one of the":
        "bachelor of arts in development studies",
    # CNMS
    "bachelor of science in information systems degree":
        "bachelor of science in information systems",
    # SL
    "bachelor of laws is a":
        "bachelor of laws",
    # COE — "Policy, Planning and Management" is the full name;
    # "Management and Administration" is a duplicate entry for the same program
    "bachelor of education in policy, planning and management":
        "bachelor of education in management and administration",
    "bachelor of education in policy":
        "bachelor of education in management and administration",
    # CHSS
    "bachelor of arts in tourism and cultural heritage":
        "bachelor of arts in cultural heritage and tourism",
    # Misc
    "bachelor of science in information systems degree":
        "bachelor of science in information systems",
    "bachelor of science in statistics is":
        "bachelor of science in statistics",
    "bachelor of science in nursing course of study":
        "bachelor of science in nursing",
    "diploma in nursing course of study":
        "diploma in nursing",
    "diploma in nursing are core courses":
        "diploma in nursing",
    "diploma in medical laboratories are core courses":
        "diploma in medical laboratory technology",
    # CHSS — broken multi-line TOC name fragments
    "bachelor of arts in project planning, management":
        "bachelor of arts in project planning, management, and community development",
    "bachelor of arts in project planning, management, and community":
        "bachelor of arts in project planning, management, and community development",
    "bachelor of arts in project planning":
        "bachelor of arts in project planning, management, and community development",
    # CHSS — sociology description line leaked
    # Both pre and post comma-fix forms covered
    "bachelor of arts in sociology , in the third year , shall conduct":
        "bachelor of arts in sociology",
    "bachelor of arts in sociology, in the third year, shall conduct":
        "bachelor of arts in sociology",
    "bachelor of arts in sociology at the university of dodoma":
        "bachelor of arts in sociology",
    # CHSS — tourism/cultural heritage word-order variant
    "bachelor of arts in tourism and cultural heritage":
        "bachelor of arts in cultural heritage and tourism",
    # CBE — entrepreneurship description leaked
    "bachelor of commerce in entrepreneurship, graduates will be able":
        "bachelor of commerce in entrepreneurship",
    # CESE — environmental engineering (missing from previous map)
    "bachelor of science in environmental engineering":
        "bachelor of science in environmental engineering",
    # COE — same program, word order flipped between sections
    "bachelor of education in management and administration":
        "bachelor of education in administration and management",
    # IDS — OCR garbled variants (pla nning / management ,)
    "bachelor of arts in project pla nning, management , and community":
        "bachelor of arts in project planning, management, and community development",
    "bachelor of arts in project pla nning, management , and community development":
        "bachelor of arts in project planning, management, and community development",
    "bachelor of arts in project planning, management , and community":
        "bachelor of arts in project planning, management, and community development",
    "bachelor of arts in project planning, management , and community development":
        "bachelor of arts in project planning, management, and community development",
}

def clean_program_name(raw):
    """
    Strips description text from a raw program name match.
    Returns the cleaned name, or None if it should be rejected.
    """
    name = raw.lower().strip()

    # 1. Fix OCR garbles
    for pattern, replacement in OCR_FIXES:
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)

    # 2. Remove page number artifacts (e.g. "... 149")
    name = re.sub(r"\.{2,}\s*\d+\s*$", "", name)

    # 3. Remove abbreviations — both closed "(b.com)" and unclosed "(b.com"
    name = re.sub(r"\([^)]{0,30}\)?$", "", name)
    name = re.sub(r"\(.*?\)", "", name)

    # 4. Cut at sentence boundary triggers (keep everything BEFORE the match)
    for trigger in SENTENCE_TRIGGERS:
        m = re.search(trigger, name, flags=re.IGNORECASE)
        if m:
            name = name[:m.start()].strip()

    # 5. Cut at "programme" / "program"
    m = re.search(PROGRAMME_PATTERN, name, flags=re.IGNORECASE)
    if m:
        name = name[:m.start()].strip()

    # 6. Remove trailing punctuation and normalize whitespace
    name = re.sub(r"[^a-zA-Z0-9\s&\-]+$", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # 7. Hard length guard
    if len(name) > 80:
        name = name[:80].rsplit(" ", 1)[0].strip()

    # 8. Strip trailing junk words
    words = name.split()
    while words and words[-1].lower() in TRAILING_JUNK:
        words.pop()
    name = " ".join(words)

    # 9. Reject if too short
    if len(name) < 10:
        return None

    # 10. Reject generic stubs
    if name.strip() in GENERIC_STUBS:
        return None

    # 10b. Canonicalize known variant/truncated names to their full correct form
    if name in CANONICAL_PROGRAM_MAP:
        name = CANONICAL_PROGRAM_MAP[name]

    # 11. Strip degree prefix and validate subject
    stripped = re.sub(
        r"^(bachelor of science in?|bachelor of arts in?|bachelor of education in?"
        r"|bachelor of business administration in?|bachelor of|diploma in?|doctor of)\s*",
        "", name
    ).strip()

    subj_words = stripped.split()

    if not subj_words:
        return None

    if len(subj_words) == 1:
        # Single-word subjects only valid if they are known academic nouns
        if subj_words[0].lower() not in VALID_SINGLE_SUBJECTS:
            return None
    else:
        # Multi-word subjects: last word cannot be a known invalid ending
        if subj_words[-1].lower() in INVALID_SUBJECT_ENDINGS:
            return None

    return name


# =====================================
# PARSE CURRICULUM
# =====================================

def parse_curriculum(text):
    lines = text.split("\n")
    curriculum_docs = []

    current_college = None
    current_program = None
    current_year = None
    current_semester = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove leading numbering like "1.", "2.3", "1.2.3"
        line_clean = re.sub(r"^\d+(\.\d+)*\s*", "", line)
        lower = line_clean.lower()

        # -----------------------------------------------
        # COLLEGE / SCHOOL / INSTITUTE DETECTION
        # -----------------------------------------------
        college_match = re.search(
            r"^(college of [a-z\s&]+|school of [a-z\s&]+|institute of [a-z\s&]+|confucius institute)",
            lower
        )

        if college_match:
            raw_college = college_match.group(0)
            # Strip trailing page dots/numbers and noise
            raw_college = re.sub(r"\.{2,}\s*\d+\s*$", "", raw_college)
            raw_college = re.sub(r"\(.*?\)", "", raw_college)
            raw_college = re.sub(r"\s+", " ", raw_college).strip()

            resolved_college = resolve_entity(raw_college, colleges, normalize_college)
            current_college = resolved_college.title()
            current_program = None
            current_year = None
            current_semester = None

            curriculum_docs.append(Document(
                page_content=f"College: {current_college}",
                metadata={
                    "type": "college",
                    "college": current_college,
                    "level": "college"
                }
            ))
            continue

        # -----------------------------------------------
        # PROGRAM DETECTION
        # -----------------------------------------------
        if current_college and re.search(
            r"(bachelor of|doctor of|diploma in|shahada ya awali)",
            lower
        ):
            program_match = re.search(
                r"(bachelor of [a-z\s\.,\-&\(\)]+|b\.sc\.\s*[a-z\s\.,\-&\(\)]+|doctor of [a-z\s\.,\-&\(\)]+|diploma in [a-z\s\.,\-&\(\)]+|shahada ya awali[^\n]*)",
                lower
            )

            if program_match:
                raw_program = program_match.group(1)
                cleaned = clean_program_name(raw_program)

                # Skip if cleaner rejected it
                if not cleaned:
                    continue

                # Second-pass clean: run cleaner again on its own output to
                # catch any residue left by the first pass (e.g. "is a" that
                # only becomes visible after "programme" is stripped first)
                cleaned = clean_program_name(cleaned) or cleaned

                # Final validation: reject if cleaned name ends with a
                # trailing junk word that slipped through both passes
                _words = cleaned.split()
                TRAILING_JUNK_FINAL = {
                    "is","are","a","an","the","and","of","in","to",
                    "at","by","from","it","or","as","be","not","that",
                    "which","where","will","has","have","degree","offered",
                    "course","courses","study",
                }
                while _words and _words[-1].lower() in TRAILING_JUNK_FINAL:
                    _words.pop()
                if len(_words) < 3:
                    continue
                cleaned = " ".join(_words)

                # Final canonical lookup — normalize any remaining variants
                if cleaned in CANONICAL_PROGRAM_MAP:
                    cleaned = CANONICAL_PROGRAM_MAP[cleaned]

                resolved_program = resolve_entity(cleaned, programs, normalize_program, college=current_college)

                # Only emit a new program doc if this is genuinely a new program
                if resolved_program != current_program:
                    current_program = resolved_program
                    current_semester = None
                    current_year = None

                    curriculum_docs.append(Document(
                        page_content=f"{current_program} under {current_college}",
                        metadata={
                            "type": "program",
                            "college": current_college,
                            "program": current_program,
                            "level": "program"
                        }
                    ))

                    curriculum_docs.append(Document(
                        page_content=f"College: {current_college}\nProgram: {current_program}",
                        metadata={
                            "type": "college_program_map",
                            "college": current_college,
                            "program": current_program
                        }
                    ))

            continue

        # -----------------------------------------------
        # YEAR DETECTION
        # -----------------------------------------------
        year_match = re.search(r"year\s+(one|two|three|four)", lower)
        if year_match:
            current_year = f"year {year_match.group(1).strip()}"
            continue

        # -----------------------------------------------
        # SEMESTER DETECTION
        # -----------------------------------------------
        semester_match = re.search(r"semester\s+(one|two)", lower)
        if semester_match:
            current_semester = f"semester {semester_match.group(1)}"
            continue

        # -----------------------------------------------
        # COURSE DETECTION
        # -----------------------------------------------
        if current_program and current_year and current_semester:
            course_match = re.search(r"^([A-Z]{2,}\s*\d{2,4})\s+(.*)", line_clean)
            if course_match:
                course_code = course_match.group(1).strip()
                course_title = course_match.group(2).strip() or "Unknown Course Title"
                credits_match = re.search(r"core\s*([\d\.]+)", lower)
                credits = credits_match.group(1) if credits_match else "N/A"

                # Scope course deduplication to (college, program, course_code)
                # so the same course code appearing in multiple programs
                # creates separate entries per program rather than merging.
                course_key = f"{current_college}::{current_program}::{course_code}"
                resolved_course = resolve_entity(
                    f"{course_code} - {course_title}",
                    courses,
                    normalize_course,
                    college=course_key
                )

                curriculum_docs.append(Document(
                    page_content=(
                        f"College: {current_college}\n"
                        f"Program: {current_program}\n"
                        f"Year: {current_year}\n"
                        f"Semester: {current_semester}\n"
                        f"Course Code: {course_code}\n"
                        f"Course Title: {course_title}\n"
                        f"Credits: {credits}"
                    ),
                    metadata={
                        "type": "course",
                        "level": "course",
                        "college": current_college,
                        "program": current_program,
                        "year": current_year,
                        "semester": current_semester,
                        "course_code": course_code,
                        "course_title": course_title,
                        "credits": credits,
                        "course": resolved_course
                    }
                ))

    return curriculum_docs


# =====================================
# PARSE REGULATION
# =====================================

def parse_regulation(text):
    regulation_docs = []
    sections = re.split(r"\n(?=\d+\.\d+)", text)

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        section_match = re.search(r"(\d+\.\d+)", sec)
        if not section_match:
            continue

        section_number = section_match.group(1)

        def topic_tagging(text):
            text = text.lower()
            if any(x in text for x in ["gpa", "grade point average", "academic standing"]):
                return "gpa"
            if any(x in text for x in ["credit", "course credit"]):
                return "credit"
            if any(x in text for x in ["carry", "carrying"]):
                return "carry"
            if any(x in text for x in ["classification", "degree classification"]):
                return "degree_classification"
            if any(x in text for x in ["discontinue", "discontinuation"]):
                return "discontinuous"
            return "general"

        paragraphs = sec.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if len(para) < 40:
                continue

            topic = topic_tagging(para)

            regulation_docs.append(Document(
                page_content=f"Section: {section_number}\n\nTopic: {topic}\n\nContent:\n{para}",
                metadata={
                    "type": "regulation",
                    "level": "regulation",
                    "section": section_number,
                    "topic": topic
                }
            ))

    return regulation_docs


# =====================================
# PARSE BYLAW
# =====================================

def parse_bylaw(text):
    bylaw_docs = []
    current_section = None
    buffer = ""

    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        lower = line.lower()

        section_match = re.search(r"^\d+(\.\d+)*", line)
        if section_match:
            # Flush previous buffer before moving to new section
            if buffer and current_section:
                bylaw_docs.append(Document(
                    page_content=f"Section: {current_section}\n\nContent:\n{buffer.strip()}",
                    metadata={
                        "type": "bylaw",
                        "level": "bylaw",
                        "section": current_section
                    }
                ))
                buffer = ""
            current_section = section_match.group(0)
            continue

        bullet_match = re.search(r"^\(?[ivx]+\)", lower)
        if bullet_match:
            if buffer and current_section:
                bylaw_docs.append(Document(
                    page_content=f"Section: {current_section}\n\nContent:\n{buffer.strip()}",
                    metadata={
                        "type": "bylaw",
                        "level": "bylaw",
                        "section": current_section
                    }
                ))
            buffer = line
            continue

        buffer += " " + line

    # Flush final buffer
    if buffer and current_section:
        bylaw_docs.append(Document(
            page_content=f"Section: {current_section}\n\nContent:\n{buffer.strip()}",
            metadata={
                "type": "bylaw",
                "level": "bylaw",
                "section": current_section
            }
        ))

    return bylaw_docs


# =====================================
# AGGREGATE ALL DOCUMENTS
# =====================================

curriculum_text = ""
regulation_text = ""
bylaw_text = ""

for doc in raw_documents:
    source = doc.metadata.get("source", "").lower()

    if "curriculum" in source:
        curriculum_text += doc.page_content + "\n"
    elif "regulation" in source:
        regulation_text += doc.page_content + "\n"
    elif "student" in source:
        bylaw_text += doc.page_content + "\n"

curriculum_docs = parse_curriculum(curriculum_text)
regulation_docs = parse_regulation(regulation_text)
bylaw_docs = parse_bylaw(bylaw_text)

all_documents = []
all_documents.extend(curriculum_docs)
all_documents.extend(regulation_docs)
all_documents.extend(bylaw_docs)

# NOTE: colleges registry is already populated by parse_curriculum.
# We do NOT append college docs again here — that was causing duplicates.
# The college docs are already inside curriculum_docs.

print(f"\nTotal documents to store: {len(all_documents)}")
print(f"  Curriculum docs : {len(curriculum_docs)}")
print(f"  Regulation docs : {len(regulation_docs)}")
print(f"  Bylaw docs      : {len(bylaw_docs)}")
print(f"  Unique colleges : {len(colleges)}")
print(f"  Unique programs : {len(programs)}")
print(f"  Unique courses  : {len(courses)}")


# =====================================
# STORE IN CHROMADB
# =====================================

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

persist_directory = "./chroma_db"

shutil.rmtree(persist_directory, ignore_errors=True)

stored_vector = Chroma.from_documents(
    documents=all_documents,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="university_docs"
)

print(f"\nDocuments stored successfully → {persist_directory}")
print(f"Total vectors in collection: {stored_vector._collection.count()}")

# ── Diagnostic: detect program name mismatches ─────────────────────────
# Course docs and college_program_map docs MUST use identical program
# name strings — any mismatch means course filter returns 0 results.
all_stored = stored_vector.get()
map_programs = set()
course_programs = set()
for meta in all_stored["metadatas"]:
    t = meta.get("type", "")
    p = meta.get("program", "")
    if p:
        if t == "college_program_map":
            map_programs.add(p)
        elif t == "course":
            course_programs.add(p)

missing_in_map = course_programs - map_programs
missing_in_courses = map_programs - course_programs

if missing_in_map:
    print("\n WARNING - Programs with COURSES but NO college_program_map entry:")
    for p in sorted(missing_in_map):
        print(f"  - {p}")
if missing_in_courses:
    print("\n WARNING - Programs in college_program_map with NO course docs:")
    for p in sorted(missing_in_courses):
        print(f"  - {p}")
if not missing_in_map and not missing_in_courses:
    print("\n OK - All program names consistent between course docs and map")

# ── Near-duplicate detection: warn about suspiciously similar program names ──
from difflib import SequenceMatcher
all_map_programs = sorted(map_programs)
near_dupes = []
for i, a in enumerate(all_map_programs):
    for b in all_map_programs[i+1:]:
        ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        if ratio > 0.88:
            near_dupes.append((ratio, a, b))

if near_dupes:
    print("\n WARNING - Near-duplicate program names (may need deduplication):")
    for ratio, a, b in sorted(near_dupes, reverse=True):
        print(f"  {ratio:.2f}  {repr(a)}")
        print(f"         {repr(b)}")
else:
    print("\n OK - No near-duplicate program names detected")