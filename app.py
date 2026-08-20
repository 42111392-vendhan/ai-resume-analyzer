import os
import re
import time
import hashlib
from collections import Counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1 {
        font-size: 2.7rem !important;
        font-weight: 800 !important;
    }

    h2 {
        margin-top: 2rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 14px;
    }

    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    textarea {
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GEMINI SETUP
# ============================================================

load_dotenv()


def get_gemini_api_key():

    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")

        if secret_key:
            return secret_key

    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


GEMINI_API_KEY = get_gemini_api_key()

PRIMARY_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.5-flash-lite"
)

client = (
    genai.Client(
        api_key=GEMINI_API_KEY
    )
    if GEMINI_API_KEY
    else None
)


# ============================================================
# GEMINI RETRY + FALLBACK
# ============================================================

def generate_ai_response(prompt):

    if client is None:

        raise RuntimeError(
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to .env locally "
            "or Streamlit Secrets after deployment."
        )

    models = []

    for model_name in [
        PRIMARY_MODEL,
        FALLBACK_MODEL
    ]:

        if (
            model_name
            and model_name not in models
        ):

            models.append(
                model_name
            )

    last_error = None

    for model_name in models:

        for attempt in range(2):

            try:

                response = (
                    client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                )

                text = getattr(
                    response,
                    "text",
                    None
                )

                if text and text.strip():

                    return (
                        text.strip(),
                        model_name
                    )

                last_error = RuntimeError(
                    f"{model_name} returned an empty response."
                )

            except Exception as exc:

                last_error = exc

                error_text = (
                    str(exc).lower()
                )

                temporary_error = any(
                    marker in error_text
                    for marker in [
                        "503",
                        "unavailable",
                        "429",
                        "resource_exhausted",
                        "high demand",
                        "deadline",
                        "timeout",
                    ]
                )

                if temporary_error:

                    time.sleep(
                        2 ** attempt
                    )

                    continue

                model_error = any(
                    marker in error_text
                    for marker in [
                        "404",
                        "not found",
                        "not supported",
                        "invalid model",
                    ]
                )

                if model_error:
                    break

                raise

    raise RuntimeError(
        "Gemini is currently unavailable. "
        f"Last error: {last_error}"
    )


# ============================================================
# SKILL / KEYWORD CONFIGURATION
# ============================================================

SKILLS = [

    # Programming
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c#",
    "r",

    # Web
    "html",
    "css",
    "react",
    "angular",
    "node.js",
    "nodejs",
    "flask",
    "django",
    "fastapi",
    "streamlit",

    # Databases
    "sql",
    "mysql",
    "postgresql",
    "mongodb",

    # Data Analytics
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "plotly",
    "excel",
    "power bi",
    "tableau",
    "data analysis",
    "data visualization",
    "statistics",
    "data cleaning",

    # AI / ML
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "generative ai",
    "genai",
    "nlp",
    "natural language processing",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "sklearn",
    "opencv",

    # GenAI
    "llm",
    "llms",
    "rag",
    "retrieval augmented generation",
    "prompt engineering",
    "embeddings",
    "vector database",
    "langchain",
    "llamaindex",
    "gemini",
    "openai",
    "chatgpt",
    "ai agents",
    "agents",

    # Cloud / DevOps
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "github",
    "linux",

    # APIs
    "rest api",
    "api",
]


STOP_WORDS = {

    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "a",
    "an",
    "for",
    "with",
    "on",
    "is",
    "are",
    "be",
    "as",
    "at",
    "by",
    "from",
    "that",
    "this",
    "will",
    "we",
    "you",
    "your",
    "our",
    "have",
    "has",
    "should",
    "must",
    "can",
    "using",
    "use",
    "work",
    "working",
    "role",
    "job",
    "candidate",
    "required",
    "preferred",
    "skills",
    "skill",
    "responsibilities",
    "responsibility",
    "looking",
    "knowledge",
    "good",
    "strong",
    "ability",
}


ACTION_VERBS = [

    "developed",
    "created",
    "built",
    "implemented",
    "designed",
    "analyzed",
    "improved",
    "optimized",
    "generated",
    "automated",
    "integrated",
    "deployed",
    "trained",
    "performed",
    "visualized",
    "processed",
    "engineered",
    "tested",
    "configured",
    "documented",
]


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):

    return re.sub(
        r"\s+",
        " ",
        text or ""
    ).strip()


def term_in_text(
    term,
    text
):

    pattern = (
        r"(?<!\w)"
        + re.escape(
            term.lower()
        )
        + r"(?!\w)"
    )

    return bool(
        re.search(
            pattern,
            (text or "").lower()
        )
    )


def has_any_keyword(
    text,
    keywords
):

    return any(
        term_in_text(
            keyword,
            text
        )
        for keyword in keywords
    )


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(
    pdf_file
):

    reader = PdfReader(
        pdf_file
    )

    pages = []

    for page in reader.pages:

        page_text = (
            page.extract_text()
        )

        if page_text:

            pages.append(
                page_text
            )

    return (
        "\n".join(
            pages
        ).strip(),
        len(reader.pages)
    )


# ============================================================
# SKILL / KEYWORD EXTRACTION
# ============================================================

def extract_skills(text):

    return sorted(
        {
            skill
            for skill in SKILLS
            if term_in_text(
                skill,
                text
            )
        }
    )


def extract_keywords(
    text,
    top_n=20
):

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9+#.-]{2,}\b",
        (text or "").lower(),
    )

    cleaned_words = [

        word

        for word in words

        if word not in STOP_WORDS
    ]

    counts = Counter(
        cleaned_words
    )

    return [

        word

        for word, _
        in counts.most_common(
            top_n
        )
    ]


# ============================================================
# SIMILARITY
# ============================================================

def calculate_similarity(
    resume_text,
    job_description
):

    try:

        vectorizer = (
            TfidfVectorizer(
                stop_words="english"
            )
        )

        vectors = (
            vectorizer.fit_transform(
                [
                    resume_text,
                    job_description
                ]
            )
        )

        similarity = (
            cosine_similarity(
                vectors[0:1],
                vectors[1:2]
            )[0][0]
        )

        return float(
            similarity
        )

    except Exception:

        return 0.0


# ============================================================
# FRESHER STRUCTURE SCORE
# ============================================================

def calculate_structure_score(
    resume_text
):

    sections = {

        "Professional Summary": [
            "professional summary",
            "summary",
            "career objective",
            "objective",
            "profile",
            "about me",
        ],

        "Technical Skills": [
            "technical skills",
            "skills",
            "technologies",
            "tools",
        ],

        "Projects": [
            "projects",
            "project",
            "academic project",
        ],

        "Education": [
            "education",
            "academic",
            "university",
            "college",
        ],
    }

    found_sections = [

        section

        for section, keywords
        in sections.items()

        if has_any_keyword(
            resume_text,
            keywords
        )
    ]

    score = (
        len(found_sections)
        / len(sections)
    ) * 10

    return (
        round(score, 1),
        found_sections
    )


# ============================================================
# FRESHER SECTION HEALTH HELPERS
# ============================================================

def count_action_verbs(text):

    return sum(

        len(
            re.findall(
                r"\b"
                + re.escape(verb)
                + r"\b",
                (text or "").lower()
            )
        )

        for verb
        in ACTION_VERBS
    )


def contains_impact_metrics(text):

    text = text or ""

    patterns = [

        r"\b\d+(?:\.\d+)?%\b",

        r"\b\d+\+\b",

        r"\b\d+(?:\.\d+)?x\b",

        r"\b\d+\s+(?:users|records|rows|files|queries|requests|samples)\b",

        r"\b(?:reduced|increased|improved|achieved)\s+[^.\n]{0,30}\d+",
    ]

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )
        for pattern in patterns
    )


# ============================================================
# FRESHER SECTION HEALTH
# ============================================================

def calculate_section_health(
    resume_text,
    matched_skills,
    jd_skills
):

    results = {}


    # ========================================================
    # PROFESSIONAL SUMMARY — 20
    # ========================================================

    summary_score = 0

    summary_tips = []

    if has_any_keyword(
        resume_text,
        [
            "professional summary",
            "summary",
            "career objective",
            "objective",
            "profile",
            "about me",
        ],
    ):

        summary_score += 10

    else:

        summary_tips.append(
            "Add a short Professional Summary or Career Objective."
        )


    if len(matched_skills) >= 2:

        summary_score += 5

    else:

        summary_tips.append(
            "Mention genuine technical strengths that align with the target role."
        )


    word_count = len(
        resume_text.split()
    )

    if 200 <= word_count <= 900:

        summary_score += 5

    else:

        summary_tips.append(
            "Keep the full resume concise but detailed enough to show skills and projects."
        )


    results[
        "Professional Summary"
    ] = {

        "score":
            min(
                summary_score,
                20
            ),

        "max":
            20,

        "tips":
            summary_tips,
    }


    # ========================================================
    # TECHNICAL SKILLS — 30
    # ========================================================

    skills_score = 0

    skills_tips = []

    if has_any_keyword(
        resume_text,
        [
            "technical skills",
            "skills",
            "technologies",
            "tools"
        ]
    ):

        skills_score += 10

    else:

        skills_tips.append(
            "Add a clearly labeled Technical Skills section."
        )


    if jd_skills:

        match_ratio = (
            len(matched_skills)
            / len(jd_skills)
        )

        skills_score += min(
            match_ratio * 20,
            20
        )

        if match_ratio < 0.60:

            skills_tips.append(
                "Improve genuine skill alignment with the target role; "
                "do not add skills you have not learned."
            )

    else:

        skills_score += 10

        skills_tips.append(
            "Few predefined technical skills were detected "
            "in the job description, so skill scoring is limited."
        )


    results[
        "Technical Skills"
    ] = {

        "score":
            round(
                min(
                    skills_score,
                    30
                ),
                1
            ),

        "max":
            30,

        "tips":
            skills_tips,
    }


    # ========================================================
    # PROJECTS — 35
    # ========================================================

    projects_score = 0

    project_tips = []

    if has_any_keyword(
        resume_text,
        [
            "projects",
            "project",
            "academic project"
        ]
    ):

        projects_score += 15

    else:

        project_tips.append(
            "Add a Projects section. "
            "Projects are especially important for freshers."
        )


    action_count = (
        count_action_verbs(
            resume_text
        )
    )

    if action_count >= 4:

        projects_score += 10

    elif action_count >= 2:

        projects_score += 6

        project_tips.append(
            "Use a few more strong project action verbs such as "
            "Developed, Built, Implemented, or Analyzed."
        )

    elif action_count == 1:

        projects_score += 3

        project_tips.append(
            "Describe project contributions with clearer action verbs."
        )

    else:

        project_tips.append(
            "Describe project contributions using strong action verbs."
        )


    if len(matched_skills) >= 3:

        projects_score += 5

    else:

        project_tips.append(
            "Clearly mention the technologies genuinely used in your projects."
        )


    if contains_impact_metrics(
        resume_text
    ):

        projects_score += 5

    else:

        project_tips.append(
            "Add measurable project outcomes only where they are truthful and available."
        )


    results[
        "Projects"
    ] = {

        "score":
            min(
                projects_score,
                35
            ),

        "max":
            35,

        "tips":
            project_tips,
    }


    # ========================================================
    # EDUCATION — 15
    # ========================================================

    education_score = 0

    education_tips = []

    if has_any_keyword(
        resume_text,
        [
            "education",
            "academic",
            "university",
            "college"
        ]
    ):

        education_score += 7

    else:

        education_tips.append(
            "Add a clearly labeled Education section."
        )


    if has_any_keyword(
        resume_text,
        [
            "bachelor",
            "b.e",
            "b.tech",
            "bsc",
            "b.sc",
            "master",
            "m.tech",
            "computer science",
            "engineering",
            "degree",
        ],
    ):

        education_score += 4

    else:

        education_tips.append(
            "Clearly mention your degree and specialization."
        )


    if re.search(
        r"\b(?:19|20)\d{2}\b",
        resume_text
    ):

        education_score += 4

    else:

        education_tips.append(
            "Mention your graduation year or study period."
        )


    results[
        "Education"
    ] = {

        "score":
            min(
                education_score,
                15
            ),

        "max":
            15,

        "tips":
            education_tips,
    }


    total = round(
        sum(
            item["score"]
            for item
            in results.values()
        ),
        1
    )

    return (
        results,
        total
    )


# ============================================================
# ATS CALCULATOR
# ============================================================

def calculate_ats_score_for_text(
    candidate_text,
    job_description
):

    candidate_skills = set(
        extract_skills(
            candidate_text
        )
    )

    target_skills = set(
        extract_skills(
            job_description
        )
    )

    matched_skills = sorted(
        candidate_skills
        & target_skills
    )

    missing_skills = sorted(
        target_skills
        - candidate_skills
    )


    if target_skills:

        skill_score = (
            len(matched_skills)
            / len(target_skills)
        ) * 50

    else:

        skill_score = 0.0


    target_keywords = (
        extract_keywords(
            job_description
        )
    )


    matched_keywords = [

        keyword

        for keyword
        in target_keywords

        if term_in_text(
            keyword,
            candidate_text
        )
    ]


    missing_keywords = [

        keyword

        for keyword
        in target_keywords

        if not term_in_text(
            keyword,
            candidate_text
        )
    ]


    if target_keywords:

        keyword_score = (
            len(matched_keywords)
            / len(target_keywords)
        ) * 20

    else:

        keyword_score = 0.0


    similarity = (
        calculate_similarity(
            candidate_text,
            job_description
        )
    )

    similarity_score = (
        similarity * 20
    )


    (
        structure_score,
        found_sections
    ) = calculate_structure_score(
        candidate_text
    )


    total = min(
        skill_score
        + keyword_score
        + similarity_score
        + structure_score,
        100,
    )


    return {

        "total":
            round(
                total,
                1
            ),

        "skills":
            round(
                skill_score,
                1
            ),

        "keywords":
            round(
                keyword_score,
                1
            ),

        "similarity":
            round(
                similarity_score,
                1
            ),

        "similarity_raw":
            similarity,

        "structure":
            round(
                structure_score,
                1
            ),

        "matched_skills":
            matched_skills,

        "missing_skills":
            missing_skills,

        "matched_keywords":
            matched_keywords,

        "missing_keywords":
            missing_keywords,

        "found_sections":
            found_sections,

        "target_skills":
            sorted(
                target_skills
            ),
    }


# ============================================================
# SESSION STATE HELPER
# ============================================================

def reset_generated_outputs_if_input_changed(
    resume_text,
    job_description
):

    fingerprint = hashlib.sha256(
        (
            resume_text
            + "\n---JD---\n"
            + job_description
        ).encode(
            "utf-8"
        )
    ).hexdigest()


    if (
        st.session_state.get(
            "input_fingerprint"
        )
        != fingerprint
    ):

        for key in [

            "optimized_resume_content",
            "optimizer_model_used",
            "optimizer_mode_used",
            "ai_review_content",
            "ai_review_model",
            "interview_content",
            "interview_model",

        ]:

            st.session_state.pop(
                key,
                None
            )


        st.session_state[
            "input_fingerprint"
        ] = fingerprint


# ============================================================
# HERO
# ============================================================

st.title(
    "📄 AI Resume Analyzer"
)

st.markdown(
    "### 🎯 ATS Analysis • Resume Optimization • Job Fit • Interview Prep"
)

st.write(
    "Analyze a fresher resume against a job description, "
    "identify genuine skill and keyword gaps, improve resume "
    "wording, and prepare for interviews."
)

st.info(
    "🎓 Built for freshers and entry-level candidates. "
    "Professional work experience is not required and is not "
    "part of the fresher resume health score."
)

st.caption(
    "ATS, resume-health, job-fit and interview-readiness scores "
    "are estimates from this application, not official employer "
    "ATS scores or hiring predictions."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "📄 Resume AI"
    )

    st.caption(
        "Fresher-focused resume analysis"
    )

    st.divider()

    st.markdown(
        "### 🔍 Features"
    )

    st.write(
        "🎯 ATS Compatibility"
    )

    st.write(
        "🛠️ Skills & Keyword Matching"
    )

    st.write(
        "🎓 Fresher Resume Health"
    )

    st.write(
        "🎯 Job Fit Dashboard"
    )

    st.write(
        "🛠️ AI Resume Optimizer"
    )

    st.write(
        "📊 Before vs After Projection"
    )

    st.write(
        "🤖 AI Resume Review"
    )

    st.write(
        "🎤 Interview Preparation"
    )

    st.divider()

    st.markdown(
        "### ⚙️ Method"
    )

    st.caption(
        "Python + TF-IDF calculate deterministic compatibility "
        "metrics. Gemini is used for explanations, rewriting "
        "and interview preparation."
    )


# ============================================================
# INPUTS
# ============================================================

st.header(
    "📥 Resume & Job Details"
)

upload_col1, upload_col2 = (
    st.columns(
        [1, 1]
    )
)


with upload_col1:

    uploaded_resume = (
        st.file_uploader(
            "📄 Upload Resume",
            type=["pdf"],
            help="Upload a text-based PDF resume.",
        )
    )


with upload_col2:

    st.markdown(
        "#### 💡 Fresher Resume Tips"
    )

    st.write(
        "• PDF format\n"
        "• Include Technical Skills\n"
        "• Include Projects\n"
        "• Include Education\n"
        "• A short summary/objective is useful"
    )


job_description = (
    st.text_area(
        "💼 Paste Job Description",
        height=250,
        placeholder=(
            "Paste the complete job description here..."
        ),
    )
)


# ============================================================
# MAIN APP
# ============================================================

if uploaded_resume is None:

    st.info(
        "👆 Upload your PDF resume to begin."
    )

    st.stop()


try:

    (
        resume_text,
        page_count
    ) = extract_pdf_text(
        uploaded_resume
    )

except Exception as exc:

    st.error(
        f"❌ Could not read the PDF: {exc}"
    )

    st.stop()


if not normalize_text(
    resume_text
):

    st.error(
        "❌ No readable text was found in this PDF. "
        "It may be a scanned/image-only resume."
    )

    st.stop()


st.success(
    "✅ Resume uploaded successfully!"
)

st.divider()


# ============================================================
# RESUME INFORMATION
# ============================================================

st.subheader(
    "📊 Resume Information"
)

info1, info2, info3 = (
    st.columns(3)
)

info1.metric(
    "Pages",
    page_count
)

info2.metric(
    "Words",
    len(
        resume_text.split()
    )
)

info3.metric(
    "Characters",
    len(
        resume_text
    )
)


with st.expander(
    "📝 View Extracted Resume Text"
):

    st.text_area(
        "Resume Content",
        value=resume_text,
        height=350,
        disabled=True,
    )


if not job_description.strip():

    st.info(
        "👆 Paste a job description to start "
        "ATS and job-fit analysis."
    )

    st.stop()


reset_generated_outputs_if_input_changed(
    resume_text,
    job_description
)


# ============================================================
# CORE ATS ANALYSIS
# ============================================================

ats = calculate_ats_score_for_text(
    resume_text,
    job_description
)

ats_score = ats["total"]

matched_skills = (
    ats["matched_skills"]
)

missing_skills = (
    ats["missing_skills"]
)

matched_keywords = (
    ats["matched_keywords"]
)

missing_keywords = (
    ats["missing_keywords"]
)

jd_skills = set(
    ats["target_skills"]
)

similarity = (
    ats["similarity_raw"]
)

found_sections = (
    ats["found_sections"]
)


st.header(
    "🎯 ATS Compatibility Analysis"
)


if ats_score >= 80:

    st.success(
        f"🔥 Excellent Match — "
        f"{ats_score}/100"
    )

elif ats_score >= 65:

    st.info(
        f"✅ Good Match — "
        f"{ats_score}/100"
    )

elif ats_score >= 50:

    st.warning(
        f"⚠️ Moderate Match — "
        f"{ats_score}/100"
    )

else:

    st.error(
        f"❌ Low Match — "
        f"{ats_score}/100"
    )


st.progress(
    int(
        ats_score
    )
)


st.subheader(
    "📊 ATS Score Breakdown"
)

score1, score2, score3, score4 = (
    st.columns(4)
)

score1.metric(
    "🛠️ Skills",
    f"{ats['skills']}/50"
)

score2.metric(
    "🔑 Keywords",
    f"{ats['keywords']}/20"
)

score3.metric(
    "🎯 Resume ↔ JD",
    f"{ats['similarity']}/20"
)

score4.metric(
    "📄 Structure",
    f"{ats['structure']}/10"
)


# ============================================================
# SKILLS ANALYSIS
# ============================================================

st.subheader(
    "🛠️ Skills Analysis"
)

skill_col1, skill_col2 = (
    st.columns(2)
)


with skill_col1:

    st.markdown(
        "### ✅ Matched Skills"
    )

    if matched_skills:

        for skill in matched_skills:

            st.write(
                f"• {skill}"
            )

    else:

        st.write(
            "No matching predefined skills detected."
        )


with skill_col2:

    st.markdown(
        "### ⚠️ Missing Job Skills"
    )

    if missing_skills:

        for skill in missing_skills:

            st.write(
                f"• {skill}"
            )

    else:

        st.write(
            "No missing predefined job skills detected."
        )


# ============================================================
# KEYWORD ANALYSIS
# ============================================================

st.subheader(
    "🔑 Keyword Analysis"
)

key_col1, key_col2 = (
    st.columns(2)
)


with key_col1:

    st.markdown(
        "### ✅ Matched Keywords"
    )

    if matched_keywords:

        for keyword in (
            matched_keywords[:10]
        ):

            st.write(
                f"• {keyword}"
            )

    else:

        st.write(
            "No major matched keywords detected."
        )


with key_col2:

    st.markdown(
        "### ⚠️ Missing Keywords"
    )

    if missing_keywords:

        for keyword in (
            missing_keywords[:10]
        ):

            st.write(
                f"• {keyword}"
            )

    else:

        st.write(
            "No major missing keywords detected."
        )


st.subheader(
    "📄 Detected Fresher Resume Sections"
)


if found_sections:

    st.write(
        " • ".join(
            found_sections
        )
    )

else:

    st.write(
        "No standard fresher resume "
        "section headings were detected."
    )


# ============================================================
# FRESHER RESUME HEALTH
# ============================================================

(
    section_results,
    section_health_score
) = calculate_section_health(
    resume_text,
    matched_skills,
    jd_skills,
)


st.header(
    "🎓 Fresher Resume Section Health"
)

st.caption(
    "Scoring: Professional Summary 20 • "
    "Technical Skills 30 • Projects 35 • Education 15"
)


if section_health_score >= 80:

    st.success(
        f"🔥 Strong Fresher Resume — "
        f"{section_health_score}/100"
    )

elif section_health_score >= 65:

    st.info(
        f"✅ Good Fresher Resume — "
        f"{section_health_score}/100"
    )

elif section_health_score >= 50:

    st.warning(
        f"⚠️ Resume Needs Improvement — "
        f"{section_health_score}/100"
    )

else:

    st.error(
        f"❌ Resume Needs Significant Improvement — "
        f"{section_health_score}/100"
    )


st.progress(
    int(
        section_health_score
    )
)


section_names = list(
    section_results.keys()
)

section_columns = (
    st.columns(4)
)


for index, section_name in enumerate(
    section_names
):

    result = (
        section_results[
            section_name
        ]
    )

    section_columns[
        index
    ].metric(
        section_name,
        f"{result['score']}/{result['max']}",
    )


st.subheader(
    "💡 Section Improvement Suggestions"
)


for (
    section_name,
    result
) in section_results.items():

    with st.expander(
        f"{section_name} "
        f"({result['score']}/{result['max']})"
    ):

        if result["tips"]:

            for tip in result["tips"]:

                st.write(
                    f"• {tip}"
                )

        else:

            st.success(
                f"✅ {section_name} looks strong."
            )


section_chart_df = (
    pd.DataFrame(
        {
            "Section":
                list(
                    section_results.keys()
                ),

            "Percentage":
                [
                    round(
                        (
                            item["score"]
                            / item["max"]
                        ) * 100,
                        1
                    )
                    for item
                    in section_results.values()
                ],
        }
    )
    .set_index(
        "Section"
    )
)


st.subheader(
    "📈 Fresher Resume Section Comparison"
)

st.bar_chart(
    section_chart_df
)


# ============================================================
# JOB FIT DASHBOARD
# ============================================================

if jd_skills:

    job_skill_ratio = (
        len(matched_skills)
        / len(jd_skills)
    )

else:

    job_skill_ratio = 0.0


job_skill_score = (
    job_skill_ratio * 40
)


project_health = (
    section_results[
        "Projects"
    ]["score"]
)

project_max = (
    section_results[
        "Projects"
    ]["max"]
)

project_score = (
    project_health
    / project_max
) * 30


job_similarity_score = (
    similarity * 20
)


education_health = (
    section_results[
        "Education"
    ]["score"]
)

education_max = (
    section_results[
        "Education"
    ]["max"]
)

education_jobfit_score = (
    education_health
    / education_max
) * 10


job_fit_score = round(
    min(
        job_skill_score
        + project_score
        + job_similarity_score
        + education_jobfit_score,
        100,
    ),
    1,
)


st.header(
    "🎯 Fresher Job Fit Dashboard"
)

st.caption(
    "Estimated alignment for an entry-level role based on "
    "skills, projects, wording and education presentation."
)


if job_fit_score >= 80:

    st.success(
        f"🔥 Strong Job Fit — "
        f"{job_fit_score}/100"
    )

elif job_fit_score >= 65:

    st.info(
        f"✅ Good Job Fit — "
        f"{job_fit_score}/100"
    )

elif job_fit_score >= 50:

    st.warning(
        f"⚠️ Moderate Job Fit — "
        f"{job_fit_score}/100"
    )

else:

    st.error(
        f"❌ Low Job Fit — "
        f"{job_fit_score}/100"
    )


st.progress(
    int(
        job_fit_score
    )
)


fit1, fit2, fit3, fit4 = (
    st.columns(4)
)

fit1.metric(
    "🛠️ Technical Skills",
    f"{job_skill_score:.1f}/40"
)

fit2.metric(
    "🚀 Projects",
    f"{project_score:.1f}/30"
)

fit3.metric(
    "🎯 Resume ↔ JD",
    f"{job_similarity_score:.1f}/20"
)

fit4.metric(
    "🎓 Education",
    f"{education_jobfit_score:.1f}/10"
)


job_fit_df = (
    pd.DataFrame(
        {
            "Category": [
                "Technical Skills",
                "Projects",
                "Resume ↔ JD",
                "Education"
            ],

            "Percentage": [

                (
                    job_skill_score
                    / 40
                ) * 100,

                (
                    project_score
                    / 30
                ) * 100,

                (
                    job_similarity_score
                    / 20
                ) * 100,

                (
                    education_jobfit_score
                    / 10
                ) * 100,
            ],
        }
    )
    .set_index(
        "Category"
    )
)


st.subheader(
    "📊 Job Fit Breakdown"
)

st.bar_chart(
    job_fit_df
)


st.subheader(
    "💡 Job Fit Insights"
)


if not jd_skills:

    st.write(
        "• ⚠️ Few predefined technical skills were detected "
        "in the job description, so skill-based scoring "
        "may be conservative."
    )

elif job_skill_score < 24:

    st.write(
        "• 🛠️ Technical skill alignment is one "
        "of the main areas to improve."
    )

else:

    st.write(
        "• ✅ Your technical skills show useful "
        "alignment with this role."
    )


if project_score < 18:

    st.write(
        "• 🚀 Strengthen project descriptions and clearly "
        "show technologies genuinely used."
    )

else:

    st.write(
        "• ✅ Your projects provide useful evidence "
        "of practical skills."
    )


if job_similarity_score < 10:

    st.write(
        "• 🎯 Align resume wording more closely with "
        "the job description where truthful."
    )

else:

    st.write(
        "• ✅ Your resume has reasonable contextual "
        "alignment with the job description."
    )


# ============================================================
# INTERVIEW READINESS
# ============================================================

summary_health = (
    section_results[
        "Professional Summary"
    ]["score"]
)

summary_max = (
    section_results[
        "Professional Summary"
    ]["max"]
)


interview_readiness = round(
    min(
        (
            job_skill_ratio
            * 40
        )
        + (
            project_health
            / project_max
            * 30
        )
        + (
            similarity
            * 20
        )
        + (
            summary_health
            / summary_max
            * 10
        ),
        100,
    ),
    1,
)


st.header(
    "🎤 Interview Preparation Readiness"
)

st.caption(
    "An estimated preparation indicator based on resume evidence, "
    "not a prediction of interview success."
)


if interview_readiness >= 80:

    st.success(
        f"🔥 Strong Preparation Base — "
        f"{interview_readiness}/100"
    )

elif interview_readiness >= 65:

    st.info(
        f"✅ Good Preparation Base — "
        f"{interview_readiness}/100"
    )

elif interview_readiness >= 50:

    st.warning(
        f"⚠️ More Preparation Recommended — "
        f"{interview_readiness}/100"
    )

else:

    st.error(
        f"📚 Significant Preparation Recommended — "
        f"{interview_readiness}/100"
    )


st.progress(
    int(
        interview_readiness
    )
)


# ============================================================
# AI RESUME OPTIMIZER
# ============================================================

st.header(
    "🛠️ AI Resume Optimizer"
)

st.write(
    "Generate job-specific improvements using only information "
    "already supported by your resume."
)

st.caption(
    "Missing skills may be listed as skills to learn, "
    "but they must not be inserted as if you already know them."
)


optimizer_mode = (
    st.selectbox(
        "Choose what you want to optimize",
        [
            "Complete Fresher Resume",
            "Professional Summary",
            "Technical Skills",
            "Projects",
            "Education",
        ],
        key="resume_optimizer_mode",
    )
)


if st.button(
    "🚀 Optimize My Resume",
    key="optimize_resume_button"
):

    optimizer_prompt = f"""
You are an expert ATS-friendly resume optimizer
specializing in fresher and entry-level resumes.

The candidate is a FRESHER.

TARGET OPTIMIZATION:
{optimizer_mode}

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

ATS COMPATIBILITY SCORE:
{ats_score}/100

FRESHER RESUME HEALTH SCORE:
{section_health_score}/100

MATCHED SKILLS:
{matched_skills}

MISSING SKILLS:
{missing_skills}

MATCHED KEYWORDS:
{matched_keywords}

MISSING KEYWORDS:
{missing_keywords}

STRICT RULES:

- Never invent professional work experience.
- Never invent skills.
- Never invent certifications.
- Never invent achievements.
- Never invent companies.
- Never invent education.
- Never fabricate project features.
- Never fabricate percentages, numbers or outcomes.
- Never claim a project used a technology unless
  the original resume supports it.
- Missing skills may appear only under Skills to Learn.
- Keep wording concise, professional and ATS-friendly.
- Focus on Professional Summary, Technical Skills,
  Projects and Education.

If TARGET OPTIMIZATION is "Complete Fresher Resume",
use exactly these sections:

## Optimized Professional Summary

Write a strong 3-4 sentence fresher summary
using only supported facts.

## Optimized Technical Skills

Reorganize only skills already present
in the resume into clear categories.

## Improved Project Bullets

Rewrite only existing projects using stronger
action-oriented wording.

Do not invent features or results.

## Optimized Education Section

Present only the existing education
information clearly.

## Skills to Learn

List important missing skills from the
job description.

State that they should be added to the resume
only after the candidate genuinely learns
and demonstrates them.

## Final ATS Improvement Tips

Give practical, truthful improvements
for this target job.

If TARGET OPTIMIZATION is a single section,
optimize only that section and do not create
unrelated resume content.
"""

    try:

        with st.spinner(
            "🛠️ Optimizing your resume..."
        ):

            (
                optimized_content,
                optimizer_model
            ) = generate_ai_response(
                optimizer_prompt
            )


        st.session_state[
            "optimized_resume_content"
        ] = optimized_content

        st.session_state[
            "optimizer_model_used"
        ] = optimizer_model

        st.session_state[
            "optimizer_mode_used"
        ] = optimizer_mode


    except Exception as exc:

        st.error(
            "❌ Resume optimization is temporarily unavailable. "
            "Please try again."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(exc)
            )


optimized_output = (
    st.session_state.get(
        "optimized_resume_content"
    )
)


if optimized_output:

    st.success(
        "✅ Resume Optimization Generated"
    )

    st.caption(
        "Model used: "
        + st.session_state.get(
            "optimizer_model_used",
            "Gemini"
        )
    )

    st.markdown(
        optimized_output
    )


    st.download_button(
        "⬇️ Download Optimization Suggestions",
        data=optimized_output,
        file_name=(
            "optimized_resume_suggestions.txt"
        ),
        mime="text/plain",
        key="download_optimized_resume",
    )


    # ========================================================
    # BEFORE VS AFTER ATS
    # ========================================================

    if (
        st.session_state.get(
            "optimizer_mode_used"
        )
        == "Complete Fresher Resume"
    ):

        st.header(
            "📊 Before vs After ATS Comparison"
        )

        st.caption(
            "The optimized score is a projected compatibility "
            "estimate based on the AI-generated rewrite, "
            "not an official ATS result."
        )


        resume_ready_content = (
            optimized_output
        )

        cut_positions = []


        for heading in [

            "## Skills to Learn",
            "## Final ATS Improvement Tips"

        ]:

            position = (
                resume_ready_content.find(
                    heading
                )
            )

            if position != -1:

                cut_positions.append(
                    position
                )


        if cut_positions:

            resume_ready_content = (
                resume_ready_content[
                    :min(
                        cut_positions
                    )
                ]
            )


        projected = (
            calculate_ats_score_for_text(
                resume_ready_content,
                job_description,
            )
        )


        improvement = round(
            projected["total"]
            - ats_score,
            1
        )


        compare1, compare2, compare3 = (
            st.columns(3)
        )


        compare1.metric(
            "📄 Original Resume",
            f"{ats_score}/100"
        )


        compare2.metric(
            "✨ Projected Optimized",
            f"{projected['total']}/100"
        )


        compare3.metric(
            "📈 Estimated Improvement",
            f"{improvement:+.1f} points"
        )


        comparison_df = pd.DataFrame(
            {
                "Category": [
                    "Skills Match",
                    "Keyword Match",
                    "Resume ↔ JD Similarity",
                    "Resume Structure",
                ],

                "Original": [
                    ats["skills"],
                    ats["keywords"],
                    ats["similarity"],
                    ats["structure"],
                ],

                "Projected Optimized": [
                    projected["skills"],
                    projected["keywords"],
                    projected["similarity"],
                    projected["structure"],
                ],
            }
        )


        st.dataframe(
            comparison_df,
            hide_index=True,
            width="stretch"
        )


        st.bar_chart(
            comparison_df
            .set_index(
                "Category"
            )[
                [
                    "Original",
                    "Projected Optimized"
                ]
            ]
        )


        if improvement >= 10:

            st.success(
                f"🔥 Strong projected improvement: "
                f"+{improvement:.1f} points."
            )

        elif improvement > 0:

            st.info(
                f"✅ Projected improvement: "
                f"+{improvement:.1f} points."
            )

        elif improvement == 0:

            st.warning(
                "The rewrite did not materially "
                "change the projected score."
            )

        else:

            st.warning(
                "The projected score decreased. "
                "Review the AI wording before using it."
            )


# ============================================================
# AI RESUME REVIEW
# ============================================================

st.header(
    "🤖 AI Fresher Resume Review"
)

st.caption(
    "Gemini reviews the resume only after you click the button. "
    "The app does not ask Gemini to invent an ATS score."
)


if st.button(
    "✨ Generate AI Resume Review",
    key="generate_resume_review"
):

    review_prompt = f"""
You are a professional resume reviewer specializing
in fresher and entry-level candidates.

The candidate is a FRESHER.

Do not penalize the candidate for having no
professional work experience.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

ATS SCORE:
{ats_score}/100

FRESHER RESUME HEALTH SCORE:
{section_health_score}/100

JOB FIT SCORE:
{job_fit_score}/100

MATCHED SKILLS:
{matched_skills}

MISSING SKILLS:
{missing_skills}

MATCHED KEYWORDS:
{matched_keywords}

MISSING KEYWORDS:
{missing_keywords}

Use exactly these sections:

## Overall Assessment

Explain how suitable the fresher resume
is for the target role.

## Resume Strengths

Highlight genuine strengths in skills,
projects and education.

## Missing Skills & Keywords

Explain important gaps.

Never claim the candidate knows a missing skill.

## Professional Summary Improvements

Suggest truthful improvements.

## Technical Skills Improvements

Suggest organization and genuine
alignment improvements.

## Project Improvements

Explain how existing projects can be
presented more clearly.

Do not invent features.

## Education Improvements

Suggest clearer presentation of
existing academic information.

## Optimized Professional Summary

Write a concise 3-4 sentence fresher
summary using only facts supported
by the resume.

## Interview Preparation

Generate 5 likely interview questions
based on the actual resume and target job.

STRICT RULES:

- Never invent experience.
- Never invent skills.
- Never invent certifications.
- Never invent achievements.
- Never invent companies.
- Never invent education.
- Never invent project features.
- Never invent numbers.
- Do not demand professional experience.
- Use only the supplied resume and job description.
"""

    try:

        with st.spinner(
            "🤖 AI is reviewing your fresher resume..."
        ):

            (
                ai_review,
                review_model
            ) = generate_ai_response(
                review_prompt
            )


        st.session_state[
            "ai_review_content"
        ] = ai_review

        st.session_state[
            "ai_review_model"
        ] = review_model


    except Exception as exc:

        st.error(
            "❌ AI review is temporarily unavailable. "
            "Please try again."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(exc)
            )


if st.session_state.get(
    "ai_review_content"
):

    st.success(
        "✅ AI Fresher Resume Review Generated"
    )

    st.caption(
        "Model used: "
        + st.session_state.get(
            "ai_review_model",
            "Gemini"
        )
    )

    st.markdown(
        st.session_state[
            "ai_review_content"
        ]
    )


# ============================================================
# AI INTERVIEW PREPARATION
# ============================================================

st.header(
    "🎤 AI Interview Preparation"
)

st.write(
    "Generate a fresher interview preparation pack based on "
    "your actual resume and the target job."
)


if st.button(
    "🎤 Generate My Interview Questions",
    key="generate_interview_questions"
):

    interview_prompt = f"""
You are an interview coach specializing
in fresher and entry-level technology candidates.

The candidate is a FRESHER.

RESUME:
{resume_text}

TARGET JOB DESCRIPTION:
{job_description}

MATCHED SKILLS:
{matched_skills}

MISSING SKILLS:
{missing_skills}

JOB FIT SCORE:
{job_fit_score}/100

INTERVIEW PREPARATION READINESS:
{interview_readiness}/100

Create a realistic interview preparation pack
using exactly these sections:

## Technical Questions

Generate 5 questions based only on technical
skills genuinely present in the resume.

## Project Questions

Generate 5 questions about projects actually
present in the resume.

Focus on:

- Project purpose
- Technologies used
- Candidate contribution
- Problems faced
- Technical decisions

Do not invent project features.

## Job-Specific Questions

Generate 5 questions related to
the job description.

If a topic is missing from the resume,
label it as an area to prepare rather than
claiming the candidate knows it.

## Fresher HR Questions

Generate 5 suitable fresher HR questions.

## Most Important Topics to Revise

Separate into:

High Priority

Medium Priority

## Project Explanation Strategy

Explain how the candidate can present
actual projects in an interview.

## 60-Second Self Introduction Structure

Create a structure using only details
supported by the resume.

STRICT RULES:

- Do not demand professional work experience.
- Never invent skills.
- Never invent projects.
- Never invent project features.
- Never invent achievements.
- Never invent experience.
- Never claim the candidate knows a missing skill.
"""

    try:

        with st.spinner(
            "🎤 Preparing your interview pack..."
        ):

            (
                interview_content,
                interview_model
            ) = generate_ai_response(
                interview_prompt
            )


        st.session_state[
            "interview_content"
        ] = interview_content

        st.session_state[
            "interview_model"
        ] = interview_model


    except Exception as exc:

        st.error(
            "❌ Interview preparation is temporarily unavailable. "
            "Please try again."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(exc)
            )


if st.session_state.get(
    "interview_content"
):

    st.success(
        "✅ Interview Preparation Pack Generated"
    )

    st.caption(
        "Model used: "
        + st.session_state.get(
            "interview_model",
            "Gemini"
        )
    )

    st.markdown(
        st.session_state[
            "interview_content"
        ]
    )


    st.download_button(
        "⬇️ Download Interview Preparation",
        data=st.session_state[
            "interview_content"
        ],
        file_name=(
            "fresher_interview_preparation.txt"
        ),
        mime="text/plain",
        key="download_interview_pack",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Resume Analyzer • Fresher-focused portfolio project • "
    "Python • Streamlit • scikit-learn • Gemini"
)
