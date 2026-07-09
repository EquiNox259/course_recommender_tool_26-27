import os
import json
import pickle
import traceback
import numpy as np
import pandas as pd
import faiss
import re
from sentence_transformers import SentenceTransformer
from model.people_score import clean_student_course_string
from config import MODEL_ASSETS_DIR, MODEL_DATA_DIR, RUNNING_COURSES_PATH, RUNNING_COURSES_PATH_ALT, CORE_COURSES_PATH, MINOR_COURSES_PATH, SEMESTER
from model.llm_service import LLMService
import time


MODEL_PATH = os.path.join(MODEL_ASSETS_DIR, "course_encoder_new")

if SEMESTER.lower() == 'autumn':
    FAISS_INDEX_PATH = os.path.join(MODEL_DATA_DIR, "course_index_autumn.faiss")
    COURSE_META_PATH = os.path.join(MODEL_DATA_DIR, "courses_metadata_autumn.csv")
else:
    FAISS_INDEX_PATH = os.path.join(MODEL_DATA_DIR, "course_index_new.faiss")
    COURSE_META_PATH = os.path.join(MODEL_DATA_DIR, "courses_metadata_new.csv")

# Load the people score matrix asset globally when the server boots up
PEOPLE_SCORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "people_score_matrix.pkl")
DEPARTMENT_TO_CODE = {
    "Aerospace Engineering": "AE",
    "Computer Science and Engineering": "CS",
    "Chemical Engineering": "CL",
    "Chemistry": "CH",
    "Civil Engineering": "CE",
    "Economics": "EC",
    "Electrical Engineering": "EE",
    "Energy Science and Engineering": "EN",
    "Environmental Science and Engineering": "ES",
    "Applied Geophysics": "GP",
    "Industrial Engineering and Operations Research": "IE",
    "Mathematics": "MA",
    "Mechanical Engineering": "ME",
    "Metallurgical Engineering and Materials Science": "MM",
    "Physics": "PH",
}

CODE_TO_DEPT: dict = {
    'AE':    'Aerospace Engineering',
    'BB':    'Biosciences and Bioengineering',
    'CE':    'Civil Engineering',
    'CL':    'Chemical Engineering',
    'CS':    'Computer Science and Engineering',
    'CH':    'Chemistry',
    'EE':    'Electrical Engineering',
    'EN':    'Energy Science and Engineering',
    'EP':    'Engineering Physics',
    'ES':    'Environmental Science and Engineering',
    'ESE':   'Environmental Science and Engineering',
    'GNR':   'Centre of Studies in Resources Engineering',
    'GP':    'Applied Geophysics',
    'HSS':   'Humanities & Social Science',
    'IE':    'Industrial Engineering and Operations Research',
    'IEOR':  'Industrial Engineering and Operations Research',
    'MA':    'Mathematics',
    'ME':    'Mechanical Engineering',
    'MEMS':  'Metallurgical Engineering and Materials Science',
    'MM':    'Metallurgical Engineering and Materials Science',
    'PH':    'Physics',
    'SC':    'Systems and Control',
    'SOM':   'Shailesh J. Mehta School of Management',
}

# ── Comprehensive remark-abbreviation → full department name ──────────────
_DEPT_TO_ABBREVS: dict = {}
for _abbr, _full in CODE_TO_DEPT.items():
    _DEPT_TO_ABBREVS.setdefault(_full, set()).add(_abbr)


def _sentence_applies(sentence: str, student_dept: str) -> bool:
    s = sentence.strip()
    positive = set(re.findall(r'\b([A-Z]{2,6})\s+students\b', s)) & set(CODE_TO_DEPT)
    negated  = set(re.findall(r'[Nn]on-([A-Z]{2,6})', s)) & set(CODE_TO_DEPT)
    positive -= negated
    if not positive and not negated:
        return True
    student_abbrevs = _DEPT_TO_ABBREVS.get(student_dept)
    if not student_abbrevs:
        return True
    if negated and not positive:
        return not bool(student_abbrevs & negated)
    if positive and not negated:
        return bool(student_abbrevs & positive)
    return True


def parse_minor_remark(remark: str, course_hist: list, department: str = '') -> dict:
    out = {'prereq_unmet': None, 'warning': None, 'note': None}
    if not remark or str(remark).strip().lower() in ('nan', 'none', ''):
        return out
    remark = str(remark).strip()

    sentences = re.split(r'(?<=\.)\s+', remark)
    kept = [s for s in sentences if _sentence_applies(s, department)]
    if not kept:
        return out
    remark = ' '.join(kept)
    rl = remark.lower()

    prereq_m = re.search(
        r'([A-Z]{2,3}\s?\d{3,4})\s+(?:is\s+(?:a\s+)?)?(?:mandatory\s+)?prerequisite\s+for'
        r'|must\s+(?:first\s+)?complete\s+([A-Z]{2,3}\s?\d{3,4})\s+before'
        r'|([A-Z]{2,3}\s?\d{3,4})\s+is\s+(?:a\s+)?pre-?req(?:uisite)?',
        remark, re.IGNORECASE
    )
    if prereq_m:
        codes = re.findall(r'[A-Z]{2,3}\s?\d{3,4}', prereq_m.group(0))
        norm_hist = {c.replace(' ', '').upper() for c in course_hist}
        missing = [c for c in codes if c.replace(' ', '').upper() not in norm_hist]
        if missing:
            out['prereq_unmet'] = f"Minor requirement not met. Complete {', '.join(missing)} first"
        out['note'] = remark
        return out

    if re.search(r'cannot\s+be\s+counted|not\s+be\s+counted|excluded|not\s+allowed|cannot\s+count|will\s+not\s+count', rl):
        out['warning'] = remark
        return out

    out['note'] = remark
    return out


def _flatten_minor_remark(parsed: dict) -> str:
    """Collapse the classified remark back to a single display string,
    prioritising the most actionable message. Keeps the existing
    frontend contract (c.minor_remark as a plain string) unchanged."""
    return parsed.get('prereq_unmet') or parsed.get('warning') or parsed.get('note') or ''

DEPT_TO_DIC = {
    "AE": "AE 103",
    "CE": "CE 103",
    "CL": "CL 102",
    "CH": "CH 105",
    "CS": "CS 108",
    "EE": "EE 103",
    "EN": "EN 110",
    "ES": "ES 101",
    "IE": "IE 101",
    "GP": "GP 101",
    "MA": "MA 105",
    "ME": "ME 103",
    "MM": "MM 105",
    "PH": "PH 110",
}

try:
    with open(PEOPLE_SCORE_PATH, "rb") as f:
        semester = "autumn" 
        ps_matrix_data = pickle.load(f)
        department_popularity = ps_matrix_data["external_popularity"]
    print("[SUCCESS] Loaded People Score similarity matrix asset.")
except Exception as e:
    print(f"[WARNING] Could not load People Score matrix: {e}")
    ps_matrix_data = None
    department_popularity = None

model = SentenceTransformer(MODEL_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)

def extract_department(course_code):
    code = re.sub(r'\s+', '', str(course_code)).upper()
    m = re.match(r'^([A-Z]+)\d+', code)
    return m.group(1) if m else None

df_minor_courses = pd.read_csv(MINOR_COURSES_PATH)
df_minor_courses["Branch"] = (
    df_minor_courses["Branch"]
    .astype(str)
    .str.strip()
)
df_minor_courses["Course Code"] = (
    df_minor_courses["Course Code"]
    .astype(str)
    .str.strip()
    .str.upper()
)
minor_lookup = (
    df_minor_courses
    .groupby("Branch")["Course Code"]
    .apply(lambda x: set(x.str.strip().str.upper()))
    .to_dict()
)

minor_type_lookup = (
    df_minor_courses
    .assign(Type=df_minor_courses["Type"].fillna("").astype(str).str.strip())
    .groupby("Branch")
    .apply(lambda g: dict(zip(g["Course Code"], g["Type"])))
    .to_dict()
)

all_minor_courses = set(
    df_minor_courses["Course Code"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
)

minor_remark_lookup = (
    df_minor_courses
    .assign(
        **{
            "Course Code": df_minor_courses["Course Code"]
                .astype(str)
                .str.strip()
                .str.upper(),
            "Remarks": df_minor_courses["Remarks"]
                .fillna("")
                .astype(str)
                .str.strip()
        }
    )
    .groupby("Branch")
    .apply(lambda g: dict(zip(g["Course Code"], g["Remarks"])))
    .to_dict()
)
df_courses = pd.read_csv(COURSE_META_PATH)

df_courses.columns = (
    df_courses.columns
    .str.lstrip('\ufeff')
    .str.strip()
)
column_mapping = {
    "code": "Course Code",
    "course code": "Course Code",
    "Course code": "Course Code",
    "COURSE CODE": "Course Code",
    "description": "Description",
    "course description": "Description",
    "Course description": "Description"
}

df_courses.rename(columns=column_mapping, inplace=True)

df_courses["Department"] = df_courses["Course Code"].apply(extract_department)

df_courses["Course Code"] = (
    df_courses["Course Code"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df_core = pd.read_csv(CORE_COURSES_PATH)

core_course_codes = set(
    df_core["Course Code"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
)

description_lookup = (
    df_courses
    .set_index("Course Code")["Description"]
    .fillna("")
    .to_dict()
)

def build_candidate_pool(student_history=None,
                         include_departments=None,
                         exclude_departments=None,
                         exclude_courses=None,
                         minor = None,
                         degree = None,
                         department = None
                         ):
    if student_history is None:
        student_history = []

    own_core_codes = set()
    if department and degree:
        dept_core = df_core[
            (df_core['Branch'] == department) &
            (df_core['Degree'] == degree)
        ]
        own_core_codes = {
            str(c).strip().upper()
            for c in dept_core['Course Code']
            if pd.notna(c)
        }

    exclusions = own_core_codes.union(
        {
            str(c).strip().upper()
            for c in student_history
            if c
        }
    )

    minor = {
        m.strip()
        for m in (minor or [])
        if m
    }
    include_departments = {
        d.strip().upper()
        for d in (include_departments or [])
    }
    exclude_departments = {
        d.strip().upper()
        for d in (exclude_departments or [])
    }
    exclude_courses = {
        c.strip().upper().replace(" ", "")
        for c in (exclude_courses or [])
    }
    pool = df_courses[
        ~df_courses["Course Code"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(exclusions)
    ].copy()

    # Include department filter (extract prefix ignoring spaces e.g. "SOM" from "SOM101")
    if include_departments:
        pool = pool[
            pool["Course Code"]
            .str.extract(r'^([A-Za-z]+)', expand=False)
            .str.upper()
            .isin(include_departments)
        ]
    # Exclude department filter
    if exclude_departments:
        pool = pool[
            ~pool["Course Code"]
            .str.extract(r'^([A-Za-z]+)', expand=False)
            .str.upper()
            .isin(exclude_departments)
        ]

    # Exclude individual courses
    if exclude_courses:
        pool = pool[
            ~pool["Course Code"]
            .str.replace(" ", "", regex=False)
            .str.upper()
            .isin(exclude_courses)
        ]
    if minor:
        allowed_courses = set()
        for m in minor:
            allowed_courses |= minor_lookup.get(m, set())

        allowed_courses_norm = {
            c.replace(" ", "").upper()
            for c in allowed_courses
            if c
        }

        pool = pool[
            pool["Course Code"]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.upper()
            .isin(allowed_courses_norm)
        ]
    return pool


def calculate_people_score(
    student_history,
    target_course_code,
    student_department,
    year=None,
):
    """
    If use_department_popularity=True (recommended for browse mode),
    returns department popularity instead of similarity score.

    target_course_code can be:
        - "CS 419"
        - ["CS 419", "EE 782", ...]
    """
    if "department_popularity" in ps_matrix_data:
        dp = ps_matrix_data["department_popularity"]

    if not ps_matrix_data:
        if isinstance(target_course_code, (list, tuple, set)):
            return {c: 0.0 for c in target_course_code}
        return 0.0

    # Sophomore browse mode
    if year == "2025":
        dept_pop = ps_matrix_data.get("sophomore_popularity", {}).get(student_department, {})
        def popularity(course):
            clean = str(course).strip().upper()
            score = dept_pop.get(clean, 0)
            if dept_pop:
                max_score = max(dept_pop.values())
                if max_score > 0:
                    score /= max_score
            return float(score)

        if isinstance(target_course_code, (list, tuple, set)):
            return {
                str(c).strip().upper(): popularity(c)
                for c in target_course_code
            }
        return popularity(target_course_code)
    # ---------------------------------------------------------
    # ORIGINAL PEOPLE SCORE
    # ---------------------------------------------------------
    if not student_history:
        if isinstance(target_course_code, (list, tuple, set)):
            return {c: 0.0 for c in target_course_code}
        return 0.0

    course_to_idx = ps_matrix_data["course_to_idx"]
    similarity_matrix = ps_matrix_data["similarity_matrix"]

    def score_one(course):
        clean_target = str(course).strip().upper()

        if clean_target not in course_to_idx:
            return 0.0

        target_idx = course_to_idx[clean_target]
        final_score = 0.0

        clean_history = []

        for c in student_history:
            cleaned = clean_student_course_string(c)
            if cleaned:
                clean_history.append(cleaned)

        for i, hist_course in enumerate(reversed(clean_history), start=1):
            if hist_course in course_to_idx:
                hist_idx = course_to_idx[hist_course]
                sim = similarity_matrix[hist_idx, target_idx]
                final_score += sim / i

        return float(final_score)

    # ---------- list input ----------
    if isinstance(target_course_code, (list, tuple, set)):
        return {
            str(c).strip().upper(): score_one(c)
            for c in target_course_code
        }

    # ---------- single course ----------
    return score_one(target_course_code)


def get_semantic_rankings(query, student_history = None, candidate_pool = None, top_k=80):
    """Runs raw FAISS semantic search and matches against unfiltered rows before exclusions."""
    if student_history is None:
        student_history = []
        
    query_emb = model.encode(query, normalize_embeddings=True).astype(np.float32)
    
    # Pull an expanded candidate window to ensure we satisfy top_k after core exclusions
    expansion_factor = top_k + len(core_course_codes) + len(student_history)
    scores, indices = index.search(query_emb.reshape(1, -1), min(expansion_factor, len(df_courses)))
   
    # FIX: Map via df_courses which perfectly retains positional alignment with FAISS indexes
    semantic_results = df_courses.iloc[indices[0]].copy()

    allowed_codes = set(
        candidate_pool["Course Code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    semantic_results = semantic_results[
        semantic_results["Course Code"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(allowed_codes)
    ]
    
    # Build complete exclusions criteria (Core courses + Student academic history)

    
    return semantic_results["Course Code"].tolist()[:top_k]


def get_keyword_rankings(primary_keywords, secondary_keywords, student_history = None, candidate_pool = None, top_k=40):
    """Runs phrase-boundary keyword matching using LLM-extracted n-grams on core-excluded data."""
    if student_history is None:
        student_history = []
    
    df_electives = candidate_pool.copy()
    
    if df_electives.empty:
        return []
    
    match_score = np.zeros(len(df_electives))

    # Expecting lists directly from get_candidate_courses
    query_tiers = [
        {"phrases": primary_keywords, "weight_multiplier": 3.0},
        {"phrases": secondary_keywords, "weight_multiplier": 1.0}
    ]
    
    for tier in query_tiers:
        phrases = tier["phrases"]
        multiplier = tier["weight_multiplier"]
        
        for phrase in phrases:
            # Clean and normalize spaces within the n-gram phrase
            clean_phrase = re.sub(r'\s+', ' ', phrase.lower().strip())
            if not clean_phrase:
                continue
            
            # Escape the phrase for regex safeness but enforce word boundaries around the phrase
            # This correctly matches "machine learning" as a complete block
            pattern = rf"\b{re.escape(clean_phrase)}\b"
            
            # Match against Course Code (Base 3.0) scaled by tier multiplier
            match_score += df_electives["Course Code"].str.lower().str.contains(pattern, na=False, regex=True).astype(float) * 3.0 * multiplier
            
            # Match against Course Name (Base 2.0) scaled by tier multiplier
            match_score += df_electives["Course Name"].str.lower().str.contains(pattern, na=False, regex=True).astype(float) * 2.0 * multiplier
            
            # Match against Course Description (Base 1.0) scaled by tier multiplier
            if "Course Description" in df_electives.columns:
                match_score += df_electives["Course Description"].str.lower().str.contains(pattern, na=False, regex=True).astype(float) * 1.0 * multiplier

    # Assign the calculated composite score back to the DataFrame
    df_electives["keyword_score"] = match_score
    df_filtered = df_electives[df_electives["keyword_score"] > 0]

    if df_filtered.empty:
        return []

    # Sort and return the top_k course codes
    keyword_results = df_filtered.sort_values(by="keyword_score", ascending=False)
    return keyword_results["Course Code"].tolist()[:top_k]
    
def compute_rrf(semantic_codes, keyword_codes, k=60):
    """Applies Reciprocal Rank Fusion formula to merge input streams."""
    rrf_scores = {}
    for rank, code in enumerate(semantic_codes):
        clean_code = str(code).strip().upper()
        rrf_scores[clean_code] = rrf_scores.get(clean_code, 0.0) + (1.0 / (k + (rank + 1)))

    for rank, code in enumerate(keyword_codes):
        clean_code = str(code).strip().upper()
        rrf_scores[clean_code] = rrf_scores.get(clean_code, 0.0) + (1.0 / (k + (rank + 1)))

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def get_candidate_courses(query, student_history=None, top_k=40, w_rrf=0.0, w_ps=1.0, degree=None, year=None, department=None, processed_query=None):
    """Unified entrypoint called by app.py."""
    llm = LLMService()
    query = " ".join(CODE_TO_DEPT.get(w.upper(), w) for w in query.split())   # expand dept codes
    clean_query = query.lower().strip()
    student_dept = DEPARTMENT_TO_CODE.get(
        str(department).strip(),
        str(department).strip().upper()
    )

    print(f"[PRE-PROCESSING] Original Query: '{query}'")
    try:
        if processed_query is None and clean_query:
            processed_query = llm.rephrase_and_extract_intent(query)
            print("intent detected")
        if processed_query is None and not clean_query:
            processed_query = {
                "is_valid": True,
                "combined": [],
                "primary": [],
                "secondary": [],
                "expanded": [],
                "constraints": {
                    "include_departments": [],
                    "exclude_departments": [],
                    "exclude_courses": [],
                    "minor": [],
                    "easy_grading": False,
                    "popular": False
                }
            }
        
        print(f"[PRE-PROCESSING] LLM Optimized Query: '{processed_query}'")

        # Garbage-query short circuit
        if not processed_query.get("is_valid", True):
            print(
                f"[QUERY REJECTED] {processed_query.get('reject_reason', 'Invalid query')}"
            )
            return [], False, processed_query.get("reject_reason", "Invalid query"), False, {}
    
        semantic_query = processed_query.get("combined", [])
        primary_keywords = processed_query.get("primary", [])
        secondary_keywords = processed_query.get("secondary", [])
        expanded_keywords = processed_query.get("expanded", [])
        constraints = processed_query.get("constraints") or {}
        include_courses = constraints.get("include_courses", [])
        minor = constraints.get("minor", [])
        include_departments = constraints.get("include_departments", [])
        exclude_courses = constraints.get("exclude_courses", [])
        exclude_departments = constraints.get("exclude_departments", [])
        easy_grading = constraints.get("easy_grading", False)
        popular = constraints.get("popular", False)

        print(f"[PRE-PROCESSING] LLM Optimized Query: '{processed_query}'")
    except Exception as e:
        print(f"[WARNING] LLM query optimization failed: {e}. Falling back to baseline query.")
        processed_query = {
            "is_valid": True,
            "combined": [clean_query],
            "primary": [clean_query],
            "secondary": [],
            "expanded": [],
            "constraints": {
                "include_departments": [],
                "exclude_departments": [],
                "exclude_courses": [],
                "minor": [],
                "easy_grading": False,
                "popular": False
            }
        }
        is_valid = processed_query["is_valid"]
        semantic_query = processed_query["combined"]
        primary_keywords = processed_query["primary"]
        secondary_keywords = processed_query["secondary"]
        expanded_keywords = processed_query["expanded"]
        constraints = processed_query["constraints"]
        minor = constraints.get("minor", [])
        easy_grading = constraints.get("easy_grading", False)
        popular = constraints.get("popular", False)
    
    candidate_pool = build_candidate_pool(
        student_history=student_history,
        include_departments=include_departments,
        exclude_departments=exclude_departments,
        exclude_courses=exclude_courses,
        minor=minor,
        degree=degree,
        department=department
    )

    is_minor_query = bool(minor)

    # Curriculum-mandated Type ('Minor'/'Elective') per requested minor branch,
    # merged across all requested branches for this query.
    requested_minor_types = {}
    for m in minor:
        for k, v in minor_type_lookup.get(m.strip(), {}).items():
            requested_minor_types[k.replace(" ", "")] = v

    requested_minor_remarks = {}
    for m in minor:
        requested_minor_remarks.update(minor_remark_lookup.get(m.strip(), {}))

    if student_history is None:
        student_history = []
    
    try:
        description_lookup = (
            df_courses
            .set_index("Course Code")["Description"]
            .to_dict()
        )

        # 1. Generate the initial candidate list (RRF of semantic & keyword search)
        if primary_keywords:
            semantic_query_text = " ".join(
                processed_query["combined"] +
                processed_query["expanded"]
            )
            
            semantic_list = get_semantic_rankings(semantic_query_text, student_history, candidate_pool, top_k=100)
            keyword_list = get_keyword_rankings(
                processed_query["primary"],
                processed_query["secondary"],
                student_history,
                candidate_pool,
                top_k=100
            )
            all_fused_candidates = compute_rrf(semantic_list, keyword_list, k=60)[:100]
        else:
            all_fused_candidates = []

        # Dynamic weights assignment
        if not primary_keywords:
            if easy_grading and popular:
                w_rrf = 0.00
                w_ps = 0.50
                w_grade = 0.50
            elif popular:
                w_rrf = 0.00
                w_ps = 1.00
                w_grade = 0.00
            elif easy_grading:
                w_rrf = 0.00
                w_ps = 0.00
                w_grade = 1.00
            else:
                # Default empty search: 100% peer history (same as baseline)
                w_rrf = 0.00
                w_ps = 1.00
                w_grade = 0.00
        else:
            if easy_grading and popular:
                w_rrf = 0.60
                w_ps = 0.20
                w_grade = 0.20
            elif popular:
                w_rrf = 0.70
                w_ps = 0.30
                w_grade = 0.00
            elif easy_grading:
                w_rrf = 0.70
                w_ps = 0.00
                w_grade = 0.30
            else:
                w_rrf = 1.00
                w_ps = 0.00
                w_grade = 0.00

        # Build initial candidate pool
        candidates_enriched = []
        seen_codes = set()
        
        # Helper to get course details
        course_lookup = (
            df_courses
            .assign(
                _code=lambda x:
                x["Course Code"]
                .astype(str)
                .str.strip()
                .str.upper()
            )
            .drop_duplicates("_code")                
            .set_index("_code")
            .to_dict("index")
        )

        if primary_keywords:
            for code, rrf_score in all_fused_candidates:
                clean_code = str(code).strip().upper()
                if clean_code in seen_codes:
                    continue
                seen_codes.add(clean_code)
                
                ps_score = calculate_people_score(student_history, clean_code, student_dept, year) if (w_ps > 0.0 or not primary_keywords) else 0.0
                
                row = course_lookup.get(clean_code)
                if row:
                    candidates_enriched.append({
                        "code": clean_code,
                        "name": str(row["Course Name"]),
                        "description": description_lookup.get(clean_code, ""),
                        "raw_rrf": rrf_score,
                        "raw_ps": ps_score,
                    })

            # 2. Enrich with top popular courses if w_ps > 0 (popular is active)
            if w_ps > 0.0:
                exclusions = core_course_codes.union(set(str(c).strip().upper() for c in student_history if c))
                df_electives = df_courses[~df_courses["Course Code"].astype(str).str.strip().str.upper().isin(exclusions)]
                
                ps_vals_2 = {}
                for code in df_electives["Course Code"].unique():
                    clean_code = str(code).strip().upper()
                    # Check if it is in candidate_pool
                    if clean_code in candidate_pool["Course Code"].astype(str).str.strip().str.upper().values:
                        ps_vals_2[clean_code] = calculate_people_score(student_history, clean_code, student_dept, year)
                    
                top_50_ps = sorted(ps_vals_2.items(), key=lambda item: item[1], reverse=True)[:50]

                master_pool = {c["code"]: c for c in candidates_enriched}
                text_fused_dict = {str(code).strip().upper(): score for code, score in all_fused_candidates}

                for code, calculated_ps in top_50_ps:
                    clean_code = str(code).strip().upper()
                    if clean_code in master_pool:
                        master_pool[clean_code]["raw_ps"] = calculated_ps
                    else:
                        row = course_lookup.get(clean_code)
                        if row:
                            calculated_rrf = text_fused_dict.get(clean_code, 0.0)
                            master_pool[clean_code] = {
                                "code": clean_code,
                                "name": str(row["Course Name"]),
                                "description": description_lookup.get(clean_code, ""),
                                "raw_rrf": calculated_rrf,
                                "raw_ps": calculated_ps
                            }
                candidates_enriched = list(master_pool.values())
        else:
            # If no primary keywords (e.g. "cminds minor courses"), populate from candidate_pool directly (all of them)
            for code in candidate_pool["Course Code"].unique():
                clean_code = str(code).strip().upper()
                row = course_lookup.get(clean_code)
                if row:
                    ps_score = calculate_people_score(student_history, clean_code, student_dept, year) if (w_ps > 0.0 or not primary_keywords) else 0.0
                    candidates_enriched.append({
                        "code": clean_code,
                        "name": str(row["Course Name"]),
                        "description": description_lookup.get(clean_code, ""),
                        "raw_rrf": 0.0,
                        "raw_ps": ps_score
                    })

        # 3. Apply standard filters (remove completed, restricted, and LLM constraints)
        from eligibility.rules import check_restriction
        filtered_candidates = []
        seen = set()
        history_set = {str(ch).strip().upper() for ch in student_history if ch}
        exclude_depts = set(d.strip().upper() for d in constraints.get("exclude_departments", []) if d)
        exclude_codes = set(str(c).strip().upper() for c in constraints.get("exclude_courses", []) if c)
        exclude_codes_clean = {c.replace(" ", "") for c in exclude_codes}

        for c in candidates_enriched:
            clean_code = c["code"].strip().upper()
            clean_code_nospace = clean_code.replace(" ", "")
            
            # Remove duplicates
            if clean_code in seen:
                continue
            seen.add(clean_code)
            
            # Remove completed
            if clean_code in history_set:
                continue
                
            # Exclude departments and courses
            match_dept = re.match(r'^([A-Z]+)', clean_code)
            if match_dept:
                dept_prefix = match_dept.group(1)
                if dept_prefix in exclude_depts:
                    continue
            if clean_code in exclude_codes or clean_code_nospace in exclude_codes_clean:
                continue
                
            # Restricted courses check
            if degree and year and department:
                r_status_reg = check_restriction(degree, year, department, clean_code, is_minor=False)
                r_status_min = check_restriction(degree, year, department, clean_code, is_minor=True)
                
                def is_candidate_eligible(r):
                    return r == 'Valid' or 'year students' in r or r == 'Restricted'
                
                if not is_candidate_eligible(r_status_reg) and not is_candidate_eligible(r_status_min):
                    continue
            
            filtered_candidates.append(c)
        candidates_enriched = filtered_candidates

        # 4. Integrate grading statistics early & filter
        from eligibility.rules import grade_stats_db
        grade_filter_ran = False
        grade_killed = 0
        initial_pool_len = len(candidates_enriched)

        for c in candidates_enriched:
            clean_code = c["code"].strip().upper()
            clean_code_nospace = clean_code.replace(" ", "")
            # Look up grading stats (fix space-mismatch bug)
            grade_stats = grade_stats_db.get(clean_code_nospace)
            
            if not grade_stats:
                c["avg_grade_score"] = 0.303629  # default neutral
                continue
                
            scores = []
            for _, entries in grade_stats.items():
                for entry in entries:
                    scores.append(entry["score_aa_ab"])
            if not scores:
                c["avg_grade_score"] = 0.303629  # default neutral
            else:
                c["avg_grade_score"] = sum(scores) / len(scores)

        if easy_grading:
            grade_filter_ran = True
            filtered_candidates = []
            for c in candidates_enriched:
                if c["avg_grade_score"] < 0.303629:
                    continue
                filtered_candidates.append(c)
            candidates_enriched = filtered_candidates
            grade_killed = initial_pool_len - len(candidates_enriched)

        # 5. Integrate popularity statistics early & filter
        pop_filter_ran = False
        pop_killed = 0
        post_grade_len = len(candidates_enriched)

        if popular:
            pop_filter_ran = True
            
            # Determine threshold dynamically based on year and department
            if year == "2025":
                dept_pop = ps_matrix_data.get("sophomore_popularity", {}).get(student_dept, {})
                non_zero_pops = [v for v in dept_pop.values() if v > 0]
                threshold = np.percentile(non_zero_pops, 30) if non_zero_pops else 1.0
            else:
                ext_pop = ps_matrix_data.get("external_popularity", {})
                non_zero_pops = [v for v in ext_pop.values() if v > 0]
                threshold = np.percentile(non_zero_pops, 30) if non_zero_pops else 1.0

            filtered_candidates = []
            for c in candidates_enriched:
                clean_code = c["code"].strip().upper()
                if year == "2025":
                    pop_score = ps_matrix_data.get("sophomore_popularity", {}).get(student_dept, {}).get(clean_code, 0)
                else:
                    pop_score = ps_matrix_data.get("external_popularity", {}).get(clean_code, 0)
                
                if pop_score < threshold:
                    continue
                filtered_candidates.append(c)
            candidates_enriched = filtered_candidates
            pop_killed = post_grade_len - len(candidates_enriched)

        # Handle empty candidate pool with dynamic feedback
        if len(candidates_enriched) == 0:
            if grade_killed > 0 and pop_killed > 0:
                reject_reason = "No courses matching your interest satisfy both easy grading and popularity constraints."
            elif grade_killed > 0:
                reject_reason = "No courses matching your interest satisfy the easy grading constraint."
            elif pop_killed > 0:
                reject_reason = "No courses matching your interest satisfy the popularity constraint."
            else:
                reject_reason = "No courses found matching your query. Try a different search interest."
            return [], False, reject_reason, is_minor_query, requested_minor_types

        # 6. Normalize and Rank
        valid_rrf_vals = [c["raw_rrf"] for c in candidates_enriched if c["raw_rrf"] > 0.0]
        valid_ps_vals = [c["raw_ps"] for c in candidates_enriched if c["raw_ps"] > 0.0]
        valid_grade_vals = [c["avg_grade_score"] for c in candidates_enriched if c["avg_grade_score"] > 0.0]

        max_rrf, min_rrf = (max(valid_rrf_vals), min(valid_rrf_vals)) if valid_rrf_vals else (1.0, 0.0)
        max_ps, min_ps = (max(valid_ps_vals), min(valid_ps_vals)) if valid_ps_vals else (1.0, 0.0)
        max_grade, min_grade = (max(valid_grade_vals), min(valid_grade_vals)) if valid_grade_vals else (1.0, 0.0)

        for c in candidates_enriched:
            if c["raw_rrf"] == 0.0:
                norm_rrf = 0.0
            else:
                norm_rrf = (c["raw_rrf"] - min_rrf) / (max_rrf - min_rrf) if max_rrf != min_rrf else 1.0
                
            if c["raw_ps"] == 0.0:
                norm_ps = 0.0
            else:
                norm_ps = (c["raw_ps"] - min_ps) / (max_ps - min_ps) if max_ps != min_ps else 0.0

            if c["avg_grade_score"] == 0.0:
                norm_grade = 0.0
            else:
                norm_grade = (c["avg_grade_score"] - min_grade) / (max_grade - min_grade) if max_grade != min_grade else 1.0
            
            c["norm_rrf"] = norm_rrf
            c["norm_ps"] = norm_ps
            c["norm_grade"] = norm_grade
            
            # Early weighted combined score
            c["combined_score"] = (w_rrf * norm_rrf) + (w_ps * norm_ps) + (w_grade * norm_grade)

        candidates_enriched = sorted(candidates_enriched, key=lambda x: x["combined_score"], reverse=True)
        llm_input_pool = candidates_enriched[:30]

    except Exception as e:
        print(f"\n[CRITICAL LOCAL PIPELINE EXCEPTION]: {e}")
        import traceback
        traceback.print_exc()
        return [], True, "", is_minor_query, requested_minor_types
  
    try:
        if clean_query:
            # If there are no primary keywords, we bypass the LLM filter and return all candidates directly
            if not primary_keywords:
                fallback_output = []
                for c in candidates_enriched:
                    code = c["code"]
                    fallback_output.append({
                        "code": code,
                        "name": c["name"],
                        "description": c["description"],
                        "raw_rrf": c["raw_rrf"],
                        "raw_ps": c["raw_ps"],
                        "norm_rrf": c["norm_rrf"],  
                        "norm_ps": c["norm_ps"], 
                        "raw_ts" : c["combined_score"],
                        "easy_grading": easy_grading,
                        "popular": popular,
                        "minor_remark": _flatten_minor_remark(
                        parse_minor_remark(requested_minor_remarks.get(code, ""), student_history, department))
                    })
                return fallback_output, True, "", is_minor_query, requested_minor_types

            cleaned_json_string = llm.filter_courses(query, processed_query, llm_input_pool)
            parsed_data = json.loads(cleaned_json_string)
            valid_codes = set(str(c).strip().upper() for c in parsed_data.get("valid_course_codes", []))
            
            final_output = []
            for c in llm_input_pool:
                code = c["code"]
                if code in valid_codes:
                    final_output.append({
                        "code": code,
                        "name": c["name"],
                        "description": c["description"],
                        "raw_rrf": c["raw_rrf"],
                        "raw_ps": c["raw_ps"],
                        "norm_rrf": c["norm_rrf"],
                        "norm_ps": c["norm_ps"], 
                        "raw_ts" : c["combined_score"],
                        "easy_grading": easy_grading,
                        "popular": popular,
                        "minor_remark": _flatten_minor_remark(
                        parse_minor_remark(requested_minor_remarks.get(code, ""), student_history, department))
                    })
                    if len(final_output) == top_k:
                        break
            return final_output, True, "", is_minor_query, requested_minor_types
        else:
        # No query text at all -- nothing to filter by, return the ranked pool directly
            fallback_output = []
            for c in candidates_enriched:
                code = c["code"]
                fallback_output.append({
                    "code": code,
                    "name": c["name"],
                    "description": c["description"],
                    "raw_rrf": c["raw_rrf"],
                    "raw_ps": c["raw_ps"],
                    "norm_rrf": c["norm_rrf"],
                    "norm_ps": c["norm_ps"],
                    "raw_ts": c["combined_score"],
                    "easy_grading": easy_grading,
                    "popular": popular,
                    "minor_remark": _flatten_minor_remark(
                        parse_minor_remark(requested_minor_remarks.get(code, ""), student_history, department))
                })
            return fallback_output, True, "", is_minor_query, requested_minor_types

    except Exception as e:
        print(f"\n[WARNING - GEMINI FILTER FAILED]: {e}. Falling back to pre-filtered rank pool.")
        
        fallback_output = []
        for c in llm_input_pool[:top_k]:
            code = c["code"]
            fallback_output.append({
                "code": code,
                "name": c["name"],
                "description": c["description"],
                "raw_rrf": c["raw_rrf"],
                "raw_ps": c["raw_ps"],
                "norm_rrf": c["norm_rrf"],  
                "norm_ps": c["norm_ps"], 
                "raw_ts" : c["combined_score"],
                "easy_grading": easy_grading,
                "popular": popular,
                "minor_remark": _flatten_minor_remark(
                parse_minor_remark(requested_minor_remarks.get(code, ""), student_history, department))
            })
        return fallback_output, True, "", is_minor_query, requested_minor_types
