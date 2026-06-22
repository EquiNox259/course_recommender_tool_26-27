import os
import json
import pickle
import traceback
import numpy as np
import pandas as pd
import faiss
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from config import MODEL_ASSETS_DIR, MODEL_DATA_DIR
from model.llm_service import LLMService

MODEL_PATH = os.path.join(MODEL_ASSETS_DIR, "course_encoder_new")
FAISS_INDEX_PATH = os.path.join(MODEL_DATA_DIR, "course_index_new.faiss")
COURSE_META_PATH = os.path.join(MODEL_DATA_DIR, "courses_metadata_new.csv")

# Load the people score matrix asset globally when the server boots up
PEOPLE_SCORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "people_score_matrix.pkl")

try:
    with open(PEOPLE_SCORE_PATH, "rb") as f:
        ps_matrix_data = pickle.load(f)
    print("[SUCCESS] Loaded People Score similarity matrix asset.")
except Exception as e:
    print(f"[WARNING] Could not load People Score matrix: {e}")
    ps_matrix_data = None

model = SentenceTransformer(MODEL_PATH)
model_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

index = faiss.read_index(FAISS_INDEX_PATH)

# CRITICAL FIX: Keep df_courses completely unfiltered for pristine FAISS index mapping
df_courses = pd.read_csv(COURSE_META_PATH)

# Pre-parse core courses into an uppercase lookup set
df_core = pd.read_csv("dataset/ASC_Core_Courses.csv")
core_course_codes = set(df_core["Course Code"].dropna().astype(str).str.strip().str.upper())


def calculate_people_score(student_history: list, target_course_code: str) -> float:
    """
    Computes the decayed People Score (FS) for a candidate elective course
    based on a student's taken course codes.
    """
    if not ps_matrix_data or not student_history:
        return 0.0
        
    course_to_idx = ps_matrix_data["course_to_idx"]
    similarity_matrix = ps_matrix_data["similarity_matrix"]
    
    # Standardize codes to matching lookup keys
    clean_target = str(target_course_code).strip().upper()
    if clean_target not in course_to_idx:
        return 0.0
        
    target_idx = course_to_idx[clean_target]
    final_score = 0.0
    
    clean_history = [str(c).strip().upper() for c in student_history if c]
    
    for i, hist_course in enumerate(reversed(clean_history), start=1):
        if hist_course in course_to_idx:
            hist_idx = course_to_idx[hist_course]
            ps_h_c = similarity_matrix[hist_idx, target_idx]
            final_score += ps_h_c / (1.0 + (i - 1))
            
    return final_score


def get_semantic_rankings(query, student_history=None, top_k=50):
    """Runs raw FAISS semantic search and matches against unfiltered rows before exclusions."""
    if student_history is None:
        student_history = []
        
    query_emb = model.encode(query, normalize_embeddings=True).astype(np.float32)
    
    # Pull an expanded candidate window to ensure we satisfy top_k after core exclusions
    expansion_factor = top_k + len(core_course_codes) + len(student_history)
    scores, indices = index.search(query_emb.reshape(1, -1), min(expansion_factor, len(df_courses)))
   
    # FIX: Map via df_courses which perfectly retains positional alignment with FAISS indexes
    semantic_results = df_courses.iloc[indices[0]].copy()
    
    # Build complete exclusions criteria (Core courses + Student academic history)
    exclusions = core_course_codes.union(set(str(c).strip().upper() for c in student_history if c))
    
    filtered_results = semantic_results[
        ~semantic_results["Course Code"].astype(str).str.strip().str.upper().isin(exclusions)
    ]
    
    return filtered_results["Course Code"].tolist()[:top_k]


def get_keyword_rankings(primary_keywords, secondary_keywords, student_history=None, top_k=50):
    """Runs phrase-boundary keyword matching using LLM-extracted n-grams on core-excluded data."""
    if student_history is None:
        student_history = []
        
    # Apply exclusions to separate our elective pool dynamically from global catalog
    exclusions = core_course_codes.union(set(str(c).strip().upper() for c in student_history if c))
    df_electives = df_courses[
        ~df_courses["Course Code"].astype(str).str.strip().str.upper().isin(exclusions)
    ].copy()
    
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


def get_candidate_courses(query, student_history=None, top_k=40, w_rrf=0.0, w_ps=1.0):
    """Unified entrypoint called by app.py."""
    llm = LLMService()       
    clean_query = query.lower().strip()

    print(f"[PRE-PROCESSING] Original Query: '{query}'")
    try:
        processed_query = llm.rephrase_and_extract_intent(query)

        print(f"[PRE-PROCESSING] LLM Optimized Query: '{processed_query}'")

        # NEW: Garbage-query short circuit
        if not processed_query.get("is_valid", True):
            print(
                f"[QUERY REJECTED] {processed_query.get('reject_reason', 'Invalid query')}"
            )
            w_rrf = 0.0
            w_ps = 1.0
            clean_query = ""  # Override to empty to prevent any semantic/keyword matches

        semantic_query = processed_query.get("combined", [])
        primary_keywords = processed_query.get("primary", [])
        secondary_keywords = processed_query.get("secondary", [])

        print(f"[PRE-PROCESSING] LLM Optimized Query: '{processed_query}'")
    except Exception as e:
        print(f"[WARNING] LLM query optimization failed: {e}. Falling back to baseline baseline query.")
        processed_query = {
            "is_valid": True,
            "combined": [clean_query],
            "primary": [clean_query],
            "secondary": []
        }

        semantic_query = processed_query["combined"]
        primary_keywords = processed_query["primary"]
        secondary_keywords = processed_query["secondary"]

    if student_history is None:
        student_history = []

    try:
        # =====================================================================
        # STREAM 1: RRF -> PS
        # =====================================================================
        description_lookup = (
            df_courses
            .set_index("Course Code")["Description"]
            .to_dict()
        )

        if clean_query:
            # Convert the array of structured phrases back into a unified contextual string for the semantic models
            semantic_query_text = " ".join(processed_query["combined"])
            
            # Update the rankings call to use text for semantic search, and pass arrays directly to keyword search
            semantic_list = get_semantic_rankings(semantic_query_text, student_history, top_k=50)
            keyword_list = get_keyword_rankings(processed_query["primary"], processed_query["secondary"], student_history, top_k=50)
            
            all_fused_candidates = compute_rrf(semantic_list, keyword_list, k=60)
            if not all_fused_candidates:
                return []
        else:
            all_fused_candidates = []

        candidates_enriched = []
        rrf_vals = []
        ps_vals = []

        for code, rrf_score in all_fused_candidates:
            clean_code = str(code).strip().upper()
            ps_score = calculate_people_score(student_history, clean_code)
            
            rrf_vals.append(rrf_score if clean_query else 0.0)
            ps_vals.append(ps_score)
            
            row = df_courses[df_courses["Course Code"].astype(str).str.strip().str.upper() == clean_code].iloc[0]
            candidates_enriched.append({
                "code": clean_code,
                "name": str(row["Course Name"]),
                "raw_rrf": rrf_score if clean_query else 0.0,
                "raw_ps": ps_score
            })

        for candidate in candidates_enriched:
            course_text = (
                candidate["name"] + " " + description_lookup.get(candidate["code"], "")
            )
            semantic_query_text = " ".join(semantic_query)
            score = model_cross_encoder.predict(
                [[semantic_query_text, course_text]]
            )[0]
            candidate["cross_encoder_score"] = score
        
        candidates_enriched = sorted(candidates_enriched, key=lambda x: x["cross_encoder_score"], reverse=True)
        candidates_enriched = candidates_enriched[:top_k]

        text_fused_dict = {str(code).strip().upper(): score for code, score in all_fused_candidates}     

        # =====================================================================
        # STREAM 2: PS -> RRF (With Dynamic Fallback Calculation)
        # =====================================================================
        if w_ps > 0.0:    
            exclusions = core_course_codes.union(set(str(c).strip().upper() for c in student_history if c))
            df_electives = df_courses[~df_courses["Course Code"].astype(str).str.strip().str.upper().isin(exclusions)]
            
            ps_vals_2 = {}
            for code in df_electives["Course Code"].unique():
                clean_code = str(code).strip().upper()
                ps_vals_2[clean_code] = calculate_people_score(student_history, clean_code)
                
            top_50_ps = sorted(ps_vals_2.items(), key=lambda item: item[1], reverse=True)[:50]
            master_pool = {c["code"]: c for c in candidates_enriched}

            for code, calculated_ps in top_50_ps:
                clean_code = str(code).strip().upper() 
                
                if clean_code in master_pool:
                    master_pool[clean_code]["raw_ps"] = calculated_ps
                    # Sync score into tracker array for proper Min-Max bounds calculation
                    ps_vals.append(calculated_ps)
                else:
                    row = df_courses[df_courses["Course Code"].astype(str).str.strip().str.upper() == clean_code].iloc[0]
                    
                    if clean_code in text_fused_dict:
                        calculated_rrf = text_fused_dict[clean_code]
                    elif clean_query:
                        single_semantic = [
                            item for item in get_semantic_rankings(
                                semantic_query_text,
                                student_history,
                                top_k=150
                            )
                            if str(item).strip().upper() == clean_code
                        ]
                        single_keyword = [item for item in get_keyword_rankings(primary_keywords, secondary_keywords, student_history, top_k=150) if str(item).strip().upper() == clean_code]
                        fused_single = compute_rrf(single_semantic, single_keyword, k=60)
                        calculated_rrf = fused_single[0][1] if fused_single else 0.0
                        text_fused_dict[clean_code] = calculated_rrf
                    else:
                        calculated_rrf = 0.0
                    
                    rrf_vals.append(calculated_rrf)
                    ps_vals.append(calculated_ps)
                    
                    master_pool[clean_code] = {
                        "code": clean_code,
                        "name": str(row["Course Name"]),
                        "raw_rrf": calculated_rrf,
                        "raw_ps": calculated_ps
                    }

            candidates_enriched = list(master_pool.values())
        # =====================================================================
        # SCALE SCORE NORMALIZATION
        # =====================================================================
        valid_rrf_vals = [c["raw_rrf"] for c in candidates_enriched if c["raw_rrf"] > 0.0]
        valid_ps_vals = [c["raw_ps"] for c in candidates_enriched if c["raw_ps"] > 0.0]

        max_rrf, min_rrf = (max(valid_rrf_vals), min(valid_rrf_vals)) if valid_rrf_vals else (1.0, 0.0)
        max_ps, min_ps = (max(valid_ps_vals), min(valid_ps_vals)) if valid_ps_vals else (1.0, 0.0)

        for c in candidates_enriched:
            # If a course has 0 raw_rrf, its normalized score must be 0, don't let it scale up
            if c["raw_rrf"] == 0.0:
                norm_rrf = 0.0
            else:
                norm_rrf = (c["raw_rrf"] - min_rrf) / (max_rrf - min_rrf) if max_rrf != min_rrf else 1.0
                
            if c["raw_ps"] == 0.0:
                norm_ps = 0.0
            else:
                norm_ps = (c["raw_ps"] - min_ps) / (max_ps - min_ps) if max_ps != min_ps else 0.0
            
            c["combined_score"] = (w_rrf * norm_rrf) + (w_ps * norm_ps)
            c["norm_rrf"] = norm_rrf
            c["norm_ps"] = norm_ps
            print(f"[CHECKPOINT 2] Course: {c['code']} -> Formula: ({w_rrf} * {norm_rrf:.4f}) + ({w_ps} * {norm_ps:.4f}) = {c['combined_score']:.4f}")
        candidates_enriched = sorted(candidates_enriched, key=lambda x: x["combined_score"], reverse=True)
        
        llm_input_pool = candidates_enriched[:20]

    except Exception as e:
        print(f"\n[CRITICAL LOCAL PIPELINE EXCEPTION]: {e}")
        traceback.print_exc()
        return []

    # =====================================================================
    # STEP 3: LLM RELEVANCE FILTERING & RESPONSES
    # =====================================================================
    try:
        if clean_query:
            cleaned_json_string = llm.filter_courses(processed_query, llm_input_pool)
            parsed_data = json.loads(cleaned_json_string)
            valid_codes = set(str(c).strip().upper() for c in parsed_data.get("valid_course_codes", []))
        else:
            valid_codes = set(c["code"] for c in llm_input_pool)
    

        final_output = []
        for c in llm_input_pool:
            if c["code"] in valid_codes:
                final_output.append({
                    "code": c["code"],
                    "name": c["name"],
                    "raw_rrf": c["raw_rrf"],
                    "raw_ps": c["raw_ps"],
                    "norm_rrf": c["norm_rrf"],  
                    "norm_ps": c["norm_ps"], 
                    "raw_ts" : c["combined_score"]
                })
                if len(final_output) == top_k:
                    break
        return final_output

    except Exception as e:
        print(f"\n[WARNING - GEMINI FILTER FAILED]: {e}. Falling back to pre-filtered rank pool.")
        fallback_output = []
        for c in llm_input_pool:
            fallback_output.append({
                "code": c["code"],
                "name": c["name"],
                "raw_rrf": c["raw_rrf"],
                "raw_ps": c["raw_ps"],
                "norm_rrf": c["norm_rrf"],  
                "norm_ps": c["norm_ps"], 
                "raw_ts" : c["combined_score"]
            })
        return fallback_output
