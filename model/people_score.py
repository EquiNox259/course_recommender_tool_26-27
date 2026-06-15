import pandas as pd
import numpy as np
import json
import re
import pickle
import ast
import os

# 1. Establish absolute baseline directories relative to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# 2. Hard-map directly to the actual files visible in your dataset folder
COURSES_DATA_PATH = os.path.join(ROOT_DIR, "dataset", "Courses25-26_2_new.csv")
STUDENT_DATA_PATH = os.path.join(ROOT_DIR, "dataset", "Student_data.csv")
OUTPUT_PICKLE_PATH = os.path.join(CURRENT_DIR, "people_score_matrix.pkl")

def clean_student_course_string(course_str):
    """
    Parses structural strings like "'PH 227-2025-1 AI and Data Science'"
    and extracts the clean course code prefix: "PH 227".
    """
    if not isinstance(course_str, str) or not course_str.strip():
        return None
    
    # Matches alphanumeric department prefixes followed by whitespace and a course number
    # Examples: "PH 227", "SOM 101", "MA 105", "GC_101"
    match = re.match(r'^\s*([A-Z0-9_]+[-_ ]+\d+[A-Z]?)', course_str.strip())
    if match:
        clean_code = match.group(1).replace('_', ' ').strip()
        # Ensure a uniform single-space format between text prefix and digits
        clean_code = re.sub(r'\s+', ' ', clean_code)
        return clean_code
    return None

def main():
    global COURSES_DATA_PATH
    print("--- FIXED PATH INITIALIZATION ---")
    print(f"Target Metadata: {COURSES_DATA_PATH}")
    print(f"Target Student Data: {STUDENT_DATA_PATH}")
    print("---------------------------------\n")
    
    # Quick structural check before reading
    if not os.path.exists(COURSES_DATA_PATH):
        print(f"ERROR: Cannot find metadata sheet. Trying alternative name...")
        # Fallback to the other new file variant if Courses_new isn't the primary one
        COURSES_DATA_PATH = os.path.join(ROOT_DIR, "dataset", "Courses25-26_2_new.csv")
        if not os.path.exists(COURSES_DATA_PATH):
            print("CRITICAL: Neither course file variant was found.")
            return  
    if not os.path.exists(COURSES_DATA_PATH):
        print(f"CRITICAL ERROR: Metadata file missing at {COURSES_DATA_PATH}")
        return
    
    print("Step 1: Extracting master course codes...")
    # Load all unique valid course codes from your core courses metadata sheet
    courses_df = pd.read_csv(COURSES_DATA_PATH)

   # FIX 1: Clean hidden BOM characters and strip extra whitespace from column headers
    courses_df.columns = courses_df.columns.str.replace(r'^\ufeff', '', regex=True).str.strip()
    
    # PRINT DEBUG: Let's see exactly what Pandas sees for columns
    print(f"Detected columns in metadata: {list(courses_df.columns)}")
    
    # FIX 2: Normalize known variations back to 'Course Code'
    column_mapping = {
        'code': 'Course Code',
        'course code': 'Course Code',
        'Course code': 'Course Code',
        'COURSE CODE': 'Course Code'
    }
    courses_df.rename(columns=column_mapping, inplace=True)
    
    # Ultimate defensive safety check
    if "Course Code" not in courses_df.columns:
        print("\nCRITICAL ERROR: Could not find the course identifier column.")
        print(f"Available columns are: {list(courses_df.columns)}")
        print("Please check which column holds the codes (e.g., 'AE 152') and map it.")
        return

    
    master_courses = sorted(courses_df["Course Code"].str.strip().unique().tolist())
    num_courses = len(master_courses)
    
    # Create mapping indices for fast matrix tracking lookup arrays
    course_to_idx = {code: i for i, code in enumerate(master_courses)}
    idx_to_course = {i: code for i, code in enumerate(master_courses)}
    
    print(f"-> Tracked {num_courses} unique master courses from metadata.")

    print("\nStep 2: Processing student enrollment histories...")
    student_df = pd.read_csv(STUDENT_DATA_PATH)
    
    # Dictionary tracking: course_code -> set of student emails
    course_enrollments = {code: set() for code in master_courses}
    
    for idx, row in student_df.iterrows():
        email = str(row["Emails"]).strip()
        raw_courses_field = row["Courses"]
        
        if pd.isna(raw_courses_field) or not email:
            continue
            
        # Safely parse the python list string representation out of the CSV cell
        try:
            if isinstance(raw_courses_field, str) and raw_courses_field.startswith('['):
                raw_course_list = ast.literal_eval(raw_courses_field)
            else:
                raw_course_list = [raw_courses_field]
        except Exception:
            # Fallback split if literal_eval hits corrupted quote layouts
            raw_course_list = [c.strip("[]'\" ") for c in str(raw_courses_field).split(',')]

        for raw_item in raw_course_list:
            clean_code = clean_student_course_string(raw_item)
            if clean_code in course_enrollments:
                course_enrollments[clean_code].add(email)

    print("\nStep 3: Calculating intersection and union enrollment matrices...")
    # Initialize empty mathematical tracking spaces
    intersection_matrix = np.zeros((num_courses, num_courses), dtype=float)
    union_matrix = np.zeros((num_courses, num_courses), dtype=float)
    similarity_matrix = np.zeros((num_courses, num_courses), dtype=float)

    for i in range(num_courses):
        code_i = idx_to_course[i]
        set_i = course_enrollments[code_i]
        
        # Self similarity identity optimization
        intersection_matrix[i, i] = len(set_i)
        union_matrix[i, i] = len(set_i)
        similarity_matrix[i, i] = 1.0 if len(set_i) > 0 else 0.0
        
        for j in range(i + 1, num_courses):
            code_j = idx_to_course[j]
            set_j = course_enrollments[code_j]
            
            # Run set logic operations
            intersection_count = len(set_i.intersection(set_j))
            union_count = len(set_i.union(set_j))
            
            intersection_matrix[i, j] = intersection_count
            intersection_matrix[j, i] = intersection_count
            
            union_matrix[i, j] = union_count
            union_matrix[j, i] = union_count
            
            # Compute Jaccard Similarity J(H, C) = |H ∩ C| / |H ∪ C|
            if union_count > 0:
                score = float(intersection_count) / float(union_count)
                similarity_matrix[i, j] = score
                similarity_matrix[j, i] = score
            else:
                similarity_matrix[i, j] = 0.0
                similarity_matrix[j, i] = 0.0

    print("\nStep 4: Compiling matrix metadata package...")
    # Wrap all matrix artifacts inside a clean dictionary package for pickle ingestion
    pickle_package = {
        "course_to_idx": course_to_idx,
        "idx_to_course": idx_to_course,
        "intersection_matrix": intersection_matrix,
        "union_matrix": union_matrix,
        "similarity_matrix": similarity_matrix
    }

    print(f"Saving pickle structural map payload to: {OUTPUT_PICKLE_PATH}...")
    with open(OUTPUT_PICKLE_PATH, "wb") as f:
        pickle.dump(pickle_package, f)
        
    print("\n[SUCCESS] People score similarity matrices constructed successfully!")

if __name__ == "__main__":
    main()