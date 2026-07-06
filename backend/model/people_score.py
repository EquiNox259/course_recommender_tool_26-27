import pandas as pd
import numpy as np
import json
import re
import pickle
import ast
import os
from collections import defaultdict
import math

# 1. Establish absolute baseline directories relative to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

# 2. Map directly to the actual files visible in your dataset folder
AUTUMN_COURSES_PATH = os.path.join(DATASET_DIR, "courses_autumn_2025-2026.csv")
SPRING_COURSES_PATH = os.path.join(DATASET_DIR, "courses_spring_2025-2026.csv")

STUDENT_DATA_PATHS = [
    os.path.join(DATASET_DIR, "courseTaken.csv"),             # 2024-25
    os.path.join(DATASET_DIR, "courseTaken_2025-26.csv"),     # 2025-26
    os.path.join(DATASET_DIR, "courseTaken2023_.csv")         # 2023-24
]
OUTPUT_PICKLE_PATH = os.path.join(CURRENT_DIR, "people_score_matrix.pkl")
def infer_student_department(student_courses):
    """
    Infer a student's department from their first-year courses.

    Returns:
        "MM", "ME", "EE", ... or None
    """

    courses = set(student_courses)

    # -------------------------
    # Unique department courses
    # -------------------------

    if "MM 105" in courses:
        return "MM"

    if "ME 103" in courses or "ME 104" in courses:
        return "ME"

    if "EE 103" in courses or "EE 114" in courses:
        return "EE"

    if "CS 105" in courses or "CS 108" in courses:
        return "CS"

    if "CE 102" in courses or "CE 103" in courses:
        return "CE"

    if "CL 102" in courses:
        return "CL"

    if "IE 101" in courses or "IE 102" in courses:
        return "IE"

    if "EN 110" in courses:
        return "EN"

    if "AE 103" in courses or "AE 152" in courses or "AE 153" in courses:
        return "AE"

    # -------------------------
    # Special cases
    # -------------------------

    # Environmental Science
    if "ES 101" in courses:
        return "ES"

    # Physics
    if "PH 109" in courses:
        return "PH"

    # Mathematics
    if (
        "MA 105" in courses
        and "CH 111" not in courses
    ):
        return "MA"

    # Chemistry
    if "CH 111" in courses:
        return "CH"



# Exact list of allowed 1xx introductory courses
INTRO_COURSES = {
    "AE 103", "AE 152", "AE 153", "CE 102", "CE 103", "CL 102", 
    "CS 105", "CS 108", "DE 109", "DE 110", "DE 113", "DE 115", 
    "EE 103", "EE 114", "EN 110", "ENT101", "ES 101", "GP 101", 
    "HS 110", "HS 111", "HS 113", "IE 101", "IE 102", "ME 103", 
    "ME 104", "MM 105", "SOM101"
}

DEPT_TO_DIC = {
    "AE": "AE 103",
    "CE": "CE 103",
    "CL": "CL 102",
    "CH": "CH 105",
    "CH": "CH 111",
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

def get_external_popular_courses(student_department, semester, top_k=10):
    """
    Returns courses most popular among students OUTSIDE the course's own department.
    Filters out courses belonging to the requesting student's department.
    """
    semester = semester.lower()
    
    with open(OUTPUT_PICKLE_PATH, "rb") as f:
        package = pickle.load(f)

    data = package[semester]
    external_popularity = data["external_popularity"]
    recommendations = []

    for course_code, popularity in external_popularity.items():
        course_department = course_code[:3].strip()
        # Exclude courses that belong to the student's own department
        if course_department == student_department:
            continue
        recommendations.append((course_code, popularity))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_k]

def should_keep_course(course_code):
    """
    Removes all 1xx courses EXCEPT the specified department introductory courses.
    """
    intro_courses_normalized = {
        c.replace(" ", "").upper()
        for c in INTRO_COURSES
    }
    code = re.sub(r"\s+", "", course_code).upper()
    match = re.search(r"(\d+)", code)
    if not match:
        return True
    number = int(match.group(1))
    if 100 <= number < 200:
        return code in intro_courses_normalized
    return True

def clean_student_course_string(course_str):
    """
    Converts strings like:
        'PH 227-2025-1 AI and Data Science' -> PH 227
        'ENT101-2025'                       -> ENT101
        'SOM101-2025'                       -> SOM101
        'NOCS01-2025'                       -> NOCS01
        'MM 105'                            -> MM 105
    """

    if not isinstance(course_str, str):
        return None

    course_str = course_str.strip().upper()

    # Remove semester suffixes like -2025, -2026, etc.
    course_str = re.sub(r"-20\d{2}.*$", "", course_str)

    # Standard format: "CS 101", "PH 227", etc.
    m = re.match(r"^([A-Z]{2,4})\s+(\d+[A-Z]?)$", course_str)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    # Compact format: ENT101, SOM101, NOCS01, GNR650, etc.
    m = re.match(r"^([A-Z]{2,6})(\d+[A-Z]?)$", course_str)
    if m:
        return f"{m.group(1)}{m.group(2)}"

    return None

def extract_master_courses(metadata_path):
    if not os.path.exists(metadata_path):
        print(f"WARNING: Metadata sheet not found at {metadata_path}")
        return []

    df = pd.read_csv(metadata_path)
    df.columns = df.columns.str.replace(r'^\ufeff', '', regex=True).str.strip()

<<<<<<< Updated upstream
   # FIX 1: Clean hidden BOM characters and strip extra whitespace from column headers
    courses_df.columns = courses_df.columns.str.lstrip('\ufeff').str.strip()
    
    # PRINT DEBUG: Let's see exactly what Pandas sees for columns
    print(f"Detected columns in metadata: {list(courses_df.columns)}")
    
    # FIX 2: Normalize known variations back to 'Course Code'
=======
>>>>>>> Stashed changes
    column_mapping = {
        "code": "Course Code",
        "course code": "Course Code",
        "Course code": "Course Code",
        "COURSE CODE": "Course Code"
    }
    df.rename(columns=column_mapping, inplace=True)

    if "Course Code" not in df.columns:
        print(f"ERROR: Could not locate 'Course Code' column in {os.path.basename(metadata_path)}.")
        return []

    return sorted(df["Course Code"].astype(str).str.strip().unique().tolist())

def compute_matrices(master_courses, student_data_paths):
    num_courses = len(master_courses)
    course_to_idx = {code: i for i, code in enumerate(master_courses)}
    idx_to_course = {i: code for i, code in enumerate(master_courses)}
    
    course_enrollments = {code: set() for code in master_courses}

    course_department_counts = {
        code: defaultdict(int)
        for code in master_courses
    }    
    sophomore_popularity = defaultdict(lambda: defaultdict(int))
    # FIXED: Added default dict tracking to properly compute external popularity
    intro_course_popularity = defaultdict(lambda: defaultdict(int))
    for file_path in student_data_paths:
        if not os.path.exists(file_path):
            print(f"  Skipping missing history file: {os.path.basename(file_path)}")
            continue

        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            email = str(row.iloc[0]).strip()
            if not email:
                continue
            # Extract student dept for external popularity logic
            student_dept = None
            student_courses = []
            # -----------------------
            # Read every course first
            # -----------------------
            for raw_course in row.iloc[1:-1]:
                if pd.isna(raw_course):
                    continue
                clean_code = clean_student_course_string(str(raw_course))
                if clean_code is None:
                    continue
                if not should_keep_course(clean_code):
                    continue
                student_courses.append(clean_code)
            # -----------------------
            # Find the DIC
            # -----------------------
            student_dept = infer_student_department(student_courses)

            if student_dept is None:
                continue
            # -----------------------
            # Populate enrollments
            # -----------------------
            for clean_code in student_courses:
                if clean_code in course_enrollments:
                    course_enrollments[clean_code].add(email)
                    course_department_counts[clean_code][student_dept] += 1
            # -----------------------
            # Sophomore popularity
            # -----------------------
            for clean_code in student_courses:
                match = re.search(r"(\d+)", clean_code)
                if match and 100 <= int(match.group(1)) < 200:
                    continue
                sophomore_popularity[student_dept][clean_code] += 1

    # Matrix Initialization
    intersection_matrix = np.zeros((num_courses, num_courses), dtype=float)
    union_matrix = np.zeros((num_courses, num_courses), dtype=float)
    similarity_matrix = np.zeros((num_courses, num_courses), dtype=float)

    for i in range(num_courses):
        code_i = idx_to_course[i]
        set_i = course_enrollments[code_i]

        intersection_matrix[i, i] = len(set_i)
        union_matrix[i, i] = len(set_i)
        similarity_matrix[i, i] = 1.0 if len(set_i) > 0 else 0.0

        for j in range(i + 1, num_courses):
            code_j = idx_to_course[j]
            set_j = course_enrollments[code_j]

            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))

            intersection_matrix[i, j] = intersection
            intersection_matrix[j, i] = intersection
            union_matrix[i, j] = union
            union_matrix[j, i] = union
            
            similarity_matrix[i, j] = intersection / union if union > 0 else 0.0
            similarity_matrix[j, i] = similarity_matrix[i, j]

    external_popularity = {}
    for course in master_courses:
        own_dept = course[:3].strip()
        total = 0
        for dept, count in course_department_counts[course].items():
            if dept != own_dept:
                total += count
        external_popularity[course] = total
    
    print("Students detected by department:")
    for dept, courses in sophomore_popularity.items():
        total = sum(courses.values())
        print(dept, total)
        print(dept, len(courses))

    return {
        "course_to_idx": course_to_idx,
        "idx_to_course": idx_to_course,
        "intersection_matrix": intersection_matrix,
        "union_matrix": union_matrix,
        "similarity_matrix": similarity_matrix,
        "external_popularity": external_popularity,
        "sophomore_popularity": {
            dept: dict(courses)
            for dept, courses in sophomore_popularity.items()
        }
    }

def main():
    print("--- FIXED PATH INITIALIZATION ---")
    autumn_courses = extract_master_courses(AUTUMN_COURSES_PATH)
    spring_courses = extract_master_courses(SPRING_COURSES_PATH)
    
    print(f"Tracking {len(autumn_courses)} Autumn courses and {len(spring_courses)} Spring courses.")

    print("\nProcessing Autumn Semester matrices...")

    print("Processing Spring Semester matrices...")

    master_courses = sorted(
        set(autumn_courses) | set(spring_courses)
    )
    pickle_package = compute_matrices(
        master_courses,
        STUDENT_DATA_PATHS
    )

    with open(OUTPUT_PICKLE_PATH, "wb") as f:
        pickle.dump(pickle_package, f)

    print(f"\n[SUCCESS] Matrices saved successfully to: {OUTPUT_PICKLE_PATH}")

if __name__ == "__main__":
    main()