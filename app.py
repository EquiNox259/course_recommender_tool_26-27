import os
import pandas as pd
import ast
from flask import Flask, render_template, request
from model.recommender import get_candidate_courses
from eligibility.rules import recommender as eligibility_recommender

app = Flask(__name__)

# Load student records once when the server boots to keep lookups fast
STUDENT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "Student_data.csv")

try:
    student_db = pd.read_csv(STUDENT_DATA_PATH)
    # Ensure emails/IDs are stripped for clean matching
    student_db["Emails"] = student_db["Emails"].str.strip()
    print(f"[SUCCESS] Automated Student Database loaded with {len(student_db)} records.")
except Exception as e:
    print(f"[WARNING] Failed to pre-load student database: {e}")
    student_db = None

def fetch_automatic_student_history(student_id):
    """
    Silently look up a student's historical completed courses from the CSV 
    using their LDAP ID / Email prefix.
    """
    if student_db is None or not student_id:
        return []
        
    # Standardize input to match email prefixes (e.g., "24b1814" -> "24b1814@iitb.ac.in")
    search_term = str(student_id).strip().lower()
    if "@" not in search_term:
        search_term = f"{search_term}@iitb.ac.in"
        
    # Query the dataframe
    record = student_db[student_db["Emails"].str.lower() == search_term]
    
    if record.empty:
        print(f"[LOOKUP EMPTY] No pre-existing history found for user: {search_term}")
        return []
        
    raw_courses = record.iloc[0]["Courses"]
    if pd.isna(raw_courses):
        return []
        
    # Safely parse the python list string format out of the CSV cell
    try:
        if isinstance(raw_courses, str) and raw_courses.startswith('['):
            course_list = ast.literal_eval(raw_courses)
        else:
            course_list = [raw_courses]
    except Exception:
        course_list = [c.strip("[]'\" ") for c in str(raw_courses).split(',')]
        
    # Clean up names to get pure base codes (e.g., "MA 105-2024-1-ALL Calculus" -> "MA 105")
    clean_codes = []
    import re
    for item in course_list:
        if not item:
            continue
        match = re.match(r'^\s*([A-Z0-9_]+[-_ ]+\d+[A-Z]?)', str(item).strip())
        if match:
            clean_code = match.group(1).replace('_', ' ').strip()
            clean_codes.append(re.sub(r'\s+', ' ', clean_code))
            
    return list(set(clean_codes)) # return unique codes


@app.route("/", methods=["GET", "POST"])
def index():
    # Establish strict initial baseline defaults
    w_ps = 0.5
    w_rrf = 0.5
    eligible = None
    rejected = None
    i_a_r = None
    course_history = None
    query_fallback = False
    
    if request.method == "POST":
        # --- 1. EXTRACT FROM AND SLIDER PARAMENTERS FIRST ---
        student_id = request.form.get("student_id")
        degree = request.form.get("degree")
        year = request.form.get("year")
        department = request.form.get("department")
        interest = request.form.get("interest")

        # Capture weight inputs from form fields or fallback to default
        w_ps_raw = request.form.get('w_ps') or request.args.get('w_ps')
        w_rrf_raw = request.form.get('w_rrf') or request.args.get('w_rrf')
        

        w_ps = float(w_ps_raw) if w_ps_raw else 0.5
        w_rrf = float(w_rrf_raw) if w_rrf_raw else 0.5

        print(f"\n[CHECKPOINT 1 - APP.PY] Incoming weights extracted from UI:")
        print(f" -> w_ps (Peer History Weight): {w_ps} (Type: {type(w_ps)})")
        print(f" -> w_rrf (Semantic Weight):  {w_rrf} (Type: {type(w_rrf)})")
        print("\n\n\n")

        # --- 2. AUTOMATED BACKGROUND LOOKUP ---
        automated_history = fetch_automatic_student_history(student_id) or []
        print(f"[AUTOMATION] Resolved history for {student_id}: {automated_history}")

        interests = request.form.get('interests_text', '').strip()

# Check if the user left the text field blank
        
        if not interest:
            # Force the engine into 100% Peer History mode automatically
            w_rrf = 0.0
            w_ps= 1.0
            query_fallback = True

        # --- 3. NLP CANDIDATE SEEDING (WITH TOKENLESS API PROTECTION) ---
        try:
            # Attempts standard semantic processing
            desired_courses = get_candidate_courses(
                query=interest, 
                student_history=automated_history, 
                top_k=10,
                w_rrf=w_rrf,  
                w_ps=w_ps 
            )
        except Exception as api_err:
            print(f"[OFFLINE FALLBACK] Token exhaustion detected. Using safe catalog fallback. Trace: {api_err}")

        # --- 4. ELIGIBILITY GATEWAY & SCORING ---
        # Now passing the correctly updated slider parameters down to your engine matrix!
        output = eligibility_recommender(
            student_id=student_id,
            Degree=degree,
            year=year,
            department=department,
            desired_courses=desired_courses,
            manual_course_history=automated_history,
            w_rrf=w_rrf,
            w_ps=w_ps
        )

        # --- 5. COMPILING DATA OUTPUT ARRAYS ---
        course_history = output.get("course_history", automated_history)
        eligible = output.get("eligible", [])
        i_a_r = output.get("i_a_r", [])
        rejected = output.get("rejected", [])

    return render_template(
        "index_checkbox.html",
        eligible=eligible,
        i_a_r=i_a_r,
        rejected=rejected,
        course_history=course_history,
        need_manual_history=False, 
        w_rrf=w_rrf,
        w_ps=w_ps,
        query_fallback=query_fallback
    )

if __name__ == "__main__":
    app.run(debug=True)