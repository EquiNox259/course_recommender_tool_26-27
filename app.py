import pandas as pd
import random, time, smtplib, os, ast
from email.mime.text import MIMEText
from flask import Flask, render_template, request, session, redirect, url_for, flash
from model.recommender import get_candidate_courses
from eligibility.rules import recommender as eligibility_recommender
from config import FLASK_SECRET, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, OTP_EXPIRY_SEC, FEEDBACK_PATH

app = Flask(__name__)
app.secret_key = FLASK_SECRET

# Load student records once when the server boots to keep lookups fast
STUDENT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "Student_data_combined.csv")

try:
    student_db = pd.read_csv(STUDENT_DATA_PATH)
    # Ensure emails/IDs are stripped for clean matching
    student_db["Emails"] = student_db["Emails"].str.strip()
    print(f"[SUCCESS] Automated Student Database loaded with {len(student_db)} records.")
except Exception as e:
    print(f"[WARNING] Failed to pre-load student database: {e}")
    student_db = None

try:
    _email_df = pd.read_csv(STUDENT_DATA_PATH, dtype=str)
    _email_df["Student ID"] = _email_df["Student ID"].str.strip().str.lower()
    _email_df["Emails"]     = _email_df["Emails"].str.strip()
    _email_df = _email_df.dropna(subset=["Student ID", "Emails"])
    student_email_map = dict(zip(_email_df["Student ID"], _email_df["Emails"]))
    print(f"[SUCCESS] Student email lookup loaded with {len(student_email_map)} entries.")
except Exception as e:
    print(f"[WARNING] Failed to load student email lookup: {e}")
    student_email_map = {}

def fetch_automatic_student_history(student_id):
    """
    Silently look up a student's historical completed courses from the CSV 
    using their LDAP ID / Email prefix.
    """
    if student_db is None or not student_id:
        return []
        
    # Standardize input for better ID matching at the next step
    search_term = str(student_id).strip().lower()
        
    # Query the dataframe
    record = student_db[student_db["Student ID"].str.lower() == search_term]
    
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

def resolve_student_email(student_id):
    """
    Return the registered webmail corresponding to a given roll number.
    """
    key   = str(student_id).strip().lower()
    email = student_email_map.get(key)
    if not email:
        raise ValueError(
            "No valid webmail address found for sending OTP."
        )
    return email

def send_otp(student_id):
    otp = str(random.randint(100000, 999999))
    to_email = resolve_student_email(student_id)
    msg = MIMEText(
        f"Hello,\n\nYour OTP for the DAV Course Recommender is:\n\n"
        f"    {otp}\n\nThis OTP is valid for 10 minutes.\n\n— DAV Team, IIT Bombay"
    )
    msg['Subject'] = 'Course Recommender — OTP Verification'
    msg['From']    = SMTP_USER
    msg['To']      = to_email
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
    return otp

@app.route("/", methods=["GET", "POST"])
def index():
    otp_sent     = session.get('otp_sent', False)
    otp_verified = session.get('otp_verified', False)
    otp_error    = None

    # Establish strict initial baseline defaults
    w_ps = 0.5
    w_rrf = 0.5
    eligible = None
    rejected = None
    i_a_r = None
    t_s_c = None
    course_history = None
    need_manual_history = False
    query_fallback = False

    if request.method == "POST":
        action = request.form.get("action")

         # ── Step 1: Send OTP ──────────────────────────────────────
        if action == "send_otp":
            student_id = request.form.get("student_id", "").strip().lower()
            try:
                otp = send_otp(student_id)
                session['otp_code']       = otp
                session['otp_sent_at']    = time.time()
                session['otp_student_id'] = student_id
                session['otp_sent']       = True
                session['otp_verified']   = False
                otp_sent = True
            except Exception as e:
                flash(f"Could not send OTP: {e}", "send_error")
                return redirect(url_for('index'))
            
         # ── Step 2: Verify OTP ────────────────────────────────────
        elif action == "verify_otp":
            entered  = request.form.get("otp_input", "").strip()
            stored   = session.get('otp_code')
            sent_at  = session.get('otp_sent_at', 0)
            otp_sent = True
            if time.time() - sent_at > OTP_EXPIRY_SEC:
                otp_error = "OTP has expired. Please request a new one."
                session['otp_sent'] = False
                otp_sent = False
            elif entered != stored:
                otp_error = "Incorrect OTP. Please try again."
            else:
                session['otp_verified'] = True
                otp_verified = True

         # ── Step 3: Generate recommendations ─────────────────────
        elif action == "recommend":
            if not session.get('otp_verified'):
                return redirect(url_for('index'))

            # --- 1. EXTRACT FROM INPUTS AND SLIDER PARAMENTERS FIRST ---
            student_id = session.get('otp_student_id')
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
            desired_courses = []
            try:
                # Attempts standard semantic processing
                desired_courses = get_candidate_courses(
                    query=interest, 
                    student_history=automated_history, 
                    top_k=60,
                    w_rrf=w_rrf,  
                    w_ps=w_ps 
                )
            except Exception as api_err:
                print(f"[OFFLINE FALLBACK] Token exhaustion detected. Using safe catalog fallback. Trace: {api_err}")


        

            # --- Manual history (second phase) ---
            manual_history_raw = request.form.get("manual_history")
            manual_course_history = None

            if manual_history_raw:
                manual_course_history = [
                    c.strip().upper()
                    for c in manual_history_raw.split(",")
                    if c.strip()
                ]

            # --- 4. ELIGIBILITY GATEWAY & SCORING ---
            # Now passing the correctly updated slider parameters down to your engine matrix!
            output = eligibility_recommender(
                student_id=student_id,
                Degree=degree,
                year=year,
                department=department,
                desired_courses=desired_courses,
                manual_course_history=manual_course_history or automated_history,
                w_rrf=w_rrf,
                w_ps=w_ps
            )

            # --- 5. COMPILING DATA OUTPUT ARRAYS, ACCOUNT FOR MANUAL HISTORY---
            if "need_manual_history" in output:
                need_manual_history = True
            else:
                course_history = output.get("course_history", automated_history)
                eligible = output.get("eligible", [])
                i_a_r = output.get("i_a_r", [])
                t_s_c = output.get("t_s_c", [])
                rejected = output.get("rejected", [])


    return render_template(
        "index.html",
        otp_sent=otp_sent,
        otp_verified=otp_verified,
        otp_error=otp_error,
        student_id=session.get('otp_student_id', ''),
        eligible=eligible,
        i_a_r = i_a_r,
        t_s_c = t_s_c,
        rejected=rejected,
        course_history=course_history,
        need_manual_history=need_manual_history,
        w_rrf=w_rrf,
        w_ps=w_ps,
        query_fallback=query_fallback
    )

@app.route("/feedback", methods=["POST"])
def feedback():
    if not session.get('otp_verified'):
        return redirect(url_for('index'))

    student_id = session.get('otp_student_id', '')
    email      = resolve_student_email(student_id) if student_id else ''
    rating     = request.form.get('rating', '').strip()
    text       = request.form.get('feedback_text', '').strip()

    row = {
        'Email':                email,
        'Rating (out of 5)':    rating,
        'Descriptive Feedback': text,
        'Timestamp':            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    try:
        if os.path.exists(FEEDBACK_PATH):
            df_fb = pd.read_csv(FEEDBACK_PATH)
        else:
            df_fb = pd.DataFrame(columns=['Email', 'Rating (out of 5)', 'Descriptive Feedback', 'Timestamp'])
        df_fb = pd.concat([df_fb, pd.DataFrame([row])], ignore_index=True)
        df_fb.to_csv(FEEDBACK_PATH, index=False)
        flash('Thank you for your feedback!', 'feedback_success')
    except Exception as e:
        flash(f'Could not save feedback: {e}', 'feedback_error')

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
