import pandas as pd
import random, time, smtplib, os, ast
from email.mime.text import MIMEText
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from flask_cors import CORS
from model.recommender import get_candidate_courses
from model.llm_service import LLMService
from eligibility.rules import recommender as eligibility_recommender, get_core_courses_for_bucket, detect_minor_intent, build_minor_candidates, parse_minor_remark
from config import FLASK_SECRET, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, OTP_EXPIRY_SEC, DEV_BYPASS_OTP, FEEDBACK_PATH, GSHEETS_CREDENTIALS_PATH, GSHEETS_SPREADSHEET_NAME, FRONTEND_ORIGIN
import gspread

_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
app = Flask(__name__,
            template_folder=_FRONTEND,
            static_folder=_FRONTEND,
            static_url_path='')
app.secret_key = FLASK_SECRET

CORS(app, origins=[FRONTEND_ORIGIN], allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"],
     supports_credentials=True)
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE']   = os.environ.get('FLASK_ENV') == 'production'


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
    _gc = gspread.service_account(filename=GSHEETS_CREDENTIALS_PATH)
    _feedback_sheet = _gc.open(GSHEETS_SPREADSHEET_NAME).sheet1
    print(f"[SUCCESS] Connected to feedback Google Sheet.")
except Exception as e:
    print(f"[WARNING] Failed to connect to feedback Google Sheet: {e}")
    _feedback_sheet = None

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
    return otp, to_email

@app.route("/health")
def health():
    return "ok", 200

@app.route("/", methods=["GET", "POST"])
def index():
    otp_sent     = session.get('otp_sent', False)
    otp_verified = session.get('otp_verified', False)
    otp_error    = None

    # Clear stale sessions if student ID is missing
    if otp_verified and not session.get('otp_student_id'):
        session.clear()
        otp_verified = False
        otp_sent = False

    # Establish strict initial baseline defaults
    w_ps = 0.0
    w_rrf = 1.0
    eligible = None
    rejected = None
    i_a_r = None
    t_s_c = None
    course_history = None
    need_manual_history = False
    query_fallback = False
    is_minor_mode = False
    minor_branch  = None
   # core_courses_for_bucket = []

    if request.method == "POST":
        action = request.form.get("action")

         # ── Step 1: Send OTP ──────────────────────────────────────
        if action == "send_otp":
            student_id = request.form.get("student_id", "").strip().lower()
            try:
                otp, resolved_email = send_otp(student_id)
                session['otp_code']       = otp
                session['otp_email'] = resolved_email
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

            # Guard against blank Degree/Year/Department
            if not degree or not year or not department:
                flash("Please select Degree, Batch Year, and Department before generating recommendations.", "recommend_error")
                return redirect(url_for('index'))

            # Preserve the student's choices (except for interests) across any future redirect
            # (e.g. after filling the feedback section) so the form doesn't reset.
            session['last_query'] = {
                'degree': degree, 'year': year,
                'department': department,
            }


            # Capture weight inputs from form fields or fallback to default
            w_ps_raw = request.form.get('w_ps') or request.args.get('w_ps')
            w_rrf_raw = request.form.get('w_rrf') or request.args.get('w_rrf')
            

            w_ps = float(w_ps_raw) if w_ps_raw else 0
            w_rrf = float(w_rrf_raw) if w_rrf_raw else 1

            print(f"\n[CHECKPOINT 1 - APP.PY] Incoming weights extracted from UI:")
            print(f" -> w_ps (Peer History Weight): {w_ps} (Type: {type(w_ps)})")
            print(f" -> w_rrf (Semantic Weight):  {w_rrf} (Type: {type(w_rrf)})")
            print("\n\n\n")

            # --- 2. AUTOMATED BACKGROUND LOOKUP ---
            automated_history = fetch_automatic_student_history(student_id) or []
            print(f"[AUTOMATION] Resolved history for {student_id}: {automated_history}")

            # Minor mode detection
            minor_candidates = []
            minor_query_type = "simple"

            try:                    
                llm = LLMService()
                _processed = llm.rephrase_and_extract_intent(interest)

                constraints = _processed.get("constraints", {})

                minor_list = constraints.get("minor", [])
                minor_branch = minor_list[0] if minor_list else None

                is_minor_mode = bool(minor_branch)

                minor_query_type = _processed.get("minor_query_type", "simple")

            except Exception:
                traceback.print_exc()
                _processed = None
                minor_branch = None
                is_minor_mode = False
                minor_query_type = "simple"
            if is_minor_mode:
                # PREPROCESSING (first API call): classify as simple list vs stacked with constraints
                try:
                    minor_query_type = _processed.get("minor_query_type", "simple")
                except Exception as _mq_err:
                    _processed = None
                    minor_query_type = "simple"

                minor_candidates = build_minor_candidates(minor_branch, degree)

                if minor_query_type == "simple":
                    desired_courses = minor_candidates
                    w_rrf = 0.0
                    w_ps  = 0.0

            # Check if the user left the text field blank
            if not interest:
                w_rrf = 0.0
                w_ps  = 1.0
                query_fallback = True

            if not is_minor_mode or minor_query_type == "stacked":
                _minor_codes = {c['code'] for c in minor_candidates} if (is_minor_mode and minor_query_type == "stacked") else None

                _pre_query   = _processed 
                desired_courses = []
                try:
                    desired_courses = get_candidate_courses(
                        query=interest,
                        student_history=automated_history,
                        top_k=60,
                        w_rrf=w_rrf,
                        w_ps=w_ps,
                        degree=degree,
                        year=year,
                        department=department,
                        processed_query=_pre_query,
                        minor_course_codes=_minor_codes
                    )

                except Exception:
                    print("===== REAL TRACEBACK =====")
                    traceback.print_exc()
                    print("==========================")
                    raise


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
                w_ps=w_ps,
                is_minor_mode=is_minor_mode,
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

                # --- 5a. MINOR MODE: annotate every entry with Type + Remark ─
                if is_minor_mode:
                    _minor_meta = {c['code']: c for c in minor_candidates}
                    _hist = course_history or []
                    for _section in (eligible, i_a_r, t_s_c, rejected):
                        for _entry in _section:
                            _m = _minor_meta.get(_entry['code'], {})
                            _raw = _m.get('minor_remark', '')
                            _parsed = (parse_minor_remark(_raw, _hist, department)
                                       if _raw else {})
                            _entry['minor_type']           = _m.get('minor_type', '')
                            _entry['minor_remark_note']    = _parsed.get('note', '')
                            _entry['minor_remark_warning'] = _parsed.get('warning', '')
                            # Prereq unmet from remark → move to rejected with reason
                            # (already placed in correct section by eligibility engine; we display the remark reason)
                            if _parsed.get('prereq_unmet') and _entry in eligible:
                                _entry['minor_prereq_note'] = _parsed['prereq_unmet']

                            # In minor mode, filter dual-listed courses to the
                            # single division the student should actually register under.
                            # Rule (from ASC_Minor_Courses.csv Type column):
                            #   'elective' in minor_type → non-M (regular) division
                            #   anything else            → M-tagged division
                            _mtype = (_entry.get('minor_type') or '').lower()
                            _want_m_side = 'elective' not in _mtype

                            _divs = _entry.get('divisions', [])
                            _is_dual_listed = (
                                any(d.get('is_minor') for d in _divs) and
                                any(not d.get('is_minor') for d in _divs)
                            )
                            if _is_dual_listed:
                                # 1) Registration-division dropdown
                                _preferred = [d for d in _divs if d.get('is_minor') == _want_m_side]
                                if _preferred:
                                    _entry['divisions']    = _preferred
                                    _entry['default_idx'] = 0
                                    _entry['slot']        = _preferred[0]['slot']
                                    _entry['instructor']  = _preferred[0]['instructor']

                                # 2) Grading Statistics — only filter if an 'M'
                                # division entry is actually present in the
                                # historical data; otherwise leave grade_stats untouched
                                _gstats = _entry.get('grade_stats')
                                if _gstats:
                                    _has_m_entry = any(
                                        e.get('division') == 'M'
                                        for _yr_entries in _gstats.values()
                                        for e in _yr_entries
                                    )
                                    if _has_m_entry:
                                        _filtered_gstats = {}
                                        for _yr, _yr_entries in _gstats.items():
                                            _kept = [
                                                e for e in _yr_entries
                                                if (e.get('division') == 'M') == _want_m_side
                                            ]
                                            if _kept:
                                                _filtered_gstats[_yr] = _kept
                                        _entry['grade_stats'] = _filtered_gstats or None

            # --- 6. CORE COURSES FOR BUCKET ---
            '''core_courses_for_bucket = get_core_courses_for_bucket(
                degree=degree,
                department=department,
                batch_year=year
            )'''


    # Auto-detect degree and year from student_id prefix
    default_degree = ""
    default_year = ""
    default_dept = ""
    
    otp_student_id = session.get('otp_student_id', '')
    if otp_student_id:
        clean_sid = str(otp_student_id).strip().lower()
        if len(clean_sid) >= 3 and clean_sid[:2].isdigit():
            default_year = "20" + clean_sid[:2]
            deg_char = clean_sid[2]
            if deg_char == 'b':
                default_degree = "B.Tech."
            elif deg_char == 'm':
                default_degree = "M.Tech."
            elif deg_char == 'p':
                default_degree = "Ph.D."
            elif deg_char == 'd':
                default_degree = "Dual Degree (B.Tech. + M.Tech.)"

    return render_template(
        "index.html",
        otp_sent=otp_sent,
        otp_verified=otp_verified,
        otp_error=otp_error,
        otp_email = session.get('otp_email', ''),
        student_id=session.get('otp_student_id', ''),
        eligible=eligible,
        i_a_r = i_a_r,
        t_s_c = t_s_c,
        rejected=rejected,
        course_history=course_history,
        need_manual_history=need_manual_history,
        w_rrf=w_rrf,
        w_ps=w_ps,
        query_fallback=query_fallback,
        is_minor_mode=is_minor_mode,
        form_values={
            'degree':     request.form.get('degree')     or session.get('last_query', {}).get('degree', ''),
            'year':       request.form.get('year')       or session.get('last_query', {}).get('year', ''),
            'department': request.form.get('department') or session.get('last_query', {}).get('department', ''),
            'interest':   request.form.get('interest', ''),
        },
        minor_branch=minor_branch,
        default_degree=default_degree,
        default_year=default_year,
        default_dept=default_dept
    )

@app.route("/feedback", methods=["POST"])
def feedback():
    if not session.get('otp_verified'):
        return redirect(url_for('index'))

    student_id = session.get('otp_student_id', '')
    email      = resolve_student_email(student_id) if student_id else ''
    rating     = request.form.get('rating', '').strip()
    text       = request.form.get('feedback_text', '').strip()
    timestamp  = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        if _feedback_sheet is None:
            raise RuntimeError("Feedback sheet connection was not established at startup.")
        _feedback_sheet.append_row([email, rating, text, timestamp])
        flash('Thank you for your feedback!', 'feedback_success')
    except Exception as e:
        print(f"[FEEDBACK SAVE ERROR] {e}")
        flash(f'Could not save feedback: {e}', 'feedback_error')

    return redirect(url_for('index'))

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ── API Routes (used by Vercel frontend) ──────────────────────────────────

@app.route("/api/send-otp", methods=["POST", "OPTIONS"])
def api_send_otp():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data       = request.get_json() or {}
    student_id = str(data.get("student_id") or "").strip().lower()
    try:
        resolved_email = resolve_student_email(student_id)   # always validate roll number

        if DEV_BYPASS_OTP:
            # Skip email entirely — mark session as verified immediately
            session['otp_student_id'] = student_id
            session['otp_email']      = resolved_email
            session['otp_verified']   = True
            session['otp_sent']       = True

            # Return same shape as verify-otp so the frontend can pre-fill fields
            clean_sid      = student_id.strip().lower()
            default_degree = ""
            default_year   = ""
            if len(clean_sid) >= 3 and clean_sid[:2].isdigit():
                default_year = "20" + clean_sid[:2]
                deg_char = clean_sid[2]
                if   deg_char == 'b': default_degree = "B.Tech."
                elif deg_char == 'm': default_degree = "M.Tech."
                elif deg_char == 'p': default_degree = "Ph.D."
                elif deg_char == 'd': default_degree = "Dual Degree (B.Tech. + M.Tech.)"

            return jsonify({
                "status":         "ok",
                "bypassed":       True,
                "student_id":     student_id,
                "default_degree": default_degree,
                "default_year":   default_year,
            })

        otp, _ = send_otp(student_id)
        session['otp_code']       = otp
        session['otp_email']      = resolved_email
        session['otp_sent_at']    = time.time()
        session['otp_student_id'] = student_id
        session['otp_sent']       = True
        session['otp_verified']   = False
        return jsonify({"status": "ok", "email": resolved_email})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/verify-otp", methods=["POST", "OPTIONS"])
def api_verify_otp():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data    = request.get_json() or {}
    entered = str(data.get("otp_input") or "").strip()
    stored  = session.get('otp_code')
    sent_at = session.get('otp_sent_at', 0)

    if time.time() - sent_at > OTP_EXPIRY_SEC:
        session['otp_sent'] = False
        return jsonify({"error": "OTP has expired. Please request a new one."}), 400
    if entered != stored:
        return jsonify({"error": "Incorrect OTP. Please try again."}), 400

    session['otp_verified'] = True
    student_id = session.get('otp_student_id', '')

    # Auto-detect degree and year from student_id prefix so the frontend
    # can pre-fill the form fields without the student having to select them.
    default_degree = ""
    default_year   = ""
    clean_sid = str(student_id).strip().lower()
    if len(clean_sid) >= 3 and clean_sid[:2].isdigit():
        default_year = "20" + clean_sid[:2]
        deg_char = clean_sid[2]
        if   deg_char == 'b': default_degree = "B.Tech."
        elif deg_char == 'm': default_degree = "M.Tech."
        elif deg_char == 'p': default_degree = "Ph.D."
        elif deg_char == 'd': default_degree = "Dual Degree (B.Tech. + M.Tech.)"

    return jsonify({
        "status":         "ok",
        "student_id":     student_id,
        "default_degree": default_degree,
        "default_year":   default_year,
    })


@app.route("/api/recommend", methods=["POST", "OPTIONS"])
def api_recommend():

    if request.method == "OPTIONS":
        return jsonify({}), 200
    
    if not session.get('otp_verified'):
        return jsonify({"error": "Not authenticated"}), 401

    data       = request.get_json() or {}
    student_id = session.get('otp_student_id')
    degree     = data.get("degree", "")
    year       = data.get("year", "")
    department = data.get("department", "")
    interest   = data.get("interest", "")
    w_ps_raw   = data.get("w_ps")
    w_rrf_raw  = data.get("w_rrf")
    manual_history_raw = data.get("manual_history", "")

    if not degree or not year or not department:
        return jsonify({"error": "Please select Degree, Batch Year, and Department."}), 400

    w_ps  = float(w_ps_raw)  if w_ps_raw  is not None else 0.0
    w_rrf = float(w_rrf_raw) if w_rrf_raw is not None else 1.0

    session['last_query'] = {'degree': degree, 'year': year, 'department': department}

    automated_history = fetch_automatic_student_history(student_id) or []
    print(f"[AUTOMATION] Resolved history for {student_id}: {automated_history}")

    minor_branch, _ = detect_minor_intent(interest)
    is_minor_mode   = bool(minor_branch)
    minor_candidates = []
    query_fallback   = False
    desired_courses  = []
    minor_query_type = "simple"
    _processed       = None

    if is_minor_mode:
        try:
            _llm = LLMService()
            _processed = _llm.rephrase_and_extract_intent(interest)
            minor_query_type = _processed.get("minor_query_type", "simple")
        except Exception as _mq_err:
            print(f"[MINOR CLASSIFY FALLBACK] Defaulting to simple. Trace: {_mq_err}")
            _processed       = None
            minor_query_type = "simple"

        minor_candidates = build_minor_candidates(minor_branch, degree)

        if minor_query_type == "simple":
            desired_courses = minor_candidates
            w_ps = w_rrf = 0.0

    if not interest:
        w_rrf = 0.0
        w_ps  = 1.0
        query_fallback = True

    if not is_minor_mode or minor_query_type == "stacked":
        _minor_codes = {c['code'] for c in minor_candidates} if (is_minor_mode and minor_query_type == "stacked") else None
        _pre_query   = _processed if (is_minor_mode and minor_query_type == "stacked" and _processed) else None
        desired_courses = []
        try:
            desired_courses = get_candidate_courses(
                query=interest,
                student_history=automated_history,
                top_k=60,
                w_rrf=w_rrf,
                w_ps=w_ps,
                degree=degree,
                year=year,
                department=department,
                processed_query=_pre_query,
                minor_course_codes=_minor_codes
            )
        except Exception as api_err:
            print(f"[OFFLINE FALLBACK] Token exhaustion detected. Trace: {api_err}")
            if is_minor_mode and minor_query_type == "stacked":
                desired_courses = minor_candidates

    manual_course_history = None
    if manual_history_raw:
        manual_course_history = [c.strip().upper() for c in manual_history_raw.split(",") if c.strip()]

    output = eligibility_recommender(
        student_id=student_id,
        Degree=degree,
        year=year,
        department=department,
        desired_courses=desired_courses,
        manual_course_history=manual_course_history or automated_history,
        w_rrf=w_rrf,
        w_ps=w_ps,
        is_minor_mode=is_minor_mode,
    )

    if "need_manual_history" in output:
        return jsonify({"need_manual_history": True})

    course_history = output.get("course_history", automated_history)
    eligible = output.get("eligible", [])
    i_a_r    = output.get("i_a_r", [])
    t_s_c    = output.get("t_s_c", [])
    rejected = output.get("rejected", [])

    if is_minor_mode:
        _minor_meta = {c['code']: c for c in minor_candidates}
        _hist = course_history or []
        for _section in (eligible, i_a_r, t_s_c, rejected):
            for _entry in _section:
                _m      = _minor_meta.get(_entry['code'], {})
                _raw    = _m.get('minor_remark', '')
                _parsed = (parse_minor_remark(_raw, _hist, department) if _raw else {})
                _entry['minor_type']           = _m.get('minor_type', '')
                _entry['minor_remark_note']    = _parsed.get('note', '')
                _entry['minor_remark_warning'] = _parsed.get('warning', '')
                if _parsed.get('prereq_unmet') and _entry in eligible:
                    _entry['minor_prereq_note'] = _parsed['prereq_unmet']

                _mtype       = (_entry.get('minor_type') or '').lower()
                _want_m_side = 'elective' not in _mtype
                _divs        = _entry.get('divisions', [])
                _is_dual     = (any(d.get('is_minor') for d in _divs) and
                                any(not d.get('is_minor') for d in _divs))
                if _is_dual:
                    _preferred = [d for d in _divs if d.get('is_minor') == _want_m_side]
                    if _preferred:
                        _entry['divisions']    = _preferred
                        _entry['default_idx'] = 0
                        _entry['slot']        = _preferred[0]['slot']
                        _entry['instructor']  = _preferred[0]['instructor']
                    _gstats = _entry.get('grade_stats')
                    if _gstats:
                        _has_m = any(e.get('division') == 'M'
                                     for _yr_entries in _gstats.values() for e in _yr_entries)
                        if _has_m:
                            _fg = {}
                            for _yr, _ents in _gstats.items():
                                _kept = [e for e in _ents if (e.get('division') == 'M') == _want_m_side]
                                if _kept: _fg[_yr] = _kept
                            _entry['grade_stats'] = _fg or None

    return jsonify({
        "eligible":           eligible,
        "i_a_r":              i_a_r,
        "t_s_c":              t_s_c,
        "rejected":           rejected,
        "course_history":     course_history,
        "w_ps":               w_ps,
        "w_rrf":              w_rrf,
        "query_fallback":     query_fallback,
        "is_minor_mode":      is_minor_mode,
        "minor_branch":       minor_branch,
        "need_manual_history": False,
    })


@app.route("/api/feedback", methods=["POST", "OPTIONS"])
def api_feedback():
    if not session.get('otp_verified'):
        return jsonify({"error": "Not authenticated"}), 401
    
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data       = request.get_json() or {}
    student_id = session.get('otp_student_id', '')
    email      = resolve_student_email(student_id) if student_id else ''
    timestamp  = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        if _feedback_sheet is None:
            raise RuntimeError("Feedback sheet not connected at startup.")
        _feedback_sheet.append_row([email, data.get('rating', ''), data.get('feedback_text', ''), timestamp])
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[FEEDBACK ERROR] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
