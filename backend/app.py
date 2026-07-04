import pandas as pd
import random, time, smtplib, os, ast, traceback
from email.mime.text import MIMEText
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
from model.recommender import get_candidate_courses
from model.llm_service import LLMService
from eligibility.rules import recommender as eligibility_recommender, get_core_courses_for_bucket, detect_minor_intent, build_minor_candidates, parse_minor_remark
from config import FLASK_SECRET, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, OTP_EXPIRY_SEC, DEV_BYPASS_OTP, FEEDBACK_PATH, GSHEETS_CREDENTIALS_PATH, GSHEETS_SPREADSHEET_NAME, FRONTEND_ORIGIN
import gspread
import traceback
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import hashlib, hmac
from functools import wraps


_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
app = Flask(__name__,
            template_folder=_FRONTEND,
            static_folder=_FRONTEND,
            static_url_path='')

CORS(app, origins=[FRONTEND_ORIGIN], allow_headers=["Authorization", "Content-Type"], methods=["GET", "POST", "OPTIONS"],
     supports_credentials=True)

signer = URLSafeTimedSerializer(FLASK_SECRET)

def _otp_hash(otp: str, student_id: str) -> str:
    return hmac.new(FLASK_SECRET.encode(), f"{otp}:{student_id}".encode(),
                    hashlib.sha256).hexdigest()

def _issue_auth_token(student_id: str) -> str:
    return signer.dumps({"sid": student_id, "typ": "auth"})

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == "OPTIONS":
            return jsonify({}), 200
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Not authenticated"}), 401
        try:
            payload = signer.loads(header[7:], max_age=86400)
        except SignatureExpired:
            return jsonify({"error": "Session expired. Please verify again."}), 401
        except BadSignature:
            return jsonify({"error": "Invalid token"}), 401
        if payload.get("typ") != "auth":
            return jsonify({"error": "Invalid token"}), 401
        request.student_id = payload["sid"]
        return f(*args, **kwargs)
    return wrapper


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
            "No valid webmail address found for sending OTP"
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

@app.route("/")
def index():
    return redirect(FRONTEND_ORIGIN, code=302)


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
                "token": _issue_auth_token(student_id)
            })

        otp, _ = send_otp(student_id)
        otp_token = signer.dumps({
        "sid": student_id,
        "email": resolved_email,
        "oh": _otp_hash(otp, student_id),
        "typ": "otp",
        })
        return jsonify({"status": "ok", "email": resolved_email, "otp_token": otp_token})

    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route("/api/verify-otp", methods=["POST", "OPTIONS"])
def api_verify_otp():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data      = request.get_json() or {}
    entered   = str(data.get("otp_input") or "").strip()
    otp_token = data.get("otp_token") or ""

    try:
        payload = signer.loads(otp_token, max_age=OTP_EXPIRY_SEC)
    except SignatureExpired:
        return jsonify({"error": "OTP has expired. Please request a new one."}), 400
    except BadSignature:
        return jsonify({"error": "Invalid request. Please request a new OTP."}), 400

    if payload.get("typ") != "otp" or not hmac.compare_digest(
            payload.get("oh", ""), _otp_hash(entered, payload["sid"])):
        return jsonify({"error": "Incorrect OTP. Please try again."}), 400

    student_id = payload["sid"]

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
        "token":          _issue_auth_token(student_id),
    })


@app.route("/api/recommend", methods=["POST", "OPTIONS"])
@require_auth
def api_recommend():

    # --- 1. EXTRACT FROM INPUTS AND SLIDER PARAMETERS FIRST ---
    data       = request.get_json() or {}
    student_id = request.student_id
    degree     = data.get("degree", "")
    year       = data.get("year", "")
    department = data.get("department", "")
    interest   = data.get("interest", "")

    if not degree or not year or not department:
        return jsonify({"error": "Please the select Department."}), 400

    w_ps_raw  = data.get("w_ps")
    w_rrf_raw = data.get("w_rrf")
    w_ps  = float(w_ps_raw)  if w_ps_raw  is not None else 0.0
    w_rrf = float(w_rrf_raw) if w_rrf_raw is not None else 1.0

    print(f"\n[CHECKPOINT 1 - APP.PY] Incoming weights extracted from UI:")
    print(f" -> w_ps (Peer History Weight): {w_ps}")
    print(f" -> w_rrf (Semantic Weight):  {w_rrf}")

    # --- 2. AUTOMATED BACKGROUND LOOKUP ---
    automated_history = fetch_automatic_student_history(student_id) or []
    print(f"[AUTOMATION] Resolved history for {student_id}: {automated_history}")

    # --- 3. LLM-FIRST MINOR DETECTION (mirrors current index()) ---
    minor_candidates = []
    minor_query_type = "simple"
    query_fallback   = False
    desired_courses  = []

    try:
        llm = LLMService()
        _processed = llm.rephrase_and_extract_intent(interest)

        constraints = _processed.get("constraints", {})
        minor_list  = constraints.get("minor", [])
        minor_branch = minor_list[0] if minor_list else None

        is_minor_mode    = bool(minor_branch)
        minor_query_type = _processed.get("minor_query_type", "simple")
    except Exception:
        traceback.print_exc()
        _processed       = None
        minor_branch     = None
        is_minor_mode    = False
        minor_query_type = "simple"

    if is_minor_mode:
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
        _pre_query   = _processed          # reuse the single LLM call, as index() does
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
            return jsonify({"error": "Recommendation engine failed. Check server logs."}), 500

    # --- Manual history (second phase) ---
    manual_history_raw = data.get("manual_history", "")
    manual_course_history = None
    if manual_history_raw:
        manual_course_history = [c.strip().upper() for c in manual_history_raw.split(",") if c.strip()]

    # --- 4. ELIGIBILITY GATEWAY & SCORING ---
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

    # --- 5. COMPILING DATA OUTPUT ARRAYS ---
    if "need_manual_history" in output:
        return jsonify({"need_manual_history": True})

    course_history = output.get("course_history", automated_history)
    eligible = output.get("eligible", [])
    i_a_r    = output.get("i_a_r", [])
    t_s_c    = output.get("t_s_c", [])
    rejected = output.get("rejected", [])

    # --- 5a. MINOR MODE: annotate every entry with Type + Remark ---
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
                        _entry['divisions']   = _preferred
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
        "eligible":            eligible,
        "i_a_r":               i_a_r,
        "t_s_c":               t_s_c,
        "rejected":            rejected,
        "course_history":      course_history,
        "w_ps":                w_ps,
        "w_rrf":               w_rrf,
        "query_fallback":      query_fallback,
        "is_minor_mode":       is_minor_mode,
        "minor_branch":        minor_branch,
        "need_manual_history": False,
    })

@app.route("/api/favourites-info", methods=["POST", "OPTIONS"])
def api_favourites_info():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if not session.get('otp_verified'):
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json() or {}
    student_id = session.get('otp_student_id')
    degree     = data.get("degree", "")
    year       = data.get("year", "")
    department = data.get("department", "")
    codes      = data.get("codes", [])

    if not degree or not year or not department:
        return jsonify({"error": "Profile details (degree, year, department) are required."}), 400

    # Retrieve student history
    automated_history = fetch_automatic_student_history(student_id) or []
    manual_history_raw = data.get("manual_history", "")
    manual_course_history = None
    if manual_history_raw:
        manual_course_history = [c.strip().upper() for c in manual_history_raw.split(",") if c.strip()]
    history = manual_course_history or automated_history

    # Clean and filter codes
    clean_codes = list({str(c).strip().upper() for c in codes if c})
    if not clean_codes:
        return jsonify({
            "eligible": [],
            "i_a_r": [],
            "t_s_c": [],
            "rejected": []
        })

    # Prepare desired_courses shape for eligibility_recommender
    desired_courses = [{"code": code} for code in clean_codes]

    try:
        # Run standard eligibility checks
        output = eligibility_recommender(
            student_id=student_id,
            Degree=degree,
            year=year,
            department=department,
            desired_courses=desired_courses,
            manual_course_history=history,
            w_rrf=0.0,
            w_ps=0.0,
            is_minor_mode=False
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Eligibility analysis failed: {str(e)}"}), 500

    eligible = output.get("eligible", [])
    i_a_r    = output.get("i_a_r", [])
    t_s_c    = output.get("t_s_c", [])
    rejected = output.get("rejected", [])

    # Group all active/potential courses to evaluate mutual clashes
    active_courses = eligible + i_a_r + t_s_c
    
    # Map slot_num -> list of favorite course codes
    fav_slot_map = {}
    for c in active_courses:
        code = c["code"]
        for div in c.get("divisions", []):
            sn = div.get("slot_num")
            if sn and sn not in ("N/A", "L", "X"):
                fav_slot_map.setdefault(sn, []).append(code)

    # Re-evaluate clashes for each course considering both core courses and other favorites
    new_eligible = []
    new_i_a_r    = []
    new_t_s_c    = []

    for c in active_courses:
        code = c["code"]
        divs_updated = []
        
        for div in c.get("divisions", []):
            sn = div.get("slot_num")
            core_clashes = div.get("clashes_with", [])
            fav_clashes = []
            if sn and sn not in ("N/A", "L", "X"):
                fav_clashes = [fc for fc in fav_slot_map.get(sn, []) if fc != code]
            
            combined_clashes = [f"{cc} (Core)" for cc in core_clashes] + [f"{fc} (Favourite)" for fc in fav_clashes]
            
            divs_updated.append({
                **div,
                "clashes_with": combined_clashes
            })
            
        # Check if all non-minor divisions clash
        non_m = [d for d in divs_updated if not d.get('is_minor')] or divs_updated
        all_clash = all(bool(d.get('clashes_with')) for d in non_m)
        
        # Choose a default division (prefer one with fewer/no clashes)
        default_div = next((d for d in non_m if not d.get('clashes_with')), non_m[0])
        default_idx = next(i for i, d in enumerate(divs_updated) if d is default_div)
        
        # Update the course dictionary details
        c["divisions"] = divs_updated
        c["default_idx"] = default_idx
        c["slot"] = default_div.get("slot", c.get("slot", "N/A"))
        c["instructor"] = default_div.get("instructor", c.get("instructor", "N/A"))
        
        if all_clash:
            c["clashing_with"] = default_div["clashes_with"]
            new_t_s_c.append(c)
        else:
            # Check if it was in i_a_r originally
            if code in {x["code"] for x in i_a_r}:
                new_i_a_r.append(c)
            else:
                new_eligible.append(c)

    return jsonify({
        "eligible": new_eligible,
        "i_a_r":    new_i_a_r,
        "t_s_c":    new_t_s_c,
        "rejected": rejected
    })

@app.route("/api/feedback", methods=["POST", "OPTIONS"])
@require_auth
def api_feedback():

    if request.method == "OPTIONS":
        return jsonify({}), 200

    data       = request.get_json() or {}
    student_id = student_id = request.student_id
    email      = resolve_student_email(student_id) if student_id else ''
    timestamp  = pd.Timestamp.now(tz = 'Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S')

    try:
        if _feedback_sheet is None:
            raise RuntimeError("Feedback sheet not connected at startup.")
        _feedback_sheet.append_row([email, data.get('rating', ''), data.get('feedback_text', ''), timestamp])
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[FEEDBACK ERROR] {e}")
        return jsonify({"error": str(e)}), 500
    
#import resource
#print(f"[MEM] Peak RSS at boot: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")

@app.route("/api/sign-out", methods=["POST", "OPTIONS"])
def api_sign_out():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
