import pandas as pd
import random, time, smtplib, os, ast, re
from email.mime.text import MIMEText
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
from model.recommender import get_candidate_courses
from model.llm_service import LLMService
from eligibility.rules import recommender as eligibility_recommender, get_core_courses_for_bucket, parse_minor_remark
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

import threading

_local_violations = {}
_violations_lock = threading.Lock()
_violations_sheet = None

from config import DATASET_DIR
import json

def _load_violations_from_sheet():
    global _violations_sheet
    if _violations_sheet is None:
        # Local fallback file for offline/dev testing
        local_path = os.path.join(DATASET_DIR, "violations_local.json")
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                with _violations_lock:
                    for sid, val in data.items():
                        _local_violations[sid.lower()] = {
                            "count": int(val.get("count", 0)),
                            "banned": bool(val.get("banned", False))
                        }
                print(f"[SUCCESS] Loaded {len(_local_violations)} violation records from local JSON file.")
            except Exception as e:
                print(f"[ERROR] Failed to load local violations file: {e}")
        return
    try:
        records = _violations_sheet.get_all_records()
        with _violations_lock:
            for r in records:
                sid = str(r.get("Student ID", "")).strip().lower()
                if sid:
                    _local_violations[sid] = {
                        "count": int(r.get("Violation Count", 0)),
                        "banned": str(r.get("Banned", "")).strip().lower() == "true"
                    }
        print(f"[SUCCESS] Loaded {len(_local_violations)} violation records from Google Sheet.")
    except Exception as e:
        print(f"[ERROR] Failed to load violations from Google Sheet: {e}")

def _sync_violation_to_sheet(student_id, count, banned):
    global _violations_sheet
    if _violations_sheet is None:
        # Local fallback file for offline/dev testing
        local_path = os.path.join(DATASET_DIR, "violations_local.json")
        try:
            with _violations_lock:
                data = {}
                if os.path.exists(local_path):
                    with open(local_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                data[student_id] = {"count": count, "banned": banned}
                with open(local_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            print(f"[SUCCESS] Synced violation status for {student_id} to local JSON file.")
        except Exception as e:
            print(f"[ERROR] Failed to sync violation for {student_id} to local JSON: {e}")
        return
    try:
        cell = _violations_sheet.find(student_id, in_column=1)
        if cell:
            _violations_sheet.update_cell(cell.row, 2, count)
            _violations_sheet.update_cell(cell.row, 3, "true" if banned else "false")
        else:
            _violations_sheet.append_row([student_id, count, "true" if banned else "false"])
        print(f"[SUCCESS] Synced violation status for {student_id} to Google Sheet.")
    except Exception as e:
        print(f"[ERROR] Failed to sync violation for {student_id} to sheet: {e}")

def record_violation(student_id):
    sid = str(student_id).strip().lower()
    with _violations_lock:
        rec = _local_violations.get(sid, {"count": 0, "banned": False})
        rec["count"] += 1
        if rec["count"] >= 4:
            rec["banned"] = True
        _local_violations[sid] = rec
        count = rec["count"]
        banned = rec["banned"]
    
    # Sync in a daemon thread so it runs in background without blocking query response
    threading.Thread(target=_sync_violation_to_sheet, args=(sid, count, banned), daemon=True).start()
    return count, banned

_local_usage = {}
_usage_lock = threading.Lock()
_usage_sheet = None

def _load_usage_from_sheet():
    global _usage_sheet
    if _usage_sheet is None:
        # Local fallback file for offline/dev testing
        local_path = os.path.join(DATASET_DIR, "usage_local.json")
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                with _usage_lock:
                    for sid, val in data.items():
                        _local_usage[sid.lower()] = int(val)
                print(f"[SUCCESS] Loaded {len(_local_usage)} usage records from local JSON file.")
            except Exception as e:
                print(f"[ERROR] Failed to load local usage file: {e}")
        return
    try:
        records = _usage_sheet.get_all_records()
        with _usage_lock:
            for r in records:
                sid = str(r.get("Student ID", "")).strip().lower()
                if sid:
                    _local_usage[sid] = int(r.get("Query Count", 0))
        print(f"[SUCCESS] Loaded {len(_local_usage)} usage records from Google Sheet.")
    except Exception as e:
        print(f"[ERROR] Failed to load usage from Google Sheet: {e}")

def _sync_usage_to_sheet(student_id, count):
    global _usage_sheet
    if _usage_sheet is None:
        # Local fallback file for offline/dev testing
        local_path = os.path.join(DATASET_DIR, "usage_local.json")
        try:
            with _usage_lock:
                data = {}
                if os.path.exists(local_path):
                    with open(local_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                data[student_id] = count
                with open(local_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            print(f"[SUCCESS] Synced query count for {student_id} to local JSON file.")
        except Exception as e:
            print(f"[ERROR] Failed to sync query count for {student_id} to local JSON: {e}")
        return
    try:
        cell = _usage_sheet.find(student_id, in_column=1)
        if cell:
            _usage_sheet.update_cell(cell.row, 2, count)
        else:
            _usage_sheet.append_row([student_id, count])
        print(f"[SUCCESS] Synced query count for {student_id} to Google Sheet.")
    except Exception as e:
        print(f"[ERROR] Failed to sync usage for {student_id} to sheet: {e}")

def record_query_use(student_id):
    sid = str(student_id).strip().lower()
    with _usage_lock:
        count = _local_usage.get(sid, 0) + 1
        _local_usage[sid] = count
    
    # Sync in a daemon thread so it runs in background without blocking query response
    threading.Thread(target=_sync_usage_to_sheet, args=(sid, count), daemon=True).start()
    return count

try:
    _gc = gspread.service_account(filename=GSHEETS_CREDENTIALS_PATH)
    _sh = _gc.open(GSHEETS_SPREADSHEET_NAME)
    _feedback_sheet = _sh.sheet1
    print(f"[SUCCESS] Connected to feedback Google Sheet.")
    
    try:
        _violations_sheet = _sh.worksheet("Violations")
        print(f"[SUCCESS] Connected to Violations worksheet.")
    except gspread.exceptions.WorksheetNotFound:
        _violations_sheet = _sh.add_worksheet(title="Violations", rows="1000", cols="3")
        _violations_sheet.append_row(["Student ID", "Violation Count", "Banned"])
        print(f"[SUCCESS] Created Violations worksheet.")

    try:
        _usage_sheet = _sh.worksheet("Usage")
        print(f"[SUCCESS] Connected to Usage worksheet.")
    except gspread.exceptions.WorksheetNotFound:
        _usage_sheet = _sh.add_worksheet(title="Usage", rows="1000", cols="2")
        _usage_sheet.append_row(["Student ID", "Query Count"])
        print(f"[SUCCESS] Created Usage worksheet.")
        
    _load_violations_from_sheet()
    _load_usage_from_sheet()
except Exception as e:
    print(f"[WARNING] Failed to connect to Google Sheets: {e}")
    _feedback_sheet = None
    _violations_sheet = None
    _usage_sheet = None
    _load_violations_from_sheet()
    _load_usage_from_sheet()


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

def _apply_minor_type_override(course_list, minor_type_map):
    """
    Overrides the displayed division using the authoritative 'Type' column
    from ASC_Minor_Courses.csv, collapsing to a single division (no dropdown).
    If the mandated division isn't offered this semester, the course is
    DROPPED — different minors may require different divisions (M-tagged
    vs Elective) for the same course, so showing the wrong one would
    misrepresent whether it satisfies THIS minor's requirement.
    """
    filtered = []
    for c in course_list:
        clean_code = c.get("code", "").replace(" ", "").upper()
        minor_type = minor_type_map.get(clean_code, "")
        if not minor_type:
            filtered.append(c)   # not part of the requested minor basket — leave untouched
            continue

        divisions = c.get("divisions", [])
        if not divisions:
            continue  # drop

        # Default to the Minor-tagged division if this course has one;
        # otherwise fall back to the first non-clashing division.
        minor_idx = next((i for i, d in enumerate(divisions) if d.get("is_minor")), None)
        default_idx = minor_idx if minor_idx is not None else next(
            (i for i, d in enumerate(divisions) if not d.get("clashes_with")), 0
        )

        c["default_idx"] = default_idx
        c["slot"]        = divisions[default_idx].get("slot", c.get("slot"))
        c["instructor"]  = divisions[default_idx].get("instructor", c.get("instructor"))
        c["has_minor"]   = any(d.get("is_minor") for d in divisions)
        c["minor_only"]  = divisions[default_idx].get("is_minor", False)
        c["minor_type"]  = minor_type
        filtered.append(c)
        
    return filtered

def derive_degree(student_id: str) -> str:
    sid = str(student_id).strip().lower()
    if len(sid) >= 3 and sid[:2].isdigit():
        return {'b': 'B.Tech.', 'm': 'M.Tech.', 'd': 'Ph.D.'}.get(sid[2], '')
    return ''

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
    if "127.0.0.1:5000" in FRONTEND_ORIGIN or "localhost:5000" in FRONTEND_ORIGIN:
        return render_template('index.html')
    return redirect(FRONTEND_ORIGIN, code=302)

@app.route("/documentation")
def documentation():
    if "127.0.0.1:5000" in FRONTEND_ORIGIN or "localhost:5000" in FRONTEND_ORIGIN:
        return render_template('documentation.html')
    return redirect(FRONTEND_ORIGIN, code=302)

@app.route("/about")
def about():
    if "127.0.0.1:5000" in FRONTEND_ORIGIN or "localhost:5000" in FRONTEND_ORIGIN:
        return render_template('about.html')
    return redirect(FRONTEND_ORIGIN, code=302)


# ── API Routes (used by Vercel frontend) ──────────────────────────────────

@app.route("/api/send-otp", methods=["POST", "OPTIONS"])
def api_send_otp():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data       = request.get_json() or {}
    student_id = str(data.get("student_id") or "").strip().lower()
    
    # Check if student is suspended
    is_banned = False
    with _violations_lock:
        if student_id in _local_violations and _local_violations[student_id].get("banned"):
            is_banned = True
    if is_banned:
        return jsonify({"error": "Access Denied: Your account has been suspended due to repeated policy violations."}), 403

    try:
        resolved_email = resolve_student_email(student_id)   # always validate roll number
        if derive_degree(student_id) != 'B.Tech.':
           return jsonify({"error": "This tool currently supports Undergraduate students only."}), 403

        if DEV_BYPASS_OTP:
            # Skip email entirely — mark session as verified immediately
            
            # Return same shape as verify-otp so the frontend can pre-fill fields
            clean_sid      = student_id.strip().lower()
            default_degree = derive_degree(student_id)
            default_year   = ""
            if len(clean_sid) >= 3 and clean_sid[:2].isdigit():
                default_year = "20" + clean_sid[:2]

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

    # Check if student is suspended
    is_banned = False
    with _violations_lock:
        if student_id in _local_violations and _local_violations[student_id].get("banned"):
            is_banned = True
    if is_banned:
        return jsonify({"error": "Access Denied: Your account has been suspended due to repeated policy violations."}), 403

    # Auto-detect degree and year from student_id prefix so the frontend
    # can pre-fill the form fields without the student having to select them.
    default_degree = derive_degree(student_id)
    default_year   = ""
    clean_sid = str(student_id).strip().lower()
    if len(clean_sid) >= 3 and clean_sid[:2].isdigit():
        default_year = "20" + clean_sid[:2]

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
    
    # Check if student is suspended
    is_banned = False
    with _violations_lock:
        if student_id in _local_violations and _local_violations[student_id].get("banned"):
            is_banned = True
    if is_banned:
        return jsonify({"error": "Access Denied: Your account has been suspended due to repeated policy violations.", "is_banned": True}), 403

    degree     = data.get("degree", "")
    year       = data.get("year", "")
    department = data.get("department", "")
    interest   = data.get("interest", "")

    if not degree or not year or not department:
        return jsonify({"error": "Please the select Department."}), 400

    # Log usage count for active student
    record_query_use(student_id)

    # --- 2. AUTOMATED BACKGROUND LOOKUP ---
    automated_history = fetch_automatic_student_history(student_id) or []
    print(f"[AUTOMATION] Resolved history for {student_id}: {automated_history}")

    query_fallback   = False
    _processed = None

    try:
        if interest:
            _llm = LLMService()
            _processed = _llm.rephrase_and_extract_intent(interest)
    except Exception:
        _processed = None

    # Derive weights dynamically based on parsed intent
    w_ps = 0.0
    w_rrf = 1.0
    if _processed:
        constraints = _processed.get("constraints", {})
        easy_grading = constraints.get("easy_grading", False)
        popular = constraints.get("popular", False)
        primary_keywords = _processed.get("primary", [])
        
        if not primary_keywords:
            if easy_grading and popular:
                w_rrf = 0.00
                w_ps = 0.50
            elif popular:
                w_rrf = 0.00
                w_ps = 1.00
            elif easy_grading:
                w_rrf = 0.00
                w_ps = 0.00
            else:
                w_rrf = 0.00
                w_ps = 1.00
        else:
            if easy_grading and popular:
                w_rrf = 0.60
                w_ps = 0.20
            elif popular:
                w_rrf = 0.70
                w_ps = 0.30
            elif easy_grading:
                w_rrf = 0.70
                w_ps = 0.00

    # Check if the user left the text field blank
    if not interest:
        w_rrf = 0.0
        w_ps  = 1.0
        query_fallback = True

    desired_courses = []
    is_valid = True
    is_minor_query = False
    minor_type_map = {}
    try:
        desired_courses, is_valid, reject_reason, is_minor_query, minor_type_map = get_candidate_courses(
            query=interest,
            student_history=automated_history,
            top_k=60,                
            w_rrf=w_rrf,
            w_ps=w_ps,
            degree=degree,
            year=year,
            department=department,
            processed_query=_processed
        )
    except Exception as api_err:
        print(f"[OFFLINE FALLBACK] Token exhaustion detected. Trace: {api_err}")
    if is_valid == False:
        count, banned = record_violation(student_id)
        return jsonify({
            "eligible":           [],
            "i_a_r":              [],
            "t_s_c":              [],
            "rejected":           [],
            "course_history":     [],
            "w_ps":               0,
            "w_rrf":              0,
            "query_fallback":     query_fallback,
            "need_manual_history": False,
            "is_valid" : False,
            "reject_reason": reject_reason,
            "violation_warning": True if count == 3 else False,
            "violation_count": count,
            "is_banned": banned
        })

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
        prefer_minor_division = is_minor_query
    )

    # --- 5. COMPILING DATA OUTPUT ARRAYS ---
    if "need_manual_history" in output:
        return jsonify({"need_manual_history": True})

    course_history = output.get("course_history", automated_history)
    eligible = output.get("eligible", [])
    i_a_r    = output.get("i_a_r", [])
    t_s_c    = output.get("t_s_c", [])
    rejected = output.get("rejected", [])

    if is_minor_query:
        eligible = _apply_minor_type_override(eligible, minor_type_map)
        i_a_r    = _apply_minor_type_override(i_a_r, minor_type_map)
        t_s_c    = _apply_minor_type_override(t_s_c, minor_type_map)
        rejected = _apply_minor_type_override(rejected, minor_type_map)

    return jsonify({
        "eligible":           eligible,
        "i_a_r":              i_a_r,
        "t_s_c":              t_s_c,
        "rejected":           rejected,
        "course_history":     course_history,
        "w_ps":               w_ps,
        "w_rrf":              w_rrf,
        "query_fallback":     query_fallback,
        "need_manual_history": False,
        "is_valid": True,
        "reject_reason": "",
        "is_minor_mode": is_minor_query,
    })

@app.route("/api/favourites-info", methods=["POST", "OPTIONS"])
@require_auth
def api_favourites_info():
    data = request.get_json() or {}
    student_id = request.student_id
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
    desired_courses = [{"code": code, "raw_rrf": 0.0, "raw_ps": 0.0, "raw_ts": 0.0} for code in clean_codes]

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
            prefer_minor_division = False
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
    data       = request.get_json() or {}
    student_id = request.student_id
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
