import ast
import numpy as np
import re
import pandas as pd
from config import RUNNING_COURSES_PATH, PREREQ_PATH, COURSES_HISTORY_PATH, CORE_COURSES_PATH, SEMESTER, GRADES_2024_PATH, GRADES_2025_PATH
from collections import defaultdict
import google.genai as genai
import json
from datetime import datetime

data = pd.read_csv(COURSES_HISTORY_PATH)
df = pd.read_csv(RUNNING_COURSES_PATH)
rename_map = {
    "Restrictions": "Restriction",
    "Course Description": "Description",
    "Instructor(s)": "Instructor",
    "Biometric Attendance Enabled?": "Biometric Attendance Enabled",
    "Division Defined": "Session/Division Defined",
    "Self-Study": "Self Study"
}
df.rename(columns=rename_map, inplace=True)
df_prereq = pd.read_excel(PREREQ_PATH)
df_core = pd.read_csv(CORE_COURSES_PATH)
df_core_sem = df_core[df_core['Sem'] == SEMESTER].copy()

rename_map = {
    "Restrictions": "Restriction",
    "Course Description": "Description",
    "Instructor(s)": "Instructor",
    "Biometric Attendance Enabled?": "Biometric Attendance Enabled",
    "Division Defined": "Session/Division Defined",
    "Self-Study": "Self Study"
}
df.rename(columns=rename_map, inplace=True)

# Build slot-number lookup from running courses: course_code -> set of slot numbers
# (a course may appear in multiple divisions with different slots)
def extract_slot_num(slot_str):
    """Returns the slot number (e.g. '4', '11', 'L') from a raw Slot cell."""
    if pd.isna(slot_str) or not str(slot_str).strip():
        return None
    first = str(slot_str).strip().split('\n')[0].strip()
    return first if first else None

running_slot_map = {}
for _, _row in df.iterrows():
    _code = str(_row['Course Code']).strip()
    _sn   = extract_slot_num(_row['Slot'])
    if _sn:
        running_slot_map.setdefault(_code, set()).add(_sn)

# course_meta : one default entry per code (first non-minor row)
# course_divisions : all rows per code (for the division dropdown)

_raw = defaultdict(list)
for _, row in df.iterrows():
    code = str(row['Course Code']).strip()
    div  = str(row.get('Division', '')).strip()
    _raw[code].append({
        "slot":        str(row.get('Slot',        'N/A')),
        "slot_num":    extract_slot_num(row.get('Slot', '')),
        "instructor":  str(row.get('Instructor',  'N/A')).strip(),
        "description": str(row.get('Description', '')),
        "division":    div,
        "is_minor":    (div == 'M'),
        "label":       ("Minor"   if div == 'M'
                        else "Regular" if not div
                        else div.strip()),
    })

course_meta      = {}
course_divisions = {}

for code, rows in _raw.items():
    course_divisions[code] = rows
    default = next((r for r in rows if not r['is_minor']), rows[0])
    course_meta[code] = {
        "slot":        default['slot'],
        "instructor":  default['instructor'],
        "description": default['description'],
    }

# M-Tag course codes (no spaces) — for equiv filtering
m_tag_set = {
    str(row['Course Code']).strip().replace(' ', '')
    for _, row in df.iterrows()
    if str(row.get('Division', '')).strip() == 'M'
}

# Equivalent courses map: code → {regular: [...], minor: [...]}
_peq = re.compile(r'[A-Z]{2,3}\d{3,4}')
equiv_map = {}
for _, erow in df_prereq[df_prereq['Type'] == 'equivalent'].iterrows():
    code  = str(erow['CourseCode']).strip().replace(' ', '')
    raw   = re.sub(r'\s+(\d)', r'\1', str(erow['Courses'])).upper()
    codes = _peq.findall(raw)
    equiv_map[code] = {
        'regular': [c for c in codes if c not in m_tag_set],
        'minor':   [c for c in codes if c in m_tag_set],
    }

data_dict = {}

# Matches: MA105, CS101, PH227, BB101 etc.
course_code_pattern = r'[A-Z]{2,3}\s?\d{3}'

data_cleaned = data.dropna(subset=['Emails']).copy()

for _, row in data_cleaned.iterrows():
    student_id = row["Emails"][:7]
    raw_courses = row["Courses"]

    if not isinstance(raw_courses, str):
        continue

    # Extract course codes safely
    courses_list = re.findall(course_code_pattern, raw_courses)

    if courses_list:
        # Normalize spacing: "MA 105" → "MA105"
        normalized = [c.replace(" ", "") for c in courses_list]
        data_dict[student_id] = list(set(normalized))


''' this is to convert restriction from string to numpy array with transpose so that we can process it easily '''
df_restrictions = df['Restriction']
restrictions_modified = []

for i in df_restrictions:
    if pd.isna(i):
        i_array_np_T = 'No restrictions'
    elif isinstance(i, str):
    if pd.isna(i):
        i_array_np_T = 'No restrictions'
    elif isinstance(i, str):
        if i == 'No restrictions':
            i_array_np_T = i
        else:
            i_array = ast.literal_eval(i)
            i_array_np = np.array(i_array)
            i_array_np_T = np.transpose(i_array_np)
    elif isinstance(i, np.ndarray):
        # If it's already a numpy array, just transpose it
        i_array_np_T = np.transpose(i)
    else:
        # If it's neither string nor array, leave it as is
        i_array_np_T = i
        print('garbage')

    restrictions_modified.append(i_array_np_T)

df['Restriction'] = restrictions_modified

data_modified=df.copy()

# ── Grade statistics lookup (year-wise) ──────────────────────────
_GRADE_ORDER = ['AA', 'AB', 'AP', 'AU', 'BB', 'BC', 'CC', 'CD', 'DD', 'FF', 'FR', 'II', 'NP', 'PP']

def _build_grade_stats():
    sources = [
        ('2024 Autumn', GRADES_2024_PATH),
        ('2025 Autumn', GRADES_2025_PATH),
    ]
    year_data = {}
    for year_label, path in sources:
        try:
            df_g = pd.read_csv(path)
        except Exception:
            continue
        available = [c for c in _GRADE_ORDER if c in df_g.columns]
        num_cols  = available + ['Total']
        df_g[num_cols] = df_g[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        for _, row in df_g.iterrows():
            code     = row['Course Code'].strip()
            division = str(row['Division']).strip() if 'Division' in df_g.columns else 'Main'
            total = row['Total']
            if total <= 0:
                continue
            grades = {col: int(row[col]) for col in available if row[col] > 0}
            if not grades:          # skip if grades are not mentioned
                continue
            entry = {
                'division':   division,
                'grades':     grades,
                'total':      int(total),
                'pct_ap':     round(row.get('AP', 0) / total * 100, 1),
                'pct_aa':     round(row.get('AA', 0) / total * 100, 1),
                'pct_aa_ab':  round((row.get('AA', 0) + row.get('AB', 0)) / total * 100, 1),
            }
            year_data.setdefault(code, {}).setdefault(year_label, []).append(entry)
    return year_data

grade_stats_db = _build_grade_stats()

def check_restriction(Degree,year,department,course_code, data=data_modified):
    year=str(year)
    # Check if course_code exists in the data
    if course_code not in data['Course Code'].values:
        return 'Not in available data'

    # Fetch restriction value
    restrictions = data.loc[data['Course Code'] == course_code, 'Restriction'].values[0]

    # If no restriction (a string 'No restrictions')
    if (len(restrictions) == 0) or (isinstance(restrictions, float) and pd.isna(restrictions)) or (isinstance(restrictions, str) and restrictions.lower() == 'no restrictions'):   # ← move it here
        return 'Valid'
    elif isinstance(restrictions, str):
        return 'Restricted'
    else:
        # Handle degree-based restrictions
        if Degree in ['B.Tech.', 'M.Tech.', 'Ph.D.', 'M.Sc.', 'Dual Degree (B.Tech. + M.Tech.)', 'B.S.', 'B.Des.']:

            allowed_groups = []
            for j in restrictions:
                if j[3] == 'Allowed':
                    parts = []
                    if j[2] != 'ALL': parts.append(j[2])
                    if j[1] != 'ALL': parts.append(j[1])
                    if j[0] != 'ALL': parts.append(f"{j[0]}")
                    allowed_groups.append(', '.join(parts) if parts else 'specific students')
            if allowed_groups:
                restricted_msg = f"Open only to {' / '.join(allowed_groups)}"
            else:
               restricted_msg = 'Restricted'

            # Check if there are any 'Allowed' rules at all in the restrictions list
            has_allowed_rule = any(rule[3] == 'Allowed' for rule in restrictions)
            # Checks the restrictions fully to see if restricted
            is_explicitly_denied = False
            is_matched_allowed = False

            for i in restrictions:
              if i[0]==year or i[0]=='ALL':
                if i[1]==department or i[1]=='ALL':
                  if i[2]==Degree or i[2]=='ALL':
                    if i[3]=='Deny':
                      is_explicitly_denied = True
                    elif i[3]=='Allowed':
                      is_matched_allowed = True
            # Year-Ignored check: see if branch & degree match but year is restricted
            branch_allowed = False
            for i in restrictions:
              if i[1] == department or i[1] == 'ALL':
                if i[2] == Degree or i[2] == 'ALL':
                  if i[3] == 'Allowed':
                    branch_allowed = True
                    break
            if is_explicitly_denied:
                if branch_allowed:
                    return 'Restricted by year'
                else:
                    return restricted_msg
            else:
                if has_allowed_rule:
                    if is_matched_allowed:
                        return 'Valid'
                    else:
                       if branch_allowed:
                            return 'Restricted by year'
                       else:
                            return restricted_msg
                else:
                    return 'Valid'
        else:
            return 'Invalid Degree'
        
    return restricted_msg

#check prerequsite
def check_prereq(course_code,course_hist,degree = None, data=df_prereq):
    # Check if course_code exists in the data
    #course_hist is course history of student
    if course_code not in data['CourseCode'].values: # No prereq
        return 'Valid', None, None

    # Fetch prerequsite course code and instructor approval status
    rows = data[data['CourseCode'] == course_code]

    if degree and 'Applicable forprogram(s)' in data.columns:
        preferred = rows[rows['Applicable forprogram(s)'].str.contains(
            degree, na=False, regex=False
        )]
        row = preferred.iloc[0] if len(preferred) > 0 else rows.iloc[0]
    else:
        row = rows.iloc[0]

    prereq   = row['Courses']
    approval = row['InstructorConsent']
    type_    = row['Type']
    remark_raw = row['Remark']

    remark = (None if (pd.isna(remark_raw) or str(remark_raw).strip().upper() == 'NA')
          else re.sub(r'(\w)â(\w)', r"\1'\2",
                      str(remark_raw).strip()
                      .replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')))
    minor_prereq = None
    extra_prereqs = []
    remark_has_approval = False

    if remark:
        m_match = re.search(
            r'((?:[A-Z]{2,3}\s?\d{3,4}M?(?:\s+and\s+)?)+)\s+(?:are|is)\s+(?:a\s+)?pre-?req',
            remark, re.IGNORECASE
        )
        if m_match:
            codes = re.findall(r'[A-Z]{2,3}\s?\d{3,4}M?', m_match.group(1))
            minor_prereq = ' AND '.join(c.replace(' ', '') for c in codes)

        raw_codes = re.findall(r'[A-Z]{2,3}\s?\d{3,4}M?', remark)
        remark_codes = [c.replace(' ', '') for c in raw_codes]
        existing_codes = ([] if pd.isna(prereq) else
            re.findall(r'[A-Z]{2,3}\d{3,4}M?',
                       re.sub(r'\s+(\d)', r'\1', str(prereq).upper())))
        extra_prereqs = [c for c in remark_codes
                         if c not in existing_codes and c.rstrip('M') not in existing_codes
                        and c != course_code.replace(' ', '')]

        remark_has_approval = bool(re.search(
            r"instructor'?s?\s+(?:approval|consent)|approval\s+required|requires?\s+.*consent",
            remark, re.IGNORECASE
        ))

        stripped = re.sub(r'[A-Z]{2,3}\s?\d{3,4}M?', '', remark)
        stripped = re.sub(r'\s+', ' ', stripped).strip(' .,')
        if extra_prereqs:
        # Sentence-level: strip whole approval-containing sentences
            after_approval = re.sub(r"[^.]*instructor'?s?\s+(?:approval|consent)[^.]*\.?",
                            '', stripped, flags=re.IGNORECASE)
            
        else:
           # Phrase-level: strip only the core approval phrase itself
            after_approval = re.sub(
                r"(?:\w+\s+)?(?:need\s+)?instructor'?s?\s+(?:approval|consent)(?:\s+required)?",
                '', stripped, flags=re.IGNORECASE
            )
        
        after_approval = re.sub(r'\s+', ' ', after_approval).strip(' .,')
        remark = remark if len(after_approval) > 20 else None

    extra_status = None
    if extra_prereqs:
        missing = [c for c in extra_prereqs
                   if c not in course_hist and c.rstrip('M') not in course_hist]
        if missing:
            extra_status = f'Prerequisite not met. You need to complete {" AND ".join(missing)}.'

    if remark_has_approval and approval not in ('Required', 'Conditional'):
        approval = 'Required'
    
    if isinstance(prereq, str):
        prereq = re.sub(r'\s+(\d)', r'\1', prereq).upper()

    pattern = r'(?:[A-Z]{2,3}\d{3,4})'

    # Equivalent courses — inverted logic
    if type_ == 'equivalent':
        equiv_codes = re.findall(pattern, str(prereq).upper())
        for eq in equiv_codes:
            if eq in course_hist:
                return f'Equivalent course already completed: {eq}.', remark, minor_prereq
        return extra_status or 'Valid', remark, minor_prereq
    
    #Blank cells => Instructor approval required
    if pd.isna(prereq):
        if extra_status:
            return extra_status, remark, minor_prereq
        return 'Instructor approval required', remark, minor_prereq
    # if single prereq
    elif re.match(pattern, prereq) and (len(prereq) in [5, 6, 7]):
        if prereq in course_hist:
          if approval == 'Required':
            return 'Instructor approval required', remark, minor_prereq
          elif approval == 'Conditional':
            return 'Instructor approval is conditional', remark, minor_prereq
          else:
            return extra_status or 'Valid', remark, minor_prereq
        else:
          return f'Prerequisite not met. You need to complete {prereq}.', remark, minor_prereq
    # some boolean expression (i.e. AND , OR)
    else:
      pre_course = re.findall(pattern, prereq)
      pattern_2 = r'(?:[A-Z]{2,3}\d{3,4})|OR|AND|\(|\)'
      l = re.findall(pattern_2,prereq)
      #if prereq contain only OR
      if 'OR' in l and "AND" not in l:
        for i in pre_course:
          if i in course_hist:
            if approval == 'Required':
                return 'Instructor approval required', remark, minor_prereq
            elif approval == 'Conditional':
                return 'Instructor approval is conditional', remark, minor_prereq
            else:
                return extra_status or 'Valid', remark, minor_prereq
        return f'Prerequisite not met. You need to complete {prereq}.', remark, minor_prereq

      # if prereq only contain AND
      elif 'OR' not in l and 'AND' in l:
        for i in pre_course:
          if i not in course_hist:
            return f'Prerequisite not met. You need to complete {prereq}.', remark, minor_prereq
        if approval == 'Required':
            return 'Instructor approval required', remark, minor_prereq
        elif approval == 'Conditional':
            return 'Instructor approval is conditional', remark, minor_prereq
        else:
            return extra_status or 'Valid', remark, minor_prereq

      elif "OR" in l and 'AND' in l:
        open_close_idx = [(x,i) for i, x in enumerate(l) if x == '(' or x == ')']
        a,b=[],[]
        p_idx=[]
        for i in open_close_idx:
          if i[0]=='(':
            a.append(i[1])
          else:
            b.append(i[1])
            p_idx.append((a[-1],b[-1]))
            b.pop(-1)
            a.pop(-1)

        res_val = None
        res_ = l[p_idx[-1][1]+1:]

        if len(res_) == 2:
         if re.match(pattern, res_[-1]) and len(res_[-1]) in [5, 6, 7]:
            res_val = res_[-1] in course_hist

        list_=[]
        for i in p_idx:
          valid=False
          prereq_i_list = l[i[0]+1:i[1]]
          prereq_i = ' '.join(prereq_i_list)
          pre_course_i = re.findall(pattern, prereq_i)
          pattern_2 = r'(?:[A-Z]{2,3}\d{3,4})|OR|AND|\(|\)'
          l_i = re.findall(pattern_2,prereq_i)

          if re.match(pattern, prereq_i) and (len(prereq_i) in [5, 6, 7]):
             if prereq_i in course_hist:
               valid=True
          else:
            if 'OR' in l_i and "AND" not in l_i:
             for item in pre_course_i:
               if item in course_hist:
                  valid=True

            elif 'OR' not in l_i and 'AND' in l_i:
             for item in pre_course_i:
               if item not in course_hist:
                   break
             else:
              valid=True
          list_.append(valid)
        t=l
        p_idx_copy = p_idx.copy()
        shift = 0
        for i in p_idx_copy:
            start = i[0] - shift
            end = i[1] - shift
            length = end - start

            del t[start:end+1]
            t.insert(start, list_[0])
            list_.pop(0)

            shift += length
        if res_val !=None:
          t.pop(-1)
          t.append(res_val)
        r = t[0]
        for i in range(1, len(t), 2):
          r = r and t[i+1] if t[i] == 'AND' else r or t[i+1]
        if r:
            if approval == 'Required':
                return 'Instructor approval required', remark, minor_prereq
            elif approval == 'Conditional':
                return 'Instructor approval is conditional', remark, minor_prereq
            else:
                return extra_status or 'Valid', remark, minor_prereq

      return f'Prerequisite not met. You need to complete {prereq}.', remark, minor_prereq
    
def check_clash(course_code, core_slot_to_courses):
    slot_num = extract_slot_num(course_meta.get(course_code, {}).get("slot", ""))
    if not slot_num or slot_num in ("N/A", "L", "X"):
        return []
    return core_slot_to_courses.get(slot_num, [])
    
def norm_code(x):
    return str(x).replace(" ", "").upper().strip()


def recommender(student_id, Degree, year, department, desired_courses, manual_course_history=None, w_rrf=0.5, w_ps=0.5):
    """
    student_id       : string (email prefix)
    Degree           : string (e.g. 'B.Tech.')
    year             : string or int (e.g. '2' or 2)
    department       : string (e.g. 'ME', 'CS')
    desired_courses  : list of course codes (from NLP / preference model)

    returns          : list of eligible course codes
    """

    # 1. Fetch course history
    if student_id in data_dict:
       course_hist = data_dict[student_id]
       history_source = "auto"
    else:
      if manual_course_history is None:
        return {
            "need_manual_history": True
        }
      else:
        course_hist = manual_course_history
        history_source = "manual"

    # Build slot-number → core course name map for this student
    current_cal_year = datetime.now().year
    academic_year_start = current_cal_year if SEMESTER == 'Autumn' else current_cal_year - 1
    year_in_program = academic_year_start - int(year) + 1
    core_mask = (
        (df_core_sem['Branch']  == department) &
        (df_core_sem['Degree']  == Degree) &
        (df_core_sem['Year'] == year_in_program)
    )
    dept_core = df_core_sem[core_mask]

    all_core_codes = {
        norm_code(str(row['Course Code']))
        for _, row in df_core[
            (df_core['Branch'] == department) &
            (df_core['Degree'] == Degree)
        ].iterrows()
    }

    course_hist_norm = {norm_code(c) for c in course_hist}
    core_slot_to_courses = {}
    for _, crow in dept_core.iterrows():
        code_norm = norm_code(crow['Course Code'])
        code_orig = str(crow['Course Code']).strip()
        if code_norm in course_hist_norm:
            continue
        for sn in running_slot_map.get(code_orig, set()):
            core_slot_to_courses.setdefault(sn, []).append(code_orig)

    eligible_courses = []
    rejected_courses = []
    i_a_r_courses = []
    t_s_c_courses = []

    # 2. Loop over model-recommended courses
    seen_codes = set()
    for course in desired_courses:
        course_code = course["code"]
        if course_code in seen_codes:
            continue
        seen_codes.add(course_code)

        # some course names are NaN for some reason, this deals with that 
        course_name = course.get("name")
        if not isinstance(course_name, str) or pd.isna(course_name):
            course_name = ""
        # ignores any courses with the below words (for future DAV members, remove this and see what semantic search gives to know why its there)
        blacklist = ["SEMINAR", "MINI PROJECT", "SUPERVISED"]
        if any(term in course_name.upper() for term in blacklist):
            continue

        if re.search(r'\bPROJECT\s*(I{1,3}|[1-4])?\s*$', course_name.upper()):
           continue

        if norm_code(course_code) in all_core_codes:
            continue

        # Pre-compute division metadata once per course
        divs       = course_divisions.get(course_code, [])
        has_minor  = any(d['is_minor'] for d in divs)
        minor_only = all(d['is_minor'] for d in divs) and bool(divs)

        # 3. Check restriction
        if norm_code(course_code) in course_hist_norm:
            continue
        r_status = check_restriction(
            Degree=Degree,
            year=year,
            department=department,
            course_code=course_code
        )
        
        if r_status != 'Valid':
            if r_status == 'Restricted by year':
                meta = course_meta.get(course_code, {})

                rejected_courses.append({
    "code": course_code,
    "name": course_name,
    "divisions":  divs,
    "has_minor":  has_minor,
    "minor_only": minor_only,
    "slot": meta.get("slot", "N/A"),
    "instructor": meta.get("instructor", "N/A"),
    "description": meta.get("description", ""),
    "equiv": equiv_map.get(course_code.replace(' ', ''), {'regular': [], 'minor': []}),
    "score": course.get("score", 0),
    "reason": r_status,
    "raw_rrf": course.get("raw_rrf"),
    "raw_ps": course.get("raw_ps"),
    "final_score": course.get("raw_ts"),
    "grade_stats": grade_stats_db.get(course_code, None)
})

            continue

        # 4. Check prerequisite
        p_status, p_remark, p_minor_prereq = check_prereq(
            course_code=course_code,
            course_hist=course_hist,
            degree = Degree
        )

        if p_status != 'Valid':
            meta = course_meta.get(course_code, {})

            if p_status == 'Instructor approval required' or p_status == 'Instructor approval is conditional':
                divs_with_clash = [{
                    **d,
                    'clashes_with': (
                        core_slot_to_courses.get(d.get('slot_num'), [])
                        if d.get('slot_num') and d.get('slot_num') not in ('N/A', 'L', 'X')
                        else []
                    )
                } for d in divs]

                non_m = [d for d in divs_with_clash if not d['is_minor']] or divs_with_clash
                all_clash = all(bool(d['clashes_with']) for d in non_m)

                if all_clash:
                    default_div = next((d for d in non_m if not d['clashes_with']), non_m[0])
                    default_idx = next(i for i, d in enumerate(divs_with_clash) if d is default_div)
                    t_s_c_courses.append({
                        "code": course_code,
                        "name": course_name,
                        "divisions": divs_with_clash,
                        "has_minor": has_minor,
                        "minor_only": minor_only,
                        "slot": meta.get("slot", "N/A"),
                        "instructor": meta.get("instructor", "N/A"),
                        "description": meta.get("description", ""),
                        "default_idx": default_idx,
                        "equiv": equiv_map.get(course_code.replace(' ', ''), {'regular': [], 'minor': []}),
                        "score": course.get("score", 0),
                        "clashing_with": default_div['clashes_with'],
                        "raw_rrf": course.get("raw_rrf"),
                        "raw_ps": course.get("raw_ps"),
                        "final_score": course.get("raw_ts"),
                        "grade_stats": grade_stats_db.get(course_code, None)
                    })
                else:
                    reason = (p_remark if p_remark
                                else p_status + '. Contact them for more info.' if p_status == 'Instructor approval is conditional'
                                else '')
                    i_a_r_courses.append({
                        "code": course_code,
                        "name": course_name,
                        "divisions": divs,
                        "has_minor": has_minor,
                        "minor_only": minor_only,
                        "slot": meta.get("slot", "N/A"),
                        "instructor": meta.get("instructor", "N/A"),
                        "description": meta.get("description", ""),
                        "equiv": equiv_map.get(course_code.replace(' ', ''), {'regular': [], 'minor': []}),
                        "score": course.get("score", 0),
                        "reason": reason,
                        "remark": p_remark,
                        "minor_prereq": p_minor_prereq,
                        "raw_rrf": course.get("raw_rrf"),
                        "raw_ps": course.get("raw_ps"),
                        "final_score": course.get("raw_ts"),
                        "grade_stats": grade_stats_db.get(course_code, None)
                    })

            else:
                rejected_courses.append({
                    "code": course_code,
                    "name": course_name,
                    "divisions": divs,
                    "has_minor": has_minor,
                    "minor_only": minor_only,
                    "slot": meta.get("slot", "N/A"),
                    "instructor": meta.get("instructor", "N/A"),
                    "description": meta.get("description", ""),
                    "equiv": equiv_map.get(course_code.replace(' ', ''), {'regular': [], 'minor': []}),
                    "score": course.get("score", 0),
                    "reason": p_status,
                    "remark": p_remark,
                    "minor_prereq": p_minor_prereq,
                    "raw_rrf": course.get("raw_rrf"),
                    "raw_ps": course.get("raw_ps"),
                    "final_score": course.get("raw_ts"),
                    "grade_stats": grade_stats_db.get(course_code, None)
                })
            continue


        # 5. Check slot clash , per division, if applicable
        meta = course_meta.get(course_code, {})
        divs_with_clash = [{
            **d,
            'clashes_with': (
                core_slot_to_courses.get(d.get('slot_num'), [])
                if d.get('slot_num') and d.get('slot_num') not in ('N/A', 'L', 'X')
                else []
            )
        } for d in divs]

        non_m = [d for d in divs_with_clash if not d['is_minor']] or divs_with_clash
        all_clash   = all(bool(d['clashes_with']) for d in non_m)
        default_div = next((d for d in non_m if not d['clashes_with']), non_m[0])
        default_idx = next(i for i, d in enumerate(divs_with_clash) if d is default_div)

        entry ={
                "code":          course_code,
                "name":          course_name,
                "name":          course_name,
                "divisions":  divs_with_clash,
                "has_minor":  has_minor,
                "minor_only": minor_only,
                "slot":          meta.get("slot",        "N/A"),
                "instructor":    meta.get("instructor",  "N/A"),
                "description":   meta.get("description", ""),
                "default_idx": default_idx,
                "equiv": equiv_map.get(course_code.replace(' ', ''), {'regular': [], 'minor': []}),
                "raw_rrf": course.get("raw_rrf"),
                "raw_ps": course.get("raw_ps"),
                "final_score": course.get("raw_ts"),
                "grade_stats": grade_stats_db.get(course_code, None)
            }
        
        if all_clash:
            t_s_c_courses.append({
                **entry,
                "clashing_with": default_div['clashes_with'],
            })
            continue
        
        # 6. Passed all checks
        meta = course_meta.get(course_code, {})

        eligible_courses.append({
    "code": course_code,
    "name": course_name,
    "divisions":  divs_with_clash,
    "has_minor":  has_minor,
    "minor_only": minor_only,
    "slot": meta.get("slot", "N/A"),
    "instructor": meta.get("instructor", "N/A"),
    "description": meta.get("description", ""),
    "default_idx": default_idx,
    "equiv": equiv_map.get(course_code.replace(' ', ''), {'regular': [], 'minor': []}),
    "raw_rrf": course.get("raw_rrf"),
    "raw_ps": course.get("raw_ps"),
    "final_score": course.get("raw_ts"),
    "grade_stats": grade_stats_db.get(course_code, None)
})

        

    return {
        "course_history": course_hist,
        "history_source": history_source,
        "eligible": eligible_courses,
        "i_a_r": i_a_r_courses,
        "t_s_c": t_s_c_courses,
        "rejected": rejected_courses
    }


'''list_= recommender('24b0350','B.tech','2024','Chemical Engineering',['CS 213','SI 505','SI 427','CL 603','CL 688','SI 402'])
print(list_)'''
