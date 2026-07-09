import ast
import numpy as np
import re
import pandas as pd
from config import RUNNING_COURSES_PATH, RUNNING_COURSES_PATH_ALT, PREREQ_PATH, COURSES_HISTORY_PATH, CORE_COURSES_PATH, SEMESTER, GRADES_2024_PATH, GRADES_2025_PATH
from collections import defaultdict
from datetime import datetime

data = pd.read_csv(COURSES_HISTORY_PATH)
df = pd.read_csv(RUNNING_COURSES_PATH)
df_alt = pd.read_csv(RUNNING_COURSES_PATH_ALT)
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
df_prereq['CourseCode'] = df_prereq['CourseCode'].astype(str).str.replace(" ", "", regex=False).str.upper()
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
    _code = str(_row['Course Code']).replace(" ", "").upper().strip()
    _sn   = extract_slot_num(_row['Slot'])
    if _sn:
        running_slot_map.setdefault(_code, set()).add(_sn)

# course_meta : one default entry per code (first non-minor row)
# course_divisions : all rows per code (for the division dropdown)

_raw = defaultdict(list)
for _, row in df.iterrows():
    code = str(row['Course Code']).replace(" ", "").upper().strip()
    div  = str(row.get('Division', '')).strip()
    _raw[code].append({
        "slot":        str(row.get('Slot',        'N/A')),
        "slot_num":    extract_slot_num(row.get('Slot', '')),
        "instructor":  str(row.get('Instructor',  'N/A')).strip(),
        "description": str(row.get('Description', '')),
        "credits":     int(str(row.get('Credits', '') or '').strip() or 6),
        "division":    div,
        "is_minor":    (div == 'M'),
        "label":       ("Minor"   if div == 'M'
                        else "Elective" if not div
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
        "credits":     default.get('credits', 6),
    }

# Courses that have both a regular and an M-tagged variant
historical_divisions = {}

for meta_df in (df, df_alt):

    for _, row in meta_df.iterrows():

        code = str(row["Course Code"]).strip()
        code_norm = code.replace(" ", "")

        division = str(row.get("Division", "")).strip()

        historical_divisions.setdefault(code_norm, set()).add(division)

courses_with_minor_variant = {
    code
    for code, divs in historical_divisions.items()
    if "M" in divs and any(d != "M" for d in divs)
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
df['CourseCodeNorm'] = df['Course Code'].astype(str).str.replace(" ", "", regex=False).str.upper()

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
            code     = str(row['Course Code']).replace(" ", "").upper().strip()
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
                'score_aa_ab': (row.get('AA', 0) + row.get('AB', 0)) / total,
                'pct_ap':     round(row.get('AP', 0) / total * 100, 1),
                'pct_aa':     round(row.get('AA', 0) / total * 100, 1),
                'pct_aa_ab':  round((row.get('AA', 0) + row.get('AB', 0)) / total * 100, 1),
            }
            year_data.setdefault(code, {}).setdefault(year_label, []).append(entry)
    return year_data

grade_stats_db = _build_grade_stats()

GRADE_WEIGHT = 0.30

def apply_grade_boost(courses, w_ps):
    """
    Boost recommendation scores based on historical AA+AB percentage.
    In the query-driven weights model, this is a no-op since grading
    filters and combined scores are calculated early in candidate generation.
    """
    if not courses:
        return courses
    if w_ps == 1.0:
        courses = [course for course in courses if course.get("raw_ts", 0) != 0]
    courses.sort(key=lambda x: x.get("raw_ts", 0), reverse=True)
    return courses

def _build_year_restriction_msg(restrictions, department, Degree):
    """
    When a course is restricted by year (the student's dept/degree is eligible,
    but their batch year isn't), return a human-readable message listing which
    year(s) of students are actually allowed.
    Uses the same batch-year → year-in-program conversion as the recommender.
    """
    def _ordinal(n):
        if n == 1: return "1st"
        if n == 2: return "2nd"
        if n == 3: return "3rd"
        if n == 4: return "4th"
        if n == 5: return '5th'
        return f"{n}th"
    current_cal_year    = datetime.now().year
    academic_year_start = current_cal_year if SEMESTER == 'Autumn' else current_cal_year - 1

    # Collect every batch year that has an Allowed rule matching this dept/degree
    allowed_batch_years = set()
    for r in restrictions:
        if r[3] != 'Allowed':
            continue
        if r[1] in (department, 'ALL') and r[2] in (Degree, 'ALL') and r[0] != 'ALL':
            try:
                allowed_batch_years.add(int(r[0]))
            except (ValueError, TypeError):
                pass

    if not allowed_batch_years:
        return 'Restricted'

    # Convert each batch year → year-in-program ordinal, filter implausible values
    max_yip = 5 if Degree == 'Dual Degree (B.Tech. + M.Tech.)' else 4
    year_labels = []
    for batch_yr in allowed_batch_years:
        yip = academic_year_start - batch_yr + 1
        if 1 <= yip <= max_yip:
            year_labels.append((_ordinal(yip), yip))

    year_labels.sort(key=lambda x: x[1])   # ascending: 1st, 2nd, …
    labels = [y[0] for y in year_labels]

    if not labels:
        return 'Restricted by year'
    if len(labels) == 1:
        return f"Available for {labels[0]} year students only"
    if len(labels) == 2:
        return f"Available for {labels[0]} and {labels[1]} year students only"
    return f"Available for {', '.join(labels[:-1])}, and {labels[-1]} year students only"

def check_restriction(Degree,year,department,course_code, data=data_modified, is_minor=False, division=None):
    year=str(year)
    course_code_norm = str(course_code).replace(" ", "").upper().strip()
    # Check if course_code exists in the data
    if course_code_norm not in data['CourseCodeNorm'].values:
        return 'Not in available data'

    subset = data[data['CourseCodeNorm'] == course_code_norm]
    if division is not None:
        div_row = subset[subset['Division'].astype(str).str.strip() == str(division).strip()]
        if not div_row.empty:
            rows_to_check = div_row
        else:
            rows_to_check = subset
    elif is_minor:
        minor_row = subset[subset['Division'].astype(str).str.strip() == 'M']
        if not minor_row.empty:
            rows_to_check = minor_row
        else:
            rows_to_check = subset
    else:
        regular_rows = subset[subset['Division'].astype(str).str.strip() != 'M']
        if not regular_rows.empty:
            rows_to_check = regular_rows
        else:
            return 'Restricted to minor students'

    def evaluate_single_row(restrictions):
        # If no restriction
        if (len(restrictions) == 0) or (isinstance(restrictions, float) and pd.isna(restrictions)) or (isinstance(restrictions, str) and restrictions.lower() == 'no restrictions'):
            return 'Valid'
        elif isinstance(restrictions, str):
            return 'Restricted'

        # Support all degrees! No hardcoded lists.
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

        def get_specificity(rule):
            score = 0
            if rule[0] != 'ALL': score += 1
            if rule[1] != 'ALL': score += 1
            if rule[2] != 'ALL': score += 1
            return score

        def evaluate_rules(Degree, year, department, restrictions):
            matching_rules = []
            for r in restrictions:
                if r[0] in (year, 'ALL') and r[1] in (department, 'ALL') and r[2] in (Degree, 'ALL'):
                    matching_rules.append(r)
            
            if not matching_rules:
                has_allowed_rule = any(r[3] == 'Allowed' for r in restrictions)
                return 'Deny' if has_allowed_rule else 'Allowed'
            
            max_spec = max(get_specificity(r) for r in matching_rules)
            best_rules = [r for r in matching_rules if get_specificity(r) == max_spec]
            
            actions = [r[3] for r in best_rules]
            if 'Deny' in actions:
                return 'Deny'
            return 'Allowed'

        status = evaluate_rules(Degree, year, department, restrictions)
        
        if status == 'Allowed':
            return 'Valid'
            
        branch_allowed = False
        for r in restrictions:
            if r[1] in (department, 'ALL') and r[2] in (Degree, 'ALL') and r[3] == 'Allowed':
                branch_allowed = True
                break
                
        if branch_allowed:
            return _build_year_restriction_msg(restrictions, department, Degree)
            
        return restricted_msg

    results = []
    for _, row in rows_to_check.iterrows():
        res = evaluate_single_row(row['Restriction'])
        if res == 'Valid':
            return 'Valid'
        results.append(res)

    for r in results:
        if 'year students' in r or r == 'Restricted':
            return r
    return results[0] if results else 'Restricted'

#check prerequsite
def check_prereq(course_code,course_hist,degree = None, data=df_prereq):
    # Normalize course_code to a stripped uppercase code (without spaces)
    course_code = str(course_code).replace(" ", "").upper().strip()
    # Normalize course_hist to a set of stripped uppercase codes (without spaces)
    course_hist = {str(c).replace(" ", "").upper() for c in course_hist if c}
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
    s = str(x).replace(" ", "").upper().strip()
    return re.sub(r'-\d{4}$', '', s)

def get_core_courses_for_bucket(degree, department, batch_year):
    """
    Returns core courses for the upcoming semester based on
    degree, department and batch year.
    Slot numbers are looked up from running_slot_map where available.
    """
    from datetime import datetime
    current_year = datetime.now().year
 
    # Compute year of study from batch year and current semester
    try:
        batch = int(batch_year)
    except (TypeError, ValueError):
        return []
 
    if SEMESTER == 'Autumn':
        year_of_study = current_year - batch + 1
    else:
        year_of_study = current_year - batch
 
    # CSV Degree → CSV Branch exact match required
    degree_map = {
        'B.Tech.':      'B.Tech.',
        'B.S.':         'B.S.',
        'Dual Degree':  'Dual Degree (B.Tech. + M.Tech.)',
    }
    csv_degree = degree_map.get(str(degree).strip(), str(degree).strip())
 
    mask = (
        (df_core_sem['Degree']  == csv_degree)  &
        (df_core_sem['Branch']  == str(department).strip()) &
        (df_core_sem['Year']    == year_of_study)
    )
    rows = df_core_sem[mask]
 
    if rows.empty:
        return []
 
    result = []
    for _, row in rows.iterrows():
        code = str(row['Course Code']).strip()
        # Look up slot from running courses; take first slot found
        slot_nums = running_slot_map.get(code.replace(" ", "").upper(), set())
        slot_num  = next(iter(slot_nums), 'N/A')
        try:
            credits = int(str(row.get('Credits', 6)).strip() or 6)
        except (ValueError, TypeError):
            credits = 6
        result.append({
            'code':        code,
            'name':        str(row['Course Name']).strip(),
            'credits':     credits,
            'slot_num':    slot_num,
            'course_type': str(row['Course Type']).strip(),
            'type':        'core',
            'is_minor':    False,
            'tag':         'core',
        })
    return result

# Minor courses support

# Build course name lookup from running courses + metadata CSV.
_course_name_map: dict = {}
_cn_col = next((c for c in df.columns if 'course name' in c.lower()), None)
if _cn_col:
    for _, _r in df.iterrows():
        _code = str(_r['Course Code']).replace(" ", "").upper().strip()
        _name = str(_r.get(_cn_col, '')).strip()
        if _code and _name and _name.lower() not in ('nan', ''):
            _course_name_map.setdefault(_code, _name)

try:
    from config import MINOR_COURSES_PATH as _MINOR_PATH, RUNNING_COURSES_PATH as _META_PATH
    _df_meta_raw = pd.read_csv(_META_PATH)
    for _, _r in _df_meta_raw.iterrows():
        _code = str(_r.get('Course Code', '')).replace(" ", "").upper().strip()
        _name = str(_r.get('Course Name', '')).strip()
        if _code and _name and _name.lower() not in ('nan', ''):
            _course_name_map.setdefault(_code, _name)
    _df_minor = pd.read_csv(_MINOR_PATH)
    print(f"[SUCCESS] Minor courses loaded: {len(_df_minor)} rows, "
          f"name map: {len(_course_name_map)} entries.")
except Exception as _me:
    print(f"[WARNING] Minor courses/metadata load failed: {_me}")
    _df_minor = pd.DataFrame(
        columns=['Degree', 'Branch', 'Specialization', 'Type', 'Course Code', 'Remarks']
    )

# Department keyword → exact Branch name in ASC_Minor_Courses.csv
_MINOR_DEPT_ALIASES: dict = {
    # CMInDS
    'cminds':                                  'Centre for Machine Intelligence and Data Science',
    'centre for machine intelligence':         'Centre for Machine Intelligence and Data Science',
    'machine intelligence and data science':   'Centre for Machine Intelligence and Data Science',
    'machine intelligence':                    'Centre for Machine Intelligence and Data Science',
    'data science':                            'Centre for Machine Intelligence and Data Science',
    'ds':                                      'Centre for Machine Intelligence and Data Science',
    # CSE
    'cs':                                      'Computer Science and Engineering',
    'cse':                                     'Computer Science and Engineering',
    'computer science':                        'Computer Science and Engineering',
    # EE
    'electrical engineering':                  'Electrical Engineering',
    'electrical':                              'Electrical Engineering',
    # ME
    'mechanical engineering':                  'Mechanical Engineering',
    'mechanical':                              'Mechanical Engineering',
    'mech':                                    'Mechanical Engineering',
    # MEMS
    'metallurgical engineering':               'Metallurgical Engineering and Materials Science',
    'metallurgical':                           'Metallurgical Engineering and Materials Science',
    'materials science':                       'Metallurgical Engineering and Materials Science',
    'mems':                                    'Metallurgical Engineering and Materials Science',
    # Chemistry
    'chemistry':                               'Chemistry',
    # Maths
    'mathematics':                             'Mathematics',
    'math':                                    'Mathematics',
    'maths':                                   'Mathematics',
    # Physics
    'physics':                                 'Physics',
    # Economics
    'economics':                               'Economics',
    # Management
    'management':                              'Shailesh J. Mehta School of Management',
    'som':                                     'Shailesh J. Mehta School of Management',
    # Entrepreneurship
    'entrepreneurship':                        'Desai Sethi School of Entrepreneurship',
    'ent':                                     'Desai Sethi School of Entrepreneurship',
    # Statistics
    'statistics':                              'Statistics',
    'stats':                                   'Statistics',
    # Biosciences
    'bio':                                     'Biosciences and Bioengineering',
    'bsbe':                                    'Biosciences and Bioengineering',
    'biosciences':                             'Biosciences and Bioengineering',
    'bioengineering':                          'Biosciences and Bioengineering',
    # Aerospace
    'aero':                                    'Aerospace Engineering',
    'aerospace':                               'Aerospace Engineering',
    # Systems & Control
    'syscon':                                  'Centre for Systems and Control',
    'systems and control':                     'Centre for Systems and Control',
    # GNR
    'csre':                                    'Centre of Studies in Resources Engineering',
    'geoinformatics':                          'Centre of Studies in Resources Engineering',
    'gnr':                                     'Centre of Studies in Resources Engineering',
    'resources engineering':                   'Centre of Studies in Resources Engineering',
    # Robotics
    'robotics':                                'Robotics',
    # IEOR
    'industrial engineering':                  'Industrial Engineering and Operations Research',
    'operations research':                     'Industrial Engineering and Operations Research',
    'ieor':                                    'Industrial Engineering and Operations Research',
}


def detect_minor_intent(query):
    if not query:
        print("empty query")
        return None, None

    ql = query.lower()
    matched = None
    best_len = 0

    for alias, branch in _MINOR_DEPT_ALIASES.items():
        if alias in ql:
            print("matched alias:", alias)

        if alias in ql and len(alias) > best_len:
            matched = branch
            best_len = len(alias)

    return matched, matched


def build_minor_candidates(branch: str, degree: str) -> list:
    """
    Return running-semester courses for the given
    branch, filtered to those offered this term.
    Each dict matches the desired_courses format expected by recommender().
    PS / Semantic scoring does not apply in minor mode.
    """
    if _df_minor.empty or not branch:
        return []

    mask_deg = _df_minor['Degree'] == degree
    if not mask_deg.any():
        mask_deg = _df_minor['Degree'] == 'B.Tech.'
    subset = _df_minor[mask_deg & (_df_minor['Branch'] == branch)]

    # Build a lookup of normalized running course codes -> actual course code in course_meta
    running_normalized = {k.replace(" ", "").upper(): k for k in course_meta.keys()}

    candidates, seen = [], set()
    for _, row in subset.iterrows():
        code = str(row['Course Code']).strip()
        if code.startswith('All ') or code in seen:
            continue
        seen.add(code)

        code_norm = code.replace(" ", "").upper()
        if code_norm not in running_normalized:          # not offered this semester → skip
            continue
        actual_code = running_normalized[code_norm]
        
        # Enforce Minor/Elective variant only for courses that have
        # historically had an M-tagged variant.

        # Running-semester divisions
        divs = course_divisions.get(actual_code, [])

        has_minor_division = any(d["is_minor"] for d in divs)
        has_regular_division = any(not d["is_minor"] for d in divs)

        # Curriculum requirement
        course_type = str(row["Type"]).strip().lower() if pd.notna(row["Type"]) else ""
        expects_minor = None
        if "minor" in course_type:
            expects_minor = True
        else:
            expects_minor = False

        # Only enforce variant matching if this course has ever existed as an M-tagged course.
        if code_norm in courses_with_minor_variant:

            if expects_minor and not has_minor_division:
                continue

            if not expects_minor and not has_regular_division:
                continue

        candidates.append({
            'code':         actual_code,
            'name':         _course_name_map.get(actual_code.replace(" ", "").upper(), _course_name_map.get(code.replace(" ", "").upper(), '')),
            'score':        0.0,
            'raw_rrf':      None,            # None → scores hidden in template
            'raw_ps':       None,
            'raw_ts':       None,
            'minor_type':   str(row['Type']).strip()    if pd.notna(row['Type'])    else 'Minor',
            'minor_remark': str(row['Remarks']).strip() if pd.notna(row['Remarks']) else '',
        })
    return candidates

# ── Comprehensive remark-abbreviation → full department name (dropdown value) ─
_REMARK_DEPT_ABBREVS: dict = {
    'AE':    'Aerospace Engineering',
    'BB':    'Biosciences and Bioengineering',
    'CE':    'Civil Engineering',
    'CL':    'Chemical Engineering',
    'CS':    'Computer Science and Engineering',
    'CH':    'Chemistry',
    'EE':    'Electrical Engineering',
    'EN':    'Energy Science and Engineering',
    'EP':    'Engineering Physics',
    'ES':    'Earth Sciences',
    'ESE':   'Environmental Science and Engineering',
    'GNR':   'Centre of Studies in Resources Engineering',
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

# Reverse map: full dept name → set of abbreviations used in remark text
_DEPT_TO_ABBREVS: dict = {}
for _abbr, _full in _REMARK_DEPT_ABBREVS.items():
    _DEPT_TO_ABBREVS.setdefault(_full, set()).add(_abbr)


def _sentence_applies(sentence: str, student_dept: str) -> bool:
    """
    Returns True if this remark sentence is relevant to student_dept.

    1. Find all '[ABBREV] students' tokens  →  positive_depts
    2. Find all 'non-[ABBREV]' tokens       →  negated_depts
    3. Remove negated from positive (guards against 'non-IEOR students'
       putting IEOR in the positive set via the word-boundary match).
    4. No recognised abbreviations in either set → universal → True.
    5. Only negated  → True when student is in NONE of negated_depts.
    6. Only positive → True when student is in ONE of positive_depts.
    7. Mixed / unknown dept → True (safe default).
    """
    s = sentence.strip()

    positive = (
        set(re.findall(r'\b([A-Z]{2,6})\s+students\b', s))
        & set(_REMARK_DEPT_ABBREVS)
    )
    negated = (
        set(re.findall(r'[Nn]on-([A-Z]{2,6})', s))
        & set(_REMARK_DEPT_ABBREVS)
    )
    positive -= negated   # e.g. 'non-IEOR students' must not land in positive

    if not positive and not negated:
        return True       # no dept marker → universal

    student_abbrevs = _DEPT_TO_ABBREVS.get(student_dept)
    if not student_abbrevs:
        return True       # unknown dept → show everything (safe default)

    if negated and not positive:
        return not bool(student_abbrevs & negated)

    if positive and not negated:
        return bool(student_abbrevs & positive)

    return True           # mixed case → safe default


def parse_minor_remark(remark: str, course_hist: list, department: str = '') -> dict:
    """
    Classify a minor-program remark into actionable display categories,
    filtered to only the sentences relevant to the student's department.

    Returns a dict with keys:
      prereq_unmet  – explicit prerequisite language + course not yet done
      warning       – exclusion / cannot-count language
      note          – general informational text
    """
    out: dict = {'prereq_unmet': None, 'warning': None, 'note': None}
    if not remark or str(remark).strip().lower() in ('nan', 'none', ''):
        return out
    remark = str(remark).strip()

    # ── Department-aware sentence filtering ──────────────────────────────────
    sentences = re.split(r'(?<=\.)\s+', remark)
    kept = [s for s in sentences if _sentence_applies(s, department)]
    if not kept:
        return out          # nothing relevant for this student's department
    remark = ' '.join(kept)
    # ────────────────────────────────────────────────────────────────────────

    rl = remark.lower()

    # 1. Explicit prerequisite language
    prereq_m = re.search(
        r'([A-Z]{2,3}\s?\d{3,4})\s+(?:is\s+(?:a\s+)?)?(?:mandatory\s+)?prerequisite\s+for'
        r'|must\s+(?:first\s+)?complete\s+([A-Z]{2,3}\s?\d{3,4})\s+before'
        r'|([A-Z]{2,3}\s?\d{3,4})\s+is\s+(?:a\s+)?pre-?req(?:uisite)?',
        remark, re.IGNORECASE
    )
    if prereq_m:
        codes = re.findall(r'[A-Z]{2,3}\s?\d{3,4}', prereq_m.group(0))
        norm_hist = {c.replace(' ', '').upper() for c in course_hist}
        missing   = [c for c in codes if c.replace(' ', '').upper() not in norm_hist]
        if missing:
            out['prereq_unmet'] = (
                f"Minor requirement not met. Complete {', '.join(missing)} first"
            )
        out['note'] = remark
        return out

    # 2. Exclusion / cannot-count language → ⚠ warning
    if re.search(
        r'cannot\s+be\s+counted|not\s+be\s+counted|excluded|not\s+allowed|'
        r'cannot\s+count|will\s+not\s+count',
        rl
    ):
        out['warning'] = remark
        return out

    # 3. Everything else → ℹ note
    out['note'] = remark
    return out

def recommender(student_id, Degree, year, department, desired_courses, manual_course_history=None, w_rrf=1, w_ps=0, prefer_minor_division = False):
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
    
    try:
        year_val = int(year)
    except (ValueError, TypeError):
        clean_sid = str(student_id).strip()
        if clean_sid and len(clean_sid) >= 2 and clean_sid[:2].isdigit():
            year_val = 2000 + int(clean_sid[:2])
        else:
            year_val = current_cal_year - 2 # Assume 3rd year default fallback
            
    year_in_program = academic_year_start - year_val + 1
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
        for sn in running_slot_map.get(code_orig.replace(" ", "").upper(), set()):
            core_slot_to_courses.setdefault(sn, []).append(code_orig)

    eligible_courses = []
    rejected_courses = []
    i_a_r_courses = []
    t_s_c_courses = []
    
    desired_courses = apply_grade_boost(desired_courses, w_ps)
    # 2. Loop over model-recommended courses
    seen_codes = set()
    for course in desired_courses:
        course_code = str(course["code"]).replace(" ", "").upper().strip()
        if course_code in seen_codes:
            continue
        seen_codes.add(course_code)

        # some course names are NaN for some reason, this deals with that 
        course_name = course.get("name")
        if not isinstance(course_name, str) or pd.isna(course_name) or not course_name.strip():
            course_name = _course_name_map.get(course_code, "")
        # ignores any courses with the below words (for future DAV members, remove this and see what semantic search gives to know why its there)
        blacklist = ["SEMINAR", "MINI PROJECT", "SUPERVISED", "BTP"]
        if any(term in course_name.upper() for term in blacklist):
            continue

        if re.search(r'\bPROJECT\s*(I{1,3}|[1-4])?\s*$', course_name.upper()):
           continue
        
        if norm_code(course_code) in all_core_codes:
            continue

        # Pre-compute division metadata once per course, filtering out divisions restricted for this student
        raw_divs = course_divisions.get(course_code, [])
        valid_divs = []
        for d in raw_divs:
            div_status = check_restriction(
                Degree=Degree,
                year=year,
                department=department,
                course_code=course_code,
                division=d['division']
            )
            if div_status == 'Valid' or 'year students' in div_status or div_status == 'Restricted':
                valid_divs.append((d, div_status))

        if not valid_divs:
            continue

        divs = [vd[0] for vd in valid_divs]
        has_minor  = any(d['is_minor'] for d in divs)
        minor_only = all(d['is_minor'] for d in divs) and bool(divs)

        # 3. Check restriction
        if norm_code(course_code) in course_hist_norm:
            continue

        r_status = 'Valid' if any(vd[1] == 'Valid' for vd in valid_divs) else valid_divs[0][1]
        
        if r_status != 'Valid':
            if 'year students' in r_status or r_status == 'Restricted':
                meta = course_meta.get(course_code, {})
                print(course_code, "GOING TO REJECTED")
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
                    "norm_rrf": course.get("norm_rrf"),
                    "norm_ps": course.get("norm_ps"),
                    "final_score": course.get("raw_ts"),
                    "grade_stats": grade_stats_db.get(course_code, None),
                    "minor_remark": course.get("minor_remark")
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

                if prefer_minor_division:
                    non_m = [d for d in divs_with_clash if d['is_minor']] or divs_with_clash
                else:
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
                        "norm_rrf": course.get("norm_rrf"),
                        "norm_ps": course.get("norm_ps"),
                        "final_score": course.get("raw_ts"),
                        "grade_stats": grade_stats_db.get(course_code, None),
                        "minor_remark": course.get("minor_remark")
                    })
                else:
                    reason = (p_remark if p_remark
                                else p_status if p_status == 'Instructor approval required'
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
                        "norm_rrf": course.get("norm_rrf"),
                        "norm_ps": course.get("norm_ps"),
                        "final_score": course.get("raw_ts"),
                        "grade_stats": grade_stats_db.get(course_code, None),
                        "minor_remark": course.get("minor_remark")
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
                    "norm_rrf": course.get("norm_rrf"),
                    "norm_ps": course.get("norm_ps"),
                    "final_score": course.get("raw_ts"),
                    "grade_stats": grade_stats_db.get(course_code, None),
                    "minor_remark": course.get("minor_remark")
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
                "norm_rrf": course.get("norm_rrf"),
                "norm_ps": course.get("norm_ps"),
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
    "credits":     meta.get("credits", 6),
    "default_idx": default_idx,
    "equiv": equiv_map.get(course_code.replace(' ', ''), {'regular': [], 'minor': []}),
    "raw_rrf": course.get("raw_rrf"),
    "raw_ps": course.get("raw_ps"),
    "norm_rrf": course.get("norm_rrf"),
    "norm_ps": course.get("norm_ps"),
    "final_score": course.get("raw_ts"),
    "grade_stats": grade_stats_db.get(course_code, None),
    "minor_remark": course.get("minor_remark")
})
    print("Eligible list:")
    print([c["code"] for c in eligible_courses])

    print("TSC list:")
    print([c["code"] for c in t_s_c_courses])

    print("IAR list:")
    print([c["code"] for c in i_a_r_courses])

    print("Rejected list:")
    print([c["code"] for c in rejected_courses])

    
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
