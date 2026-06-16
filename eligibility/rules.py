import ast
import numpy as np
import re
import pandas as pd
from config import RUNNING_COURSES_PATH, PREREQ_PATH, COURSES_HISTORY_PATH

data = pd.read_csv(COURSES_HISTORY_PATH)
df = pd.read_csv(RUNNING_COURSES_PATH)
df_prereq = pd.read_excel(PREREQ_PATH)

# Build quick lookup: course_code -> slot & instructor, with description upon expansion
course_meta = {}

for _, row in df.iterrows():
    code = row["Course Code"]
    course_meta[code] = {
        "slot": row.get("Slot", "N/A"),
        "instructor": row.get("Instructor", "N/A"),
        "description": row.get("Description", "")
    }

import re

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
    if isinstance(i, str):
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

def check_restriction(Degree,year,department,course_code, data=data_modified):
    year=str(year)
    # Check if course_code exists in the data
    if course_code not in data['Course Code'].values:
        return 'Not in available data'

    # Fetch restriction value
    restrictions = data.loc[data['Course Code'] == course_code, 'Restriction'].values[0]

    # If no restriction (a string 'No restrictions')
    if isinstance(restrictions, str):
        if restrictions.lower() == 'no restrictions':
            return 'Valid'
        else:
            # You might parse restriction string here if needed
            return 'Restricted'
    else:
        # Handle degree-based restrictions
        if Degree in ['B.Tech.', 'M.Tech.', 'Ph.D.', 'M.Sc.', 'Dual Degree (B.Tech. + M.Tech.)', 'B.S.', 'B.Des.']:

            '''restrictions array have format every element contain first year then dept,then degree, then tell allowed or deny'''

            allowed_groups = []
            for j in restrictions:
                if j[3] == 'Allowed':
                    parts = []
                    if j[2] != 'ALL': parts.append(j[2])
                    if j[1] != 'ALL': parts.append(j[1])
                    if j[0] != 'ALL': parts.append(f"{j[0]}")
                    allowed_groups.append(', '.join(parts) if parts else 'specific students')
            if allowed_groups:
                restricted_msg = f"Open only to {' / '.join(allowed_groups)} students."
            else:
               restricted_msg = 'Restricted'

            for i in restrictions:
              if i[0]==year or i[0]=='ALL':
                if i[1]==department or i[1]=='ALL':
                  if i[2]==Degree or i[2]=='ALL':
                    if i[3]=='Allowed':
                      return 'Valid'
                    else:
                      return restricted_msg
        else:
            return 'Invalid Degree'
    return restricted_msg

#check prerequsite
def check_prereq(course_code,course_hist,data=df_prereq):
    # Check if course_code exists in the data
    #course_hist is course history of student
    if course_code not in data['CourseCode'].values: # No prereq
        return 'Valid'

    # Fetch prerequsite course code and instructor approval status
    prereq = data.loc[data['CourseCode'] == course_code, 'Courses'].values[0]
    approval = data.loc[data['CourseCode'] == course_code, 'InstructorConsent'].values[0]
    if isinstance(prereq, str):
        prereq = re.sub(r'\s+(\d)', r'\1', prereq).upper()

    pattern = r'(?:[A-Z]{2,3}\d{3,4})'
    
    #Blank cells => Instructor approval required
    if pd.isna(prereq):
        return 'Instructor approval required'
    # if single prereq
    elif re.match(pattern, prereq) and (len(prereq) in [5, 6, 7]):
        if prereq in course_hist:
          if approval == 'Required':
            return 'Instructor approval required'
          elif approval == 'Conditional':
            return 'Instructor approval is conditional'
          else:
            return 'Valid'
        else:
          return f'Prerequisite not met. You need to complete {prereq}.'
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
                return 'Instructor approval required'
            elif approval == 'Conditional':
                return 'Instructor approval is conditional'
            else:
                return 'Valid'
        return f'Prerequisite not met. You need to complete {prereq}.'

      # if prereq only contain AND
      elif 'OR' not in l and 'AND' in l:
        for i in pre_course:
          if i not in course_hist:
            return f'Prerequisite not met. You need to complete {prereq}.'
          if approval == 'Required':
            return 'Instructor approval required'
          elif approval == 'Conditional':
            return 'Instructor approval is conditional'
          else:
            return 'Valid'

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
                return 'Instructor approval required'
            elif approval == 'Conditional':
                return 'Instructor approval is conditional'
            else:
                return 'Valid'

      return f'Prerequisite not met. You need to complete {prereq}.'
    


def recommender(student_id, Degree, year, department, desired_courses, manual_course_history=None, w_rrf=0.5, w_ps=0.5):
    """
    student_id       : string (email prefix)
    Degree           : string (e.g. 'B.Tech.')
    year             : string or int (e.g. '2' or 2)
    department       : string (e.g. 'ME', 'CS')
    desired_courses  : list of course codes (from NLP / preference model)

    returns          : list of eligible course codes
    """
    print(f"[CHECKPOINT 3 - RULES.PY] eligibility_recommender received parameters:")
    print(f" -> Passed w_rrf: {w_rrf}")
    print(f" -> Passed w_ps:  {w_ps}")
    print("\n\n\n")

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

    eligible_courses = []
    rejected_courses = []
    i_a_r_courses = []

    # 2. Loop over model-recommended courses
    for course in desired_courses:
        course_code = course["code"]
        print(f"total score: {course.get('raw_ts')}")
        # 3. Check restriction
        if course_code.replace(" ", "").upper() in course_hist:
           continue
        r_status = check_restriction(
            Degree=Degree,
            year=year,
            department=department,
            course_code=course_code
        )

        if r_status != 'Valid':
            meta = course_meta.get(course_code, {})

            rejected_courses.append({
    "code": course_code,
    "name": course["name"],
    "slot": meta.get("slot", "N/A"),
    "instructor": meta.get("instructor", "N/A"),
    "description": meta.get("description", ""),
    "reason": r_status
})
            continue

        # 4. Check prerequisite
        p_status = check_prereq(
            course_code=course_code,
            course_hist=course_hist
        )

        if p_status != 'Valid':
            meta = course_meta.get(course_code, {})

            if p_status == 'Instructor approval required':
               i_a_r_courses.append({
                  "code": course_code,
                  "name": course["name"],
                  "slot": meta.get("slot", "N/A"),
                  "instructor": meta.get("instructor", "N/A"),
                  "description": meta.get("description", ""),
                  "raw_rrf": course.get("raw_rrf"),
                  "raw_ps": course.get("raw_ps"),
                  "raw_ts": course.get("raw_ts"),
               })

            elif p_status == 'Instructor approval is conditional':
               i_a_r_courses.append({
                  "code": course_code,
                  "name": course["name"],
                  "slot": meta.get("slot", "N/A"),
                  "instructor": meta.get("instructor", "N/A"),
                  "description": meta.get("description", ""),
                  "reason": p_status + '. Contact them for more info.',
                  "raw_rrf": course.get("raw_rrf"),
                  "raw_ps": course.get("raw_ps"),
                  "raw_courses": course.get("raw_ts"),
               })

            else:
                rejected_courses.append({
    "code": course_code,
    "name": course["name"],
    "slot": meta.get("slot", "N/A"),
    "instructor": meta.get("instructor", "N/A"),
    "description": meta.get("description", ""),
    "reason": p_status,
    "raw_rrf": course.get("raw_rrf"),
    "raw_ps": course.get("raw_ps"),
    "raw_courses": course.get("raw_ts"),
})
            continue
        # 5. Passed all checks
        meta = course_meta.get(course_code, {})

        eligible_courses.append({
    "code": course_code,
    "name": course["name"],
    "slot": meta.get("slot", "N/A"),
    "instructor": meta.get("instructor", "N/A"),
    "description": meta.get("description", ""),
    "raw_rrf": course.get("raw_rrf"),
    "raw_ps": course.get("raw_ps"),
    "final_score": course.get("raw_ts"),
})

    return {
    "course_history": course_hist,
    "history_source": history_source,
    "eligible": eligible_courses,
    "i_a_r": i_a_r_courses,
    "rejected": rejected_courses
}


'''list_= recommender('24b0350','B.tech','2024','Chemical Engineering',['CS 213','SI 505','SI 427','CL 603','CL 688','SI 402'])
print(list_)'''
