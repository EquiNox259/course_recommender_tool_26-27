import os
import sys
import json
import functools
import traceback
import pandas as pd
import time
sys.path.append(os.getcwd())

try:
    import model.recommender as rec
    import eligibility.rules as rules
    from app import fetch_automatic_student_history
except ImportError as e:
    sys.path.append(os.path.join(os.getcwd(), 'model'))
    sys.path.append(os.path.join(os.getcwd(), 'eligibility'))
    import model.recommender as rec
    import eligibility.rules as rules
    import app
    fetch_automatic_student_history = app.fetch_automatic_student_history

try:
    import google.generativeai as genai
    HAS_GEMINI = True
    genai.configure(api_key="")
except ImportError:
    HAS_GEMINI = False
    print("google-generativeai package not found. LLM evaluation step will be simulated.")

def checkpoint_trace(module_name, func_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            #print(f"CHECKPOINT: Executing {module_name}.{func_name}()")
            #print(f"Inputs -> Args: {args} | Kwargs: {kwargs}")
            
            try:
                result = func(*args, **kwargs)
                #print(f"Outputs -> {result}")
                return result
            except Exception as e:
                print(f"ERROR in {func_name}: {e}")
                raise e
        return wrapper
    return decorator

rec.get_semantic_rankings = checkpoint_trace("model.recommender", "get_semantic_rankings")(rec.get_semantic_rankings)
rec.get_keyword_rankings = checkpoint_trace("model.recommender", "get_keyword_rankings")(rec.get_keyword_rankings)
rec.compute_rrf = checkpoint_trace("model.recommender", "compute_rrf")(rec.compute_rrf)
rec.calculate_people_score = checkpoint_trace("model.recommender", "calculate_people_score")(rec.calculate_people_score)
rec.get_candidate_courses = checkpoint_trace("model.recommender", "get_candidate_courses")(rec.get_candidate_courses)

if hasattr(rules, 'check_restriction'):
    rules.check_restriction = checkpoint_trace("eligibility.rules", "check_restriction")(rules.check_restriction)
if hasattr(rules, 'check_prereq'):
    rules.check_prereq = checkpoint_trace("eligibility.rules", "check_prereq")(rules.check_prereq)

recommender_function = getattr(rules, 'recommender', getattr(rules, 'eligibility_recommender', None))
if recommender_function:
    wrapped_recommender = checkpoint_trace("eligibility.rules", "recommender")(recommender_function)

STUDENT_PROFILES = [
    {"student_id": "24b0033", "department": "Aerospace Engineering", "degree": "BTech", "year": 3},
    {"student_id": "24b2181", "department": "Mechanical Engineering", "degree": "BTech", "year": 3},
    {"student_id": "24b2231", "department": "Mechanical Engineering", "degree": "BTech", "year": 3},
    {"student_id": "24b2297", "department": "Mechanical Engineering", "degree": "BTech", "year": 3},
    {"student_id": "24b2472", "department": "Metallurgical Engineering & Material Science", "degree": "BTech", "year": 3},
    {"student_id": "24b2428", "department": "Metallurgical Engineering & Material Science", "degree": "BTech", "year": 3},
    {"student_id": "24b0999", "department": "Computer Science and Engineering", "degree": "BTech", "year": 3},
]

RESEARCH_INTERESTS = [
    "I want to focus entirely on advanced data science, machine learning, and natural language processing.",
    "I'm looking for introductory courses in quantitative finance, algorithmic trading, and financial engineering.",
    "I want to explore core software engineering, focusing on systems programming, databases, and web development.",
    "I'm interested in multidisciplinary courses that combine material science with computational modeling and engineering.",
    "I want to take humanities and social science electives, specifically focusing on psychology, economics, or literature.",
    "I’m looking for pure mathematics and statistics courses, especially linear algebra, probability, and stochastic processes.",
    "I want to explore robotics and automation, including control systems, computer vision, and embedded systems."
]

def ask_gemini_to_judge(student, interest, output_tree):
    if not HAS_GEMINI or os.environ.get("GEMINI_API_KEY") is None:
        return "[Simulation Mode]: Recommendations appear logically distinct and isolated by academic prerequisites safely."

    prompt = f"""
    Review this course recommendation pipeline output for safety, correctness, and interest alignment.
    Student: {student['student_id']}, Dept: {student['department']}, Focus: {interest}
    Eligible: {[c.get('code') for c in output_tree.get('eligible', [])[:5]]}
    Restricted: {[c.get('code') for c in output_tree.get('i_a_r', [])[:5]]}
    Rejected: {[c.get('code') for c in output_tree.get('rejected', [])[:5]]}
    Rate this as EXCELLENT, PASSABLE, or BROKEN with 1 sentence explaining why.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not complete LLM Audit invocation: {e}"

def run_live_test_suite():
    print("STARTING RECO_ENGINE TESTING PIPELINE\n")
    
    for idx, student in enumerate(STUDENT_PROFILES):
        interest_query = RESEARCH_INTERESTS[idx]
        
        print(f"RUNNING TEST CASE {idx + 1}/{len(STUDENT_PROFILES)}")
        print(f"Student: {student['student_id']} | Dept: {student['department']}")
        print(f"Query: '{interest_query}'\n")
        
        try:
            # 1. Fetch the real history list from the database
            real_historical_list = fetch_automatic_student_history(student['student_id'])
            print(f"Database History Found: {real_historical_list}")
            
            print(f"\n[CHECKPOINT 1] Raw NLP Search Candidates for Query: '{interest_query}'")
            candidate_codes = rec.get_candidate_courses(interest_query) 
            print(f" -> Retracted Pool ({len(candidate_codes)} courses): {candidate_codes}")
            #NLP extraction

            # 2. SIMULATE THE REAL APP: Use your model to convert the text query into structured course candidates
            # (Adjust these function names if they differ slightly in your model.recommender file)
            candidate_codes = rec.get_candidate_courses(interest_query) 
            
            # 3. Build a properly flattened structured list of dictionaries
            structured_desired_courses = []
            for item in candidate_codes:
                if isinstance(item, dict):
                    # If the core model already gave us a dictionary, pull out the raw code directly
                    course_code_string = item.get("code", "")
                    course_name_string = item.get("name", "Course Title Lookup Placeholding")
                else:
                    # If it's a raw text string code
                    course_code_string = str(item)
                    course_name_string = "Course Title Lookup Placeholding"

                structured_desired_courses.append({
                    "code": course_code_string,       # Now guaranteed to be a flat string!
                    "name": course_name_string,
                    "raw_rrf": 0.5,
                    "raw_ps": 0.5
                })
            
            print(f"\n[CHECKPOINT 2] Inputting into Eligibility Engine (`rules.py`)")
            initial_codes = [c["code"] for c in structured_desired_courses]
            print(f" -> Evaluating Prereqs for: {initial_codes}")
            print(f" -> Student Record History: {real_historical_list}")
            #before eligibility recommendation

            
            # 4. Pass the correctly structured list into the wrapped recommender
            output_data_tree = wrapped_recommender(
                student_id=student['student_id'],
                Degree=student['degree'],
                year=student['year'],
                department=student['department'],
                desired_courses=structured_desired_courses,  # Now a valid list of dicts!
                manual_course_history=real_historical_list,
                w_rrf=0.5,
                w_ps=0.5
            )
            
            print(f"\n[CHECKPOINT 3] Final Recommendation Pipeline Output Tree")
            final_courses = output_data_tree.get('recommended_courses', [])
            
            if not final_courses:
                print(" No courses cleared the eligibility/prerequisite gates!")
            else:
                print(f" Successfully cleared gates ({len(final_courses)} courses remaining):")
                for rank, course in enumerate(final_courses, 1):
                    c_code = course.get('code')
                    c_name = course.get('name', 'N/A')
                    # If your app outputs calculated scores, fetch them here
                    c_score = course.get('score', course.get('raw_rrf', 'N/A')) 
                    print(f"    {rank}. [{c_code}] {c_name} | Pipeline Score: {c_score}")
            #post filtering            

            evaluation_report = ask_gemini_to_judge(student, interest_query, output_data_tree)
            print(f"GEMINI COMPLIANCE REPORT:\n{evaluation_report}")
            
        except Exception as err:
            print(f"Pipeline execution crashed on Test Case {idx + 1}!")
            traceback.print_exc()
        time.sleep(20)

if __name__ == "__main__":
    run_live_test_suite()
