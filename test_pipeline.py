from model.recommender import get_candidate_courses
from eligibility.rules import recommender as eligibility_recommender

student_id = "24b2141"
degree = "B.Tech."
year = "2024"
department = "Mechanical Engineering"

query = "Machine Learning Foundational Math"

# Use top_k=60 because thats also in app.py, and test interest-only mode (w_rrf=1.0, w_ps=0.0)
desired_courses = get_candidate_courses(
    query=query,
    student_history=[],
    top_k=60,
    w_rrf=1.0,
    w_ps=0.0,
    degree=degree,
    year=year,
    department=department
)

print("\n=== TOP 20 RECOMMENDED ELECTIVES (Before Eligibility Gates) ===")
for idx, c in enumerate(desired_courses, 1):
    print(f"{idx}. {c['code']} - {c['name']} (Score: {c['raw_ts']:.4f}, RRF: {c['raw_rrf']:.4f})")
print("===============================================================\n")

final_courses = eligibility_recommender(
    student_id=student_id,
    Degree=degree,
    year=year,
    department=department,
    desired_courses=desired_courses,
    manual_course_history=[]
)

# Print the full categorized result dictionary
import json
print(json.dumps(final_courses, indent=2))
