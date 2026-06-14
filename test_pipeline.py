from model.recommender import get_candidate_courses
from eligibility.rules import recommender as eligibility_recommender

student_id = "24b2141"
degree = "B.Tech."
year = "2024"
department = "Mechanical Engineering"

query = "ML, robotics"

# Use top_k=60 because thats also in app.py
desired_courses = get_candidate_courses(query, top_k=60)

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
