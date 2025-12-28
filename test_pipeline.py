from model.recommender import get_course_codes_from_query
from eligibility.rules import recommender as eligibility_recommender

student_id = "24b0350"
degree = "B.Tech."
year = "2"
department = "Chemical Engineering"

query = "machine learning"

desired_courses = get_course_codes_from_query(query, top_k=5)

final_courses = eligibility_recommender(
    student_id=student_id,
    Degree=degree,
    year=year,
    department=department,
    desired_courses=desired_courses
)

print("Recommended courses:", final_courses)
