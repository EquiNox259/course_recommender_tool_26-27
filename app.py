from flask import Flask, render_template, request
from model.recommender import get_candidate_courses
from eligibility.rules import recommender as eligibility_recommender

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    eligible = None
    rejected = None
    course_history = None
    need_manual_history = False

    if request.method == "POST":
        # --- Read inputs ---
        student_id = request.form.get("student_id")
        degree = request.form.get("degree")
        year = request.form.get("year")
        department = request.form.get("department")
        interest = request.form.get("interest")

        # --- NLP search ---
        desired_courses = get_candidate_courses(interest, top_k=10)

        # --- Manual history (second phase) ---
        manual_history_raw = request.form.get("manual_history")
        manual_course_history = None

        if manual_history_raw:
            manual_course_history = [
                c.strip().upper()
                for c in manual_history_raw.split(",")
                if c.strip()
            ]

        # --- Eligibility engine ---
        output = eligibility_recommender(
            student_id=student_id,
            Degree=degree,
            year=year,
            department=department,
            desired_courses=desired_courses,
            manual_course_history=manual_course_history
        )

        # --- Branch: ask for manual history ---
        if "need_manual_history" in output:
            need_manual_history = True
        else:
            course_history = output["course_history"]
            eligible = output["eligible"]
            rejected = output["rejected"]

    return render_template(
        "index.html",
        eligible=eligible,
        rejected=rejected,
        course_history=course_history,
        need_manual_history=need_manual_history
    )

if __name__ == "__main__":
    app.run(debug=True)
