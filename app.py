from flask import Flask, render_template, request
from model.recommender import get_candidate_courses
from eligibility.rules import recommender as eligibility_recommender, get_llm_recommendations

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    eligible = None
    rejected = None
    i_a_r = None
    t_s_c = None
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
        from model.recommender import expand_abbreviations
        expanded_interest = expand_abbreviations(interest)
        desired_courses = get_candidate_courses(expanded_interest, top_k=60)

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
            raw_eligible = output["eligible"]
            raw_iar = output["i_a_r"]
            raw_tsc = output["t_s_c"]
            raw_rejected = output["rejected"]

            # Use LLM to select the best (max 15)  courses from the combined lists
            selected_codes = get_llm_recommendations(
                interest=expanded_interest,
                department=department,
                degree=degree,
                course_history=course_history,
                eligible_list=raw_eligible,
                iar_list=raw_iar,
                rejected_list=raw_rejected,
                tsc_list=raw_tsc,
                x=15  #x is the max courses the LLM recommends
            )

            # Filter each list to only include courses selected by the LLM
            eligible = [c for c in raw_eligible if c['code'].replace(" ", "").upper() in selected_codes]
            i_a_r = [c for c in raw_iar if c['code'].replace(" ", "").upper() in selected_codes]
            t_s_c = [c for c in raw_tsc if c['code'].replace(" ", "").upper() in selected_codes]
            rejected = [c for c in raw_rejected if c['code'].replace(" ", "").upper() in selected_codes]

    return render_template(
        "index.html",
        eligible=eligible,
        i_a_r = i_a_r,
        t_s_c = t_s_c,
        rejected=rejected,
        course_history=course_history,
        need_manual_history=need_manual_history
    )

@app.route("/docs")
def docs():
    return render_template("documentation.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
