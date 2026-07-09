import sys
import os
import unittest
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eligibility.rules import (
    check_restriction,
    check_prereq,
    get_core_courses_for_bucket,
    recommender,
    build_minor_candidates
)
from model.recommender import get_candidate_courses, build_candidate_pool

class TestRecommenderEngine(unittest.TestCase):

    def test_space_normalization(self):
        # 1. Check prerequisite check space normalization
        status, _, _ = check_prereq("CS 409", ["CS 101", "CS 105"])
        self.assertEqual(status, "Instructor approval is conditional")
        
        status_spaced, _, _ = check_prereq("CS 409 ", ["CS 101", "CS 105 "])
        self.assertEqual(status_spaced, "Instructor approval is conditional")

    def test_ambiguous_restriction_resolved_to_restricted(self):
        # DE 328 has allowed rules for 2023 and 2024 batches, deny for others
        status = check_restriction("B.Tech.", "2025", "Mechanical Engineering", "DE 328")
        self.assertEqual(status, "Available for 3rd and 4th year students only")

    def test_division_aware_restriction_cse_minor(self):
        # CS 409 has major restricted regular division and open minor division (M)
        # 1. Regular check (is_minor=False) -> should be restricted for non-CSE student
        reg_status = check_restriction("B.Tech.", "2025", "Chemical Engineering", "CS 409", is_minor=False)
        self.assertEqual(reg_status, "Open only to B.Tech., Computer Science and Engineering")

        # 2. Minor check (is_minor=True) -> should be Valid
        minor_status = check_restriction("B.Tech.", "2025", "Chemical Engineering", "CS 409", is_minor=True)
        self.assertEqual(minor_status, "Valid")

    def test_non_cse_core_courses_recommended_as_electives(self):
        # CS 105 (Discrete Structures) is core for CSE but should be recommended for Chemical students
        candidates, _, _, _, _ = get_candidate_courses(
            query="discrete structures",
            student_history=[],
            top_k=5,
            w_rrf=1.0,
            w_ps=0.0,
            degree="B.Tech.",
            year="2025",
            department="Chemical Engineering"
        )
        candidate_codes = [c["code"].replace(" ", "").upper() for c in candidates]
        self.assertIn("CS105", candidate_codes)

    def test_minor_division_only_recommended(self):
        # Chem student searching for cryptography should get CS 409 but ONLY the minor division M
        candidates, _, _, _, _ = get_candidate_courses(
            query="Cryptograpy course",
            student_history=[],
            top_k=20,
            w_rrf=1.0,
            w_ps=0.0,
            degree="B.Tech.",
            year="2025",
            department="Chemical Engineering"
        )
        output = recommender(
            student_id="testing",
            Degree="B.Tech.",
            year="2025",
            department="Chemical Engineering",
            desired_courses=candidates,
            manual_course_history=["CS 101", "CS 105"],
        )
        
        # Check if CS 409 exists in Consent Required (requires consent since it requires approval)
        consent_codes = [c["code"].replace(" ", "").upper() for c in output["i_a_r"]]
        self.assertIn("CS409", consent_codes)

        # Verify only minor division 'M' is recommended
        cs409_entry = next(c for c in output["i_a_r"] if c["code"].replace(" ", "").upper() == "CS409")
        self.assertEqual(len(cs409_entry["divisions"]), 1)
        self.assertEqual(cs409_entry["divisions"][0]["division"], "M")

if __name__ == "__main__":
    unittest.main()
