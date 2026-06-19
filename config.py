import os

# Absolute path to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw datasets (manual / academic data)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

COURSES_HISTORY_PATH = os.path.join(DATASET_DIR, "Student_data_combined.csv")
RUNNING_COURSES_PATH = os.path.join(DATASET_DIR, "courses_autumn_2025-2026.csv")
PREREQ_PATH = os.path.join(DATASET_DIR, "Prerequisite_new.xlsx")
CORE_COURSES_PATH = os.path.join(DATASET_DIR, "ASC_Core_Courses.csv")
SEMESTER = 'Autumn'
GRADES_2024_PATH = os.path.join(DATASET_DIR, "Grades2024Autumn.csv")
GRADES_2025_PATH = os.path.join(DATASET_DIR, "Grades2025Autumn.csv")
FEEDBACK_PATH = os.path.join(DATASET_DIR, "feedback.csv")

# Model artifacts
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_DATA_DIR = os.path.join(MODEL_DIR, "data")
MODEL_ASSETS_DIR = os.path.join(MODEL_DIR, "models")

# Authentication
SMTP_HOST = "smtp-auth.iitb.ac.in"
SMTP_PORT = 587
SMTP_USER = "25b2494"   # sending account
SMTP_PASS = "3f846f6c2e74e0cdd1aa7bb9fb554711"       # Gmail App Password
FLASK_SECRET = "7b374713abb88814fff4191775bf1f11f0575c710d4aaabff79a550b975f2428"
OTP_EXPIRY_SEC = 600                     # 10 minutes