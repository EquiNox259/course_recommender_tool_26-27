import os
import json, tempfile
from dotenv import load_dotenv
load_dotenv()

# Absolute path to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw datasets (manual / academic data)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

COURSES_HISTORY_PATH = os.path.join(DATASET_DIR, "Student_data_combined.csv")
RUNNING_COURSES_PATH = os.path.join(DATASET_DIR, "courses_autumn_2026-2027.csv")
RUNNING_COURSES_PATH_ALT = os.path.join(DATASET_DIR, "courses_spring_2025-2026.csv")
PREREQ_PATH = os.path.join(DATASET_DIR, "Prerequisite_2026-27.xlsx")
CORE_COURSES_PATH = os.path.join(DATASET_DIR, "ASC_Core_Courses.csv")
MINOR_COURSES_PATH = os.path.join(DATASET_DIR, "ASC_Minor_Courses.csv")
HIGH_DEMAND_COURSES_PATH = os.path.join(DATASET_DIR, "high_demand_courses_2026-27.csv")
HIGH_DEMAND_CRITERIA_PATH = os.path.join(DATASET_DIR, "high_demand_criteria_2026-27.csv")
PREREG_MINOR_PATH = os.path.join(DATASET_DIR, "ASC_Prereg_Minor_2026-27.csv")
SEMESTER = 'Autumn'
GRADES_2024_PATH = os.path.join(DATASET_DIR, "Grades2024Autumn.csv")
GRADES_2025_PATH = os.path.join(DATASET_DIR, "Grades2025Autumn.csv")
FEEDBACK_PATH = os.path.join(DATASET_DIR, "feedback.csv")

# Model artifacts
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_DATA_DIR = os.path.join(MODEL_DIR, "data")
MODEL_ASSETS_DIR = os.path.join(MODEL_DIR, "models")

# Authentication
SMTP_HOST     = os.environ["SMTP_HOST"]
SMTP_PORT     = int(os.environ["SMTP_PORT"])
SMTP_USER     = os.environ["SMTP_USER"]
SMTP_PASS     = os.environ["SMTP_PASS"]
FLASK_SECRET  = os.environ["FLASK_SECRET"]
OTP_EXPIRY_SEC = int(os.environ.get("OTP_EXPIRY_SEC", 600))                    # 10 minutes
DEV_BYPASS_OTP  = False

# Feedback Google Sheet
_gsheets_json = os.environ.get("GSHEETS_SERVICE_ACCOUNT_JSON")
if _gsheets_json:
    _tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    _tmp.write(_gsheets_json)
    _tmp.close()
    GSHEETS_CREDENTIALS_PATH = _tmp.name
else:
    GSHEETS_CREDENTIALS_PATH = os.environ.get("GSHEETS_CREDENTIALS_PATH", 
                                os.path.join(BASE_DIR, "gsheets_service_account.json"))
    
GSHEETS_SPREADSHEET_NAME = os.environ.get("GSHEETS_SPREADSHEET_NAME", "Course Recommender Feedback")

# Frontend
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://127.0.0.1:5000")

# Pre-registration
PREREG_MODE = True
