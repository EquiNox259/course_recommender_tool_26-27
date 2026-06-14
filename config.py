import os

# Absolute path to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw datasets (manual / academic data)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

COURSES_HISTORY_PATH = os.path.join(DATASET_DIR, "Student_data.csv")
RUNNING_COURSES_PATH = os.path.join(DATASET_DIR, "courses_autumn_2025-2026.csv")
PREREQ_PATH = os.path.join(DATASET_DIR, "Prerequisite_new.xlsx")
CORE_COURSES_PATH = os.path.join(DATASET_DIR, "ASC_Core_Courses.csv")
SEMESTER = 'Spring'

# Model artifacts
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_DATA_DIR = os.path.join(MODEL_DIR, "data")
MODEL_ASSETS_DIR = os.path.join(MODEL_DIR, "models")
