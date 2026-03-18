"""
Configuration module for COVID-19 ML Project.
Contains all constants and settings.
"""
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Base Paths (Assuming this is run from project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_PATH = BASE_DIR / "data" / "raw" / "CovidData.csv"

# Output Paths
OUTPUT_FIGURES_DIR = BASE_DIR / "outputs" / "figures"
OUTPUT_MODELS_DIR = BASE_DIR / "outputs" / "models"

# Ensure output directories exist
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model Hyperparameters and Configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_MAX_ITER = 1000
K_BEST_FEATURES = 10

# Missing Value Definitions
MISSING_VALUES = [97, 98, 99]

# Columns to drop missing (97/99) values from
COLS_TO_DROP_MISSING = [
    "PNEUMONIA", "DIABETES", "COPD", "ASTHMA", "INMSUPR", 
    "HIPERTENSION", "OTHER_DISEASE", "CARDIOVASCULAR", 
    "OBESITY", "RENAL_CHRONIC", "TOBACCO"
]

# Columns to completely remove from the final dataset before modeling
COLS_TO_DROP_FINAL = [
    "DATE_DIED", "USMER", "MEDICAL_UNIT", "INTUBED", "ICU", "CLASIFFICATION_FINAL"
]

# Target Variable
TARGET_COLUMN = "DEATH"
