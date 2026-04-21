"""
Configuration - Constants and Paths
"""

import os
from pathlib import Path

# Project root (2 levels up from this file)
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Data paths
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'customer_support_tickets.csv'

# Output paths
MODELS_DIR = ROOT_DIR / 'models'
REPORTS_DIR = ROOT_DIR / 'reports'

# Model filenames
TYPE_MODEL_PATH = MODELS_DIR / 'ticket_type_pipeline.pkl'
TYPE_ENCODER_PATH = MODELS_DIR / 'ticket_type_encoder.pkl'
PRIORITY_MODEL_PATH = MODELS_DIR / 'ticket_priority_pipeline.pkl'
PRIORITY_ENCODER_PATH = MODELS_DIR / 'ticket_priority_encoder.pkl'

# Report filenames
TYPE_CM_PATH = REPORTS_DIR / 'confusion_matrix_ticket_type.png'
PRIORITY_CM_PATH = REPORTS_DIR / 'confusion_matrix_ticket_priority.png'

# Ensure dirs exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Model parameters
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Target column names
TARGET_TYPE = 'Ticket Type'
TARGET_PRIORITY = 'Ticket Priority'
