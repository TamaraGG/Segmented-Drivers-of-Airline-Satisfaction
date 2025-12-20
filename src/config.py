import numpy as np

# ==========================================
# 1. General
# ==========================================
RANDOM_STATE = 42
N_JOBS = -1  # Использовать все ядра процессора

# Path's
DATA_PATH = "dataset/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
MODELS_SAVE_PATH = "models/"
REPORTS_PATH = "reports/"

# ==========================================
# 2. Data
# ==========================================
TARGET_COL = "satisfaction"
ID_COLS = ["Unnamed: 0", "id"]

DELAY_INPUT_COLS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
DELAY_OUTPUT_COL = "Total Delay Log"

# services (for 0 -> NaN)
SERVICE_COLS = [
    'Inflight wifi service',
    'Departure/Arrival time convenient',
    'Ease of Online booking',
    'Gate location',
    'Food and drink',
    'Online boarding',
    'Seat comfort',
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Inflight service',
    'Cleanliness'
]

# for auto Ordinal Encoding (Gender, Customer Type)
AUTO_ENCODING_COLS = ['Gender', 'Customer Type', 'Type of Travel']

# ==========================================
# 3. Mappings
# ==========================================

TARGET_MAP = {
    'neutral or dissatisfied': 0,
    'satisfied': 1
}

CLASS_MAP = {
    'Eco': 0,
    'Eco Plus': 1,
    'Business': 2
}

ENCODER_MANUAL_CONFIG = {
    'Class': {
        'Map': CLASS_MAP,
        'Suffix': '_Encoded'
    }
}

# ==========================================
# 4. Segmentation
# ==========================================


TYPE_BIZ = 'Business travel'
TYPE_PERS = 'Personal Travel'

CLASS_ECO = 'Eco'
CLASS_ECO_PLUS = 'Eco Plus'
CLASS_BIZ = 'Business'

ALL_TYPES = [TYPE_BIZ, TYPE_PERS]
ALL_CLASSES = [CLASS_ECO, CLASS_ECO_PLUS, CLASS_BIZ]

SEGMENT_CONFIGS = []

# --- GLOBAL SEGMENT---
SEGMENT_CONFIGS.append({
    'name': 'Global_All_Data',
    'filter': lambda df: slice(None),
    'drop_cols': ['Class']
})

# --- 2. MACRO SEGMENTS ---
for t_type in ALL_TYPES:
    SEGMENT_CONFIGS.append({
        'name': f"MACRO_{t_type.replace(' ', '_')}",
        'filter': lambda df, t=t_type: df['Type of Travel'] == t,
        'drop_cols': ['Type of Travel']
    })

# --- 3. MICRO SEGMENTS ---
for t_type in ALL_TYPES:
    for t_class in ALL_CLASSES:
        safe_name = f"MICRO_{t_type.replace(' ', '_')}_{t_class.replace(' ', '_')}"

        SEGMENT_CONFIGS.append({
            'name': safe_name,
            'filter': lambda df, t=t_type, c=t_class: (df['Type of Travel'] == t) & (df['Class'] == c),
            'drop_cols': ['Type of Travel', 'Class', 'Class_Encoded']
        })

MIN_SEGMENT_SIZE = 100

# ==========================================
# 5. Model Params
# ==========================================
# grid for GridSearchCV
XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    # scale_pos_weight
}

XGB_FIXED_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
    'n_jobs': 1
}