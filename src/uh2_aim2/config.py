"""Configuration for UH2 AIM2 behavioral QC pipeline."""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_PATH = "/oak/stanford/groups/russpold/data/uh2/aim2"
BEHAVIOR_PATH = os.path.join(BASE_PATH, "behavioral_data")
BEHAVIOR_QC_PATH = os.path.join(BEHAVIOR_PATH, "behavioral_qc")
BIDS_PATH = os.path.join(BASE_PATH, "BIDS")
EVENT_FILES_PATH = os.path.join(BEHAVIOR_PATH, "event_files")
PROCESSED_PATH = os.path.join(BEHAVIOR_PATH, "processed")
SUBJECT_DATA_PATH = os.path.join(BEHAVIOR_PATH, "aim2_incl_dropped")

# =============================================================================
# TASKS
# =============================================================================
TASKS = ["discountFix", "motorSelectiveStop", "stopSignal", "manipulationTask"]

# Task conditions for analysis
CONDITIONS = {
    "stopSignal": ["go", "stop_failure"],
    "motorSelectiveStop": ["noncrit_signal", "noncrit_nosignal", "crit_stop_failure", "crit_go"],
    "discountFix": ["smaller_sooner", "larger_later"],
    "manipulationTask": ["future_valence", "future_neutral", "present_valence", "present_neutral"],
}

# Column to use for trial type (most tasks use trial_type)
CONDITION_COLUMN = {
    "discountFix": "choice",
}

# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

SECONDS_TO_MILLISECONDS = 1000

# =============================================================================
# DATA VALUES
# =============================================================================
NO_RESPONSE = -1

# =============================================================================
# STOP TASK PARAMETERS
# =============================================================================
MAX_SSD = 1000
MIN_SSD = 0.0
MAX_GO_RT = 2000

# Stop trial types by task
STOP_TRIAL_TYPES = {
    "stopSignal": {"success": "stop_success", "failure": "stop_failure"},
    "motorSelectiveStop": {"success": "crit_stop_success", "failure": "crit_stop_failure"},
}

GO_TRIAL_TYPES = {
    "stopSignal": ["go"],
    "motorSelectiveStop": ["crit_go", "noncrit_nosignal", "noncrit_signal"],
}

# =============================================================================
# QC THRESHOLDS
# =============================================================================

# Outlier detection (z-score threshold)
OUTLIER_THRESHOLD_STD = 3.0

# Truncation: consecutive omissions to trigger check
CONSECUTIVE_OMISSION_THRESHOLD = 3

# Truncation: omission rate in remaining data to confirm truncation
TRUNCATION_OMISSION_RATE = 0.50

# =============================================================================
# EXCLUSION THRESHOLDS
# =============================================================================

# Stop success rate bounds
STOP_SUCCESS_MIN = 0.25
STOP_SUCCESS_MAX = 0.75
STOP_SIGNAL_GO_ACC = 0.55
STOP_SIGNAL_GO_RT = 850

# Motor selective stop: noncrit signal omission threshold
MOTOR_STOP_NONCRIT_OMISSION_MAX = 0.35

# Data quality thresholds
OMISSION_RATE_MAX = 0.25
TRUNCATION_RATE_MAX = 0.50

# Minimum valid tasks required (exclude all if fewer)
MIN_VALID_TASKS = 2

# Discount task: proportion thresholds for exclusion
DISCOUNT_PROPORTION_MIN = 0.0  # Exclusive - if exactly 0 or 1, exclude
DISCOUNT_PROPORTION_MAX = 1.0

# Manipulation task response values
MANIP_RESPONSE_MIN = 1
MANIP_RESPONSE_MAX = 5

# =============================================================================
# SUBJECTIVE EXCLUSIONS (manual review)
# =============================================================================
SUBJECTIVE_EXCLUSIONS = [
    {"subject_id": 1046, "task": "motorSelectiveStop", "reason": "poor_performance_subjective_rating"},
    {"subject_id": 1399, "task": "discountFix", "reason": "poor_performance_subjective_rating"},
    {"subject_id": 4592, "task": "discountFix", "reason": "poor_performance_subjective_rating"},
    {"subject_id": 5387, "task": "discountFix", "reason": "poor_performance_subjective_rating"},
    {"subject_id": 1211, "task": "stopSignal", "reason": "poor_performance_subjective_rating"},
    {"subject_id": 1211, "task": "motorSelectiveStop", "reason": "poor_performance_subjective_rating"},
]
