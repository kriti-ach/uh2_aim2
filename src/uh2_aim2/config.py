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

# Discount task: proportion thresholds for exclusion
MIN_LARGER_LATER_PROPORTION = 0.0  # Exclusive - if exactly 0 or 1, exclude
MAX_LARGER_LATER_PROPORTION = 1.0

# Manipulation task response values
MANIP_RESPONSE_MIN = 1
MANIP_RESPONSE_MAX = 5

# =============================================================================
# HISTOGRAM PLOT SPECS
# Each entry: column name in the per-task QC DataFrame, or a tuple
# (col_a, col_b) meaning "col_a minus col_b" (computed difference).
# =============================================================================
HISTOGRAM_METRICS = {
    "stopSignal": [
        ("go_acc", "Go Accuracy"),
        ("go_rt", "Go RT (ms)"),
        ("omission_rate", "Omission Rate"),
        ("stop_success_rate", "Stop Success Rate"),
        (("stop_failure_rt", "go_rt"), "Stop Failure RT − Go RT (ms)"),
    ],
    "motorSelectiveStop": [
        ("crit_go_acc", "Crit Go Accuracy"),
        ("crit_go_rt", "Crit Go RT (ms)"),
        ("noncrit_signal_rt", "Noncrit Signal RT (ms)"),
        ("noncrit_nosignal_rt", "Noncrit Nosignal RT (ms)"),
        ("omission_rate", "Omission Rate"),
        ("crit_go_omission", "Crit Go Omission"),
        ("noncrit_nosignal_omission", "Noncrit Nosignal Omission"),
        ("noncrit_signal_omission", "Noncrit Signal Omission"),
        ("stop_success_rate", "Stop Success Rate"),
        (("crit_stop_failure_rt", "crit_go_rt"), "Crit Stop Failure RT − Crit Go RT (ms)"),
        (("crit_go_rt", "noncrit_nosignal_rt"), "Crit Go RT − Noncrit Nosignal RT (ms)"),
    ],
    "manipulationTask": [
        ("omission_rate", "Omission Rate"),
    ],
}

HISTOGRAM_BINS = 20

# =============================================================================
# SUBJECTIVE EXCLUSIONS (manual review)
# =============================================================================
# SUBJECTIVE_EXCLUSIONS = [
#     {"subject_id": 1046, "task": "motorSelectiveStop", "reason": "poor_performance_subjective_rating"},
#     {"subject_id": 1399, "task": "discountFix", "reason": "poor_performance_subjective_rating"},
#     {"subject_id": 4592, "task": "discountFix", "reason": "poor_performance_subjective_rating"},
#     {"subject_id": 5387, "task": "discountFix", "reason": "poor_performance_subjective_rating"},
#     {"subject_id": 1211, "task": "stopSignal", "reason": "poor_performance_subjective_rating"},
#     {"subject_id": 1211, "task": "motorSelectiveStop", "reason": "poor_performance_subjective_rating"},
# ]

SUBJECTIVE_EXCLUSIONS = []
