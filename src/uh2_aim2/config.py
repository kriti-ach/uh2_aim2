"""Configuration for UH2 AIM2 behavioral QC pipeline."""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_PATH = "/oak/stanford/groups/russpold/data/uh2/aim2"
BEHAVIOR_PATH = os.path.join(BASE_PATH, "behavioral_data")
BEHAVIOR_QC_PATH = os.path.join(BEHAVIOR_PATH, "behavioral_qc")
# Processed cleaned behavioral CSVs for behavioral QC (e.g. ``1021_discountFix_cleaned.csv``)
BEHAVIOR_DATA = os.path.join(BEHAVIOR_PATH, "processed")
# Final analysis sample layout: ``{subject}/task/{subject}_{task}.csv`` (not used by QC scripts)
BEHAVIOR_SAMPLE_DATA = os.path.join(BEHAVIOR_PATH, "aim2_final_sample")
BEHAVIOR_TIMING_QC_CSV = os.path.join(BEHAVIOR_QC_PATH, "behavior_timing_qc.csv")
BEHAVIOR_TIMING_QC_FLAGGED_CSV = os.path.join(BEHAVIOR_QC_PATH, "behavior_timing_qc_flagged.csv")
# BIDS root (e.g. ``*events.tsv`` under ``sub-*/func/``)
BIDS_PATH = os.path.join(BASE_PATH, "BIDS")

# Flywheel project ``russpold/uh2aim2`` (group_id / project_label).
FLYWHEEL_GROUP_ID = "russpold"
FLYWHEEL_PROJECT_LABEL = "uh2aim2"
# ``flywheel.Client()`` reads credentials from the runtime environment / session (same pattern as
# ``rdoc_fmri_quality_control``), e.g. ``FW_API_KEY`` on Sherlock or ``flywheel login``.

# Optional local mirror for ``collect_ssg_num_slices.py --local`` only (not used for API pulls).
FLYWHEEL_JSON_EXPORT_PATH = os.environ.get("UH2_FLYWHEEL_JSON_ROOT") or os.path.join(
    os.path.expanduser("~"),
    FLYWHEEL_GROUP_ID,
    FLYWHEEL_PROJECT_LABEL,
)

# Project-local output path (scratch clone) for previewing trimmed BIDS event files
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRIMMED_EVENT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "trimmed_event_file_outputs")
TRIMMED_EVENT_OUTPUT_BIDS_DIR = os.path.join(TRIMMED_EVENT_OUTPUT_ROOT, "bids_outputs")
# Single exclusions file for behavioral + fMRIPrep QC (not repo-local; same path on /oak/…)
FINAL_EXCLUSIONS_JSON_PATH = os.path.join(BEHAVIOR_QC_PATH, "exclusions.json")

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

# Scanner / trigger wait windows. Raw ``time_elapsed`` is in milliseconds; span is
# converted to seconds. Nominal duration for every task below is
# ``BEHAVIOR_TIMING_NOMINAL_WAIT_DURATION_S`` (pass if span in [10.88, 10.89) s).
#
# Manipulation (scanner_wait): max(time_elapsed) - min(time_elapsed) on matching rows.
#
# discountFix / stopSignal / motorSelectiveStop (fmri_trigger_wait): in file order,
# last row's time_elapsed minus the second row's time_elapsed, plus ``FMRI_TRIGGER_TR_MS``.
# If there is only one matching row, use ``block_duration`` (ms) as fallback.
BEHAVIOR_TIMING_NOMINAL_WAIT_DURATION_S = 10.88
MANIPULATION_SCANNER_WAIT_TRIAL_ID = "scanner_wait"
MANIPULATION_SCANNER_WAIT_DURATION_S = BEHAVIOR_TIMING_NOMINAL_WAIT_DURATION_S
FMRI_TRIGGER_WAIT_TRIAL_ID = "fmri_trigger_wait"
# When ``exp_stage`` is absent, QC keeps only rows after this ``trial_id`` (file order).
EXPERIMENTOR_WAIT_TRIAL_ID = "experimentor_wait"
FMRI_TRIGGER_WAIT_DURATION_S = BEHAVIOR_TIMING_NOMINAL_WAIT_DURATION_S
FMRI_TRIGGER_WAIT_TASKS = ("discountFix", "stopSignal", "motorSelectiveStop")
FMRI_TRIGGER_TR_MS = 680
BEHAVIOR_TIMING_FLAG_DELTA_THRESHOLD = 0.5

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
STOP_SIGNAL_GO_RT = 1050

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

# Targets for ``scripts/trim_event_files.py``: BIDS ``sub-*/func/*_events.tsv`` only
BIDS_EVENT_FILES_TO_TRIM = [
    {"subject_id": 5064, "task": "manipulationTask"},
    {"subject_id": 5387, "task": "manipulationTask"},
    {"subject_id": 1143, "task": "manipulationTask"},
]

# =============================================================================
# GLOBAL MEAN SIGNAL PLOTS (NIfTI-based)
# =============================================================================
# Input BIDS directory containing sub-*/ses-*/func/*_bold.nii.gz
GLOBAL_MEAN_SIGNAL_BIDS_PATH = BIDS_PATH

# Output paths
GLOBAL_MEAN_SIGNAL_OUTPUT_DIR = os.path.join(BASE_PATH, "analysis_outputs", "figures")
GLOBAL_MEAN_SIGNAL_OUTPUT_PNG_DIR = os.path.join(
    GLOBAL_MEAN_SIGNAL_OUTPUT_DIR,
    "global_mean_signal_subject_pngs",
)
GLOBAL_MEAN_SIGNAL_OUTPUT_PDF = os.path.join(
    GLOBAL_MEAN_SIGNAL_OUTPUT_DIR,
    "global_mean_signal_subject_report.pdf",
)

# Default subject for quick single-subject plotting; if None, first subject in BIDS is used
GLOBAL_MEAN_DEFAULT_SUBJECT = None

# Tasks to plot if present (plus rest if present)
GLOBAL_MEAN_TASK_ORDER = [
    "discountFix",
    "manipulationTask",
    "motorSelectiveStop",
    "rest",
    "stopSignal",
]

# Plot styling and thresholds
GLOBAL_MEAN_PANEL2_THRESHOLD = 20.0
GLOBAL_MEAN_SHADE_START_TR = 0
GLOBAL_MEAN_SHADE_END_TR = 8
GLOBAL_MEAN_MAX_TRS = 25

# Subjects flagged from global mean signal plots as having unusually high signal
GLOBAL_MEAN_HIGH_SUBJECTS = [
    479,
    615,
    667,
    772,
    931,
    1189,
    1438,
    1646,
    2130,
    3525,
    5497,
    7124,
]

# =============================================================================
# FMRIPREP QC (confounds-based)
# =============================================================================
FMRIPREP_DERIVATIVES_PATH = os.path.join(
    BIDS_PATH,
    "derivatives",
    "fmriprep_20.2.7_ignore_sbref",
    "fmriprep",
)
FMRIPREP_QC_OUTPUT_DIR = os.path.join(BASE_PATH, "analysis_outputs")
FMRIPREP_QC_OUTPUT_CSV = os.path.join(FMRIPREP_QC_OUTPUT_DIR, "fmriprep_metrics.csv")

# Thresholds
FMRIPREP_FD_TR_THRESHOLD_MM = 0.5
FMRIPREP_DVARS_TR_THRESHOLD = 1.5
FMRIPREP_HIGH_MOTION_TR_PERCENT_THRESHOLD = 20.0
FMRIPREP_FD_MEAN_INCLUDE_THRESHOLD_MM = 0.2