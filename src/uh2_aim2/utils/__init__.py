"""Utility functions for UH2 AIM2 behavioral analysis."""

from uh2_aim2.utils.behavior_qc_utils import (
    calc_discount_rate_glm,
    calc_omission_rate,
    calc_ssrt,
    calc_stop_success_rate,
    compute_acc_summary,
    compute_rt_summary,
    format_qc_results,
)
from uh2_aim2.utils.behavior_exclusion_utils import (
    check_discount_choice_pattern,
    check_manip_pre_rating,
    check_minimum_valid_tasks,
    check_missing_data,
    check_motor_stop_noncrit_omission,
    check_omission_rate as check_omission_exclusion,
    check_stop_success_rate as check_stop_success_exclusion,
    check_truncation_rate,
    get_subjective_exclusions,
    run_all_exclusion_checks,
    summarize_exclusions,
)

__all__ = [
    # QC calculations
    "calc_discount_rate_glm",
    "calc_omission_rate",
    "calc_ssrt",
    "calc_stop_success_rate",
    "compute_acc_summary",
    "compute_rt_summary",
    "format_qc_results",
    # Exclusion checks
    "check_discount_choice_pattern",
    "check_manip_pre_rating",
    "check_minimum_valid_tasks",
    "check_missing_data",
    "check_motor_stop_noncrit_omission",
    "check_omission_exclusion",
    "check_stop_success_exclusion",
    "check_truncation_rate",
    "get_subjective_exclusions",
    "run_all_exclusion_checks",
    "summarize_exclusions",
]
