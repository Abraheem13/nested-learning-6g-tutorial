"""Statistics, aggregation and figure generation."""
from .aggregate import (WINDOWS, aggregate_table, comparison_table,
                        learning_curves, method_label, per_seed_summary)
from .statistics import (TestResult, bootstrap_ci, cohens_d, holm_bonferroni,
                         interquartile_mean, paired_permutation_test,
                         probability_of_improvement, summarise)

__all__ = [
    "WINDOWS", "per_seed_summary", "aggregate_table", "comparison_table",
    "learning_curves", "method_label", "TestResult", "summarise",
    "paired_permutation_test", "holm_bonferroni", "interquartile_mean",
    "bootstrap_ci", "probability_of_improvement", "cohens_d",
]
