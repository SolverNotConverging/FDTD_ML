"""Physical-time observables, reference convergence and mesh evaluation."""

from .metrics import compare_observables, sample_observables
from .pipeline import EvaluationConfig, converge_reference, evaluate_dataset
from .references import generate_references

__all__ = [
    "EvaluationConfig",
    "evaluate_dataset",
    "converge_reference",
    "generate_references",
    "compare_observables",
    "sample_observables",
]
