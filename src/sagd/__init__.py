"""SaGD: Saliency-Guided Knowledge Distillation."""

from sagd.data import InstructionDataset, SquadDataset
from sagd.evaluation import (
    compute_bertscore,
    compute_evidence_concentration,
    compute_exact_match_f1,
    compute_perplexity,
    compute_rouge,
    evaluate_all,
    evaluate_rouge,
    generate_responses,
)
from sagd.losses import ReverseKLLoss, StandardKDLoss
from sagd.models import load_student, load_teacher
from sagd.saliency import SaliencyAlignmentLoss, SaliencyComputer

__all__ = [
    "InstructionDataset",
    "SquadDataset",
    "StandardKDLoss",
    "ReverseKLLoss",
    "SaliencyComputer",
    "SaliencyAlignmentLoss",
    "load_teacher",
    "load_student",
    "evaluate_rouge",
    "evaluate_all",
    "generate_responses",
    "compute_rouge",
    "compute_bertscore",
    "compute_perplexity",
    "compute_exact_match_f1",
    "compute_evidence_concentration",
]
