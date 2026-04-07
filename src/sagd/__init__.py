"""SaGD: Saliency-Guided Knowledge Distillation."""

from sagd.data import (
    EvalInstructionDataset,
    GSM8KDataset,
    InstructionDataset,
    SAMSumDataset,
    SquadDataset,
)
from sagd.evaluation import (
    compute_bertscore,
    compute_evidence_concentration,
    compute_exact_match_f1,
    compute_gsm8k_accuracy,
    compute_perplexity,
    compute_rouge,
    evaluate_all,
    evaluate_rouge,
    generate_responses,
)
from sagd.losses import (
    BDLLoss,
    JSDLoss,
    ReverseKLLoss,
    SFTLoss,
    SkewKLLoss,
    StandardKDLoss,
)
from sagd.models import load_student, load_teacher
from sagd.saliency import SaliencyAlignmentLoss, SaliencyComputer

__all__ = [
    "InstructionDataset",
    "SquadDataset",
    "SAMSumDataset",
    "GSM8KDataset",
    "EvalInstructionDataset",
    "StandardKDLoss",
    "ReverseKLLoss",
    "SFTLoss",
    "JSDLoss",
    "SkewKLLoss",
    "BDLLoss",
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
    "compute_gsm8k_accuracy",
]
