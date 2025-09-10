"""Data models for RefereeSim"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json


class ErrorCategory(str, Enum):
    """Categories of errors that can be seeded in papers"""
    STATS_MISUSE = "stats_misuse"
    UNIT_MISMATCH = "unit_mismatch"
    DATA_LEAKAGE = "data_leakage"
    ROUNDING_ERROR = "rounding_error"
    TABLE_NUMBER_ERROR = "table_number_error"
    CLAIM_CONTRADICTS_TABLE = "claim_contradicts_table"
    FAKE_CITATION = "fake_citation"
    SAMPLE_SIZE_MISREPORT = "sample_size_misreport"


class DifficultyLevel(str, Enum):
    """Difficulty levels for error detection"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class StudyType(str, Enum):
    """Types of studies that can be generated"""
    AB_TEST = "ab_test"
    TWO_GROUP_COMPARISON = "two_group_comparison"
    ML_CLASSIFICATION = "ml_classification"
    LINEAR_REGRESSION = "linear_regression"
    CLINICAL_OUTCOME = "clinical_outcome"


class ErrorSeed(BaseModel):
    """Represents an error injected into a paper"""
    category: ErrorCategory
    difficulty: DifficultyLevel
    location: str  # section/sentence or table cell
    original_text: str
    modified_text: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class PaperMetadata(BaseModel):
    """Metadata for a generated paper"""
    paper_id: str
    study_type: StudyType
    title: str
    ground_truth_data: Dict[str, Any]
    errors: List[ErrorSeed]
    is_control: bool = False  # True for papers with no errors


class ReviewFinding(BaseModel):
    """A finding reported by an AI reviewer"""
    category: str
    location: str
    quoted_text: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewResult(BaseModel):
    """Results from an AI reviewer"""
    paper_id: str
    model_name: str
    prompt_style: str
    findings: List[ReviewFinding]
    overall_assessment: str
    timestamp: str


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for a reviewer"""
    model_name: str
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    coverage_rate: float
    over_flag_rate: float


class RunManifest(BaseModel):
    """Manifest for a complete experimental run"""
    run_id: str
    timestamp: str
    config: Dict[str, Any]
    seed: int
    papers_generated: int
    models_tested: List[str]
    metrics_summary: Dict[str, EvaluationMetrics]


def save_json(obj, filepath: str):
    """Save an object to JSON file"""
    with open(filepath, 'w') as f:
        if isinstance(obj, BaseModel):
            json.dump(obj.model_dump(), f, indent=2)
        else:
            json.dump(obj, f, indent=2)


def load_json(filepath: str, model_class=None):
    """Load an object from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if model_class:
        return model_class(**data)
    return data