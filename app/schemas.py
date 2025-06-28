from typing import List, Literal, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, conint

DocumentSet = List[str]

# Define prompt strategy type
PromptStrategyType = Literal["basic", "cot", "multi_agent"]

class DetectRequest(BaseModel):
    documents: DocumentSet = Field(..., description="List of N textual documents")
    actual_conflict: Optional[bool] = Field(default=None, description="Ground truth: whether contradiction is present (for evaluation/logging)")

class DetectResponse(BaseModel):
    conflict: bool

class TypeRequest(DetectRequest):
    conflict: bool = Field(default=True, description="Assume conflict present")

class TypeResponse(BaseModel):
    conflict_type: Literal["self", "pair", "conditional", "none"]

class SegmentRequest(DetectRequest):
    guided: bool = Field(default=False, description="True = guided segmentation")
    conflict_type: Optional[Literal["self", "pair", "conditional", "none"]] = None

class SegmentResponse(BaseModel):
    doc_ids: List[conint(gt=0)]  # type: ignore # type: ignore

class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    jaccard: Optional[float] = None
    sample_count: int = Field(..., description="(n)")

class DatasetMetadata(BaseModel):
    """Metadata analysis result for a dataset"""
    total: int
    type_counts: Dict[str, int]
    type_percentages: Dict[str, float]
    metadata_stats: Dict[str, Dict[str, int]]
    evidence_length_distribution: Dict[int, int]

class MetadataAnalysisRequest(BaseModel):
    """Metadata analysis request"""
    dataset_path: str = Field(..., description="Path to the dataset to analyze")
    field: Optional[str] = Field(None, description="Metadata field to analyze (statement_importance, conflicting_evidence_length, pair_proximity)")
    filter_values: Optional[List[str]] = Field(None, description="List of values to extract (e.g., ['most', 'least'])")

class MetadataAnalysisResponse(BaseModel):
    """Metadata analysis response"""
    analysis_type: str
    results: Dict[str, Any]
    sample_count: Optional[int] = Field(None, description="Total sample count")

class DatasetDistributionResponse(BaseModel):
    """Dataset distribution analysis result"""
    distribution: Dict[str, Any]
    type_percentages: Dict[str, float]
    sample_count: int = Field(..., description="Total sample count")

class AnalysisByFieldResponse(BaseModel):
    """Analysis result by specific metadata field"""
    field: str
    values: Dict[str, Dict[str, float]]
    sample_count: int = Field(..., description="Total sample count")

class ModelPerformanceResponse(BaseModel):
    """Model performance evaluation result"""
    model_name: str
    provider: str
    prompt_strategy: PromptStrategyType = "basic"
    metrics: Dict[str, Any]
    metadata_analysis: Optional[Dict[str, Any]] = None
    sample_count: int = Field(..., description="Total sample count")

class SummaryRow(BaseModel):
    type: str
    count: int
    pct: float

class SummaryResponse(BaseModel):
    rows: List[SummaryRow]

class ConflictDetectionMetrics(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    sample_count: Optional[int] = None

class TypeDetectionMetrics(BaseModel):
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None

class SegmentationSubMetrics(BaseModel):
    jaccard: Optional[float] = None
    f1: Optional[float] = None

class SegmentationMetrics(BaseModel):
    guided: Optional[SegmentationSubMetrics] = None
    blind: Optional[SegmentationSubMetrics] = None

class MetricsRow(BaseModel):
    model: str
    prompt_strategy: PromptStrategyType
    temperature: float = 0
    conflict_detection: Optional[ConflictDetectionMetrics] = None
    type_detection: Optional[TypeDetectionMetrics] = None
    segmentation: Optional[SegmentationMetrics] = None

class MetricsResponse(BaseModel):
    rows: List[MetricsRow]

class ContradictionAccuracyRow(BaseModel):
    model: str
    prompt_strategy: PromptStrategyType
    temperature: float = 0
    self_contradiction_accuracy: Optional[float] = None
    pair_contradiction_accuracy: Optional[float] = None
    conditional_contradiction_accuracy: Optional[float] = None

class ContradictionAccuracyResponse(BaseModel):
    rows: List[ContradictionAccuracyRow]

class ImportanceAccuracyRow(BaseModel):
    model: str
    prompt_strategy: PromptStrategyType
    temperature: float = 0
    most_important_accuracy: Optional[float] = None
    least_important_accuracy: Optional[float] = None

class ImportanceAccuracyResponse(BaseModel):
    rows: List[ImportanceAccuracyRow]

class ProximityAccuracyRow(BaseModel):
    model: str
    prompt_strategy: PromptStrategyType
    temperature: float = 0
    near_accuracy: Optional[float] = None
    far_accuracy: Optional[float] = None

class ProximityAccuracyResponse(BaseModel):
    rows: List[ProximityAccuracyRow]

class EvidenceLengthAccuracyRow(BaseModel):
    model: str
    prompt_strategy: PromptStrategyType
    temperature: float = 0
    evidence_50_accuracy: Optional[float] = None
    evidence_100_accuracy: Optional[float] = None
    evidence_200_accuracy: Optional[float] = None

class EvidenceLengthAccuracyResponse(BaseModel):
    rows: List[EvidenceLengthAccuracyRow]

class SensitivityAnalysisRow(BaseModel):
    model: str
    prompt_strategy: PromptStrategyType
    temperature: float = 0
    pair_contradiction_sensitivity: Optional[float] = None
    self_contradiction_sensitivity: Optional[float] = None

class SensitivityAnalysisResponse(BaseModel):
    rows: List[SensitivityAnalysisRow]