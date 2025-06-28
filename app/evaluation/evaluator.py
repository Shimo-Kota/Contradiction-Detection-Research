"""
Model Evaluation Framework for Contradiction Detection Tasks.

The evaluator processes test datasets and calculates all metrics as follows:
1. Detection: Accuracy, Precision, Recall, F1
2. Type Prediction: Accuracy, Macro-F1 (across self/pair/conditional types)
3. Segmentation (both guided and blind): Jaccard similarity, F1 score (multi-label)
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from .metrics import classification_metrics, multilabel_metrics
from app.api.detection import detect_logic
from app.api.type_prediction import predict_type
from app.api.segmentation import segment
from app.schemas import (
    TypeRequest, SegmentRequest, 
    DatasetMetadata, PromptStrategyType
)

class Evaluator:
    """
    Evaluator for contradiction detection tasks.
    
    This class provides methods to evaluate LLM performance on contradiction detection,
    type prediction, and segmentation tasks. It uses the same API endpoints as the 
    web service, allowing for consistent evaluation both online and offline.
    
    Attributes:
        base_url: Base URL for API endpoints (for future remote evaluation support)
        model: LLM model to use for evaluation
        provider: Model provider (google, anthropic)
        prompt_strategy: Prompting strategy to use (basic, cot, multi_agent)
        use_cot: Legacy attribute for backwards compatibility
    """
    def __init__(
        self, 
        base_url: str = "http://localhost:8080", 
        model: str = None, 
        provider: str = None, 
        use_cot: bool = False,
        prompt_strategy: PromptStrategyType = "basic",
        temperature: float = 0
    ):
        """
        Initialize the evaluator with model and evaluation parameters.
        
        Args:
            base_url: Base URL for API endpoints (unused in current implementation)
            model: LLM model to use for evaluation
            provider: Model provider (google, anthropic)
            use_cot: Whether to enable Chain-of-Thought reasoning (legacy parameter)
            prompt_strategy: Prompting strategy to use (basic, cot, multi_agent)
            temperature: Sampling temperature for the evaluation run
        """
        self.base = base_url
        self.model = model
        self.provider = provider
        
        if use_cot and prompt_strategy == "basic":
            prompt_strategy = "cot"
        
        self.prompt_strategy = prompt_strategy 
        self.use_cot = (prompt_strategy == "cot")
        self.temperature = temperature
        self.cached_results = {}

    def _safe_float(self, val):
        """
        Handle NaN and infinity values in evaluation metrics.
        
        Args:
            val: Value to check
            
        Returns:
            The value itself, or 0.0 if NaN or infinity
        """
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return 0.0
        return val

    def _safe_dict(self, d):
        """
        Apply safe_float to all values in a dictionary.
        
        Args:
            d: Dictionary with potentially problematic float values
            
        Returns:
            Dictionary with all NaN and infinity values replaced with 0.0
        """
        return {k: self._safe_float(v) for k, v in d.items()}

    async def evaluate(self, dataset_path: Union[str, Path], cache_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate model performance on a dataset.
        
        This method loads a dataset of contradiction examples and evaluates the model's
        performance on all three tasks from the paper. It follows the sequential evaluation
        approach described in Section 4.3, where:
        
        1. Detection is performed on all examples
        2. Type prediction is performed only on examples with contradictions
        3. Segmentation is performed using the predicted type (guided mode)
        4. Segmentation is also performed without type information (blind mode)
        
        Args:
            dataset_path: Path to the dataset file (.jsonl format)
            cache_results: Whether to cache the evaluation results for later analysis
            
        Returns:
            Dictionary containing scores and detailed results for all tasks
        """
        dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
        
        cache_key = f"{dataset_path}_{self.model}_{self.provider}_{self.prompt_strategy}"
        if cache_key in self.cached_results and cache_results:
            return self.cached_results[cache_key]
        
        y_det_true, y_det_pred = [], []
        y_type_true, y_type_pred = [], []
        y_seg_true, y_seg_pred = [], []
        y_seg_blind_true, y_seg_blind_pred = [], []
        
        detection_details, type_details, segmentation_details = [], [], []
        segmentation_blind_details = []

        with dataset_path.open() as f:
            records = [json.loads(l) for l in f]

        for idx, rec in enumerate(records):
            docs = rec["docs"]
            docs_text = [d["text"] if isinstance(d, dict) else d for d in docs]
            metadata = {
                "conflict_type": rec.get("conflict_type", "none"),
                "statement_importance": rec.get("statement_importance"),
                # Include conflicting_evidence_length only for self/pair types
                "conflicting_evidence_length": rec.get("conflicting_evidence_length") if rec.get("conflict_type") in ["self", "pair"] else None,
                "pair_proximity": rec.get("pair_proximity")
            }
            try:
                is_conflict = await detect_logic(
                    docs_text,
                    model=self.model,
                    provider=self.provider,
                    prompt_strategy=self.prompt_strategy,
                    actual_conflict=rec["conflict"],
                    temperature=self.temperature
                )
                class DummyResp:
                    def __init__(self, conflict):
                        self.conflict = conflict
                det_resp = DummyResp(is_conflict)
            except Exception as e:
                print(f"[Detection] Skipped sample {idx} due to error: {e}")
                continue
            y_det_true.append(rec["conflict"])
            y_det_pred.append(det_resp.conflict)
            detection_details.append({
                "index": idx,
                "true": rec["conflict"],
                "pred": det_resp.conflict,
                "success": rec["conflict"] == det_resp.conflict,
                **metadata
            })

            if rec["conflict"]:
                try:
                    type_resp = await predict_type(
                        TypeRequest(documents=docs_text, conflict=True), 
                        model=self.model, 
                        provider=self.provider, 
                        prompt_strategy=self.prompt_strategy,
                        temperature=self.temperature
                    )
                except Exception as e:
                    print(f"[Type] Skipped sample {idx} due to error: {e}")
                    continue
                y_type_true.append(rec["conflict_type"])
                y_type_pred.append(type_resp.conflict_type)
                type_details.append({
                    "index": idx,
                    "true": rec["conflict_type"],
                    "pred": type_resp.conflict_type,
                    "success": rec["conflict_type"] == type_resp.conflict_type,
                    **metadata
                })
                try:
                    seg_resp = await segment(
                        SegmentRequest(documents=docs_text, guided=True, conflict_type=type_resp.conflict_type), 
                        model=self.model, 
                        provider=self.provider, 
                        prompt_strategy=self.prompt_strategy,
                        temperature=self.temperature
                    )
                except Exception as e:
                    print(f"[Segmentation] Skipped sample {idx} due to error: {e}")
                    continue
                y_seg_true.append(rec["doc_ids"])
                y_seg_pred.append(seg_resp.doc_ids)
                segmentation_details.append({
                    "index": idx,
                    "true": rec["doc_ids"],
                    "pred": seg_resp.doc_ids,
                    "success": rec["doc_ids"] == seg_resp.doc_ids,
                    **metadata
                })
                try:
                    seg_blind_resp = await segment(
                        SegmentRequest(documents=docs_text, guided=False, conflict_type=None), 
                        model=self.model, 
                        provider=self.provider, 
                        prompt_strategy=self.prompt_strategy,
                        temperature=self.temperature
                    )
                except Exception as e:
                    print(f"[SegmentationBlind] Skipped sample {idx} due to error: {e}")
                    continue
                y_seg_blind_true.append(rec["doc_ids"])
                y_seg_blind_pred.append(seg_blind_resp.doc_ids)
                segmentation_blind_details.append({
                    "index": idx,
                    "true": rec["doc_ids"],
                    "pred": seg_blind_resp.doc_ids,
                    "success": rec["doc_ids"] == seg_blind_resp.doc_ids,
                    **metadata
                })

        # Record total number of trials (n)
        total_samples = len(records)
        detection_samples = len(detection_details)
        type_samples = len(type_details)
        segmentation_samples = len(segmentation_details)
        segmentation_blind_samples = len(segmentation_blind_details)

        det_scores = classification_metrics(y_det_true, y_det_pred)
        det_scores["sample_count"] = detection_samples
        
        type_scores = classification_metrics(y_type_true, y_type_pred, average_method="macro")
        type_scores["sample_count"] = type_samples
        
        seg_scores = multilabel_metrics(y_seg_true, y_seg_pred)
        seg_scores["sample_count"] = segmentation_samples
        
        seg_blind_scores = multilabel_metrics(y_seg_blind_true, y_seg_blind_pred)
        seg_blind_scores["sample_count"] = segmentation_blind_samples
        
        results = {
            "detection": self._safe_dict(det_scores),
            "type": self._safe_dict(type_scores),
            "segmentation": self._safe_dict(seg_scores),
            "segmentation_blind": self._safe_dict(seg_blind_scores),
            "detection_details": detection_details,
            "type_details": type_details,
            "segmentation_details": segmentation_details,
            "segmentation_blind_details": segmentation_blind_details,
            "total_sample_count": total_samples,
            "prompt_strategy": self.prompt_strategy,
            "temperature": self.temperature
        }
        
        if cache_results:
            self.cached_results[cache_key] = results
        
        return results
    
    @staticmethod
    def analyze_by_metadata(results: Dict[str, Any], field: str, values: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Analyze results by a specific metadata field for reproducing paper figures.
        
        This method groups detection/type/segmentation results by specific metadata fields
        such as statement_importance, conflicting_evidence_length, or pair_proximity,
        allowing for the reproduction of the paper's figures showing impact of these factors.
        
        Args:
            results: Evaluation results from the evaluate method
            field: Metadata field to group by (e.g., "statement_importance")
            values: Optional list of values to filter by. If None, uses all unique values.
            
        Returns:
            Dictionary with metrics grouped by the specified metadata field
        """
        analysis = {}
        
        # Extract relevant details from results
        for task in ["detection", "type", "segmentation", "segmentation_blind"]:
            task_details = results.get(f"{task}_details", [])
            if not task_details:
                continue
                
            if values is None:
                values = set()
                for detail in task_details:
                    if field in detail and detail[field] is not None:
                        values.add(detail[field])
                values = sorted(list(values))
            
            analysis[task] = {}
            for value in values:
                filtered_details = [d for d in task_details if d.get(field) == value]
                if not filtered_details:
                    continue
                    
                success_rate = sum(1 for d in filtered_details if d.get("success", False)) / len(filtered_details)
                
                analysis[task][value] = {
                    "count": len(filtered_details),
                    "accuracy": success_rate
                }
        
        analysis["total_sample_count"] = results.get("total_sample_count", 0)
        
        return analysis
    
    @staticmethod
    def analyze_by_contradiction_type(results: Dict[str, Any]) -> Dict[str, Dict]:
        """Analyze detection results by contradiction type."""
        return Evaluator.analyze_by_metadata(results, "conflict_type", ["self", "pair", "conditional"])

    @staticmethod
    def analyze_evidence_length(results: Dict[str, Any]) -> Dict[str, Dict]:
        """Analyze results by evidence length."""
        return Evaluator.analyze_by_metadata(results, "conflicting_evidence_length", [50, 100, 200])

    @staticmethod
    def summarize_dataset(dataset_path: Union[str, Path]) -> DatasetMetadata:
        """Generate statistics about a dataset."""
        dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
        
        type_counts = {"none": 0, "self": 0, "pair": 0, "conditional": 0}
        metadata_stats = {
            "statement_importance": {"most": 0, "least": 0},
            "pair_proximity": {"near": 0, "far": 0},
        }
        
        evidence_lengths = []
        
        with dataset_path.open() as f:
            records = [json.loads(l) for l in f]
        
        for rec in records:
            ctype = rec.get("conflict_type", "none")
            type_counts[ctype] += 1
            
            if ctype in ["self", "pair"]:
                importance = rec.get("statement_importance")
                if importance:
                    metadata_stats["statement_importance"][importance] = metadata_stats["statement_importance"].get(importance, 0) + 1
                    
                length = rec.get("conflicting_evidence_length")
                if length:
                    evidence_lengths.append(length)
            
            if ctype == "pair":
                proximity = rec.get("pair_proximity")
                if proximity:
                    metadata_stats["pair_proximity"][proximity] = metadata_stats["pair_proximity"].get(proximity, 0) + 1
        
        # Calculate percentages
        total = sum(type_counts.values())
        type_percentages = {k: (v / total) * 100 if total > 0 else 0 for k, v in type_counts.items()}
        
        return DatasetMetadata(
            total=total,
            type_counts=type_counts,
            type_percentages=type_percentages,
            metadata_stats=metadata_stats,
            evidence_length_distribution={}
        )