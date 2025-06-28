"""
Model Evaluation API Router.

This module implements the evaluation framework for the three context validator
tasks described in "Contradiction Detection in RAG Systems" (Gokul et al., 2025).

The evaluation endpoint assesses model performance on:
1. Conflict Detection: Binary classification (contradiction present/absent)
   - Metrics: Accuracy, Precision, Recall, F1
2. Conflict Type Prediction: Classification among self/pair/conditional
   - Metrics: Accuracy, Macro-F1
3. Conflicting Context Segmentation: Identifying documents involved in contradictions
   - Metrics: Jaccard similarity, F1 score (multi-label)

The implementation allows evaluation of different models, providers, and prompt
strategies (basic, CoT, multi_agent) as outlined in Section 4 of the paper. Results are
stored in a database for historical comparison and trend analysis.
"""
from fastapi import APIRouter, Query
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from fastapi import APIRouter, HTTPException, Query

from app.core.db import save_evaluation_result, update_aggregate, get_all_results, get_all_aggregate, \
    get_conflict_detection_history, get_type_detection_history, get_segmentation_history
from ..evaluation.evaluator import Evaluator
from app.schemas import (
    Metrics, DatasetMetadata, MetadataAnalysisRequest, MetadataAnalysisResponse,
    DatasetDistributionResponse, AnalysisByFieldResponse, ModelPerformanceResponse,
    PromptStrategyType
)

router = APIRouter()

@router.post("/")
async def evaluate(
    path: str,
    model: str = Query(
        None,
        description="Model name",
        enum=[
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gpt-4o-2024-11-20",
            "gpt-4o-mini-2024-07-18"
        ]
    ),
    provider: str = Query(
        None,
        description="Provider",
        enum=["google", "anthropic", "openai"]
    ),
    prompt_strategy: PromptStrategyType = Query("basic", description="Prompt strategy"),
    temperature: float = Query(0, description="Sampling temperature (0, 0.5, 1.0)", enum=[0, 0.5, 1.0])
):
    """
    Evaluate model performance on contradiction detection tasks.
    
    This endpoint runs a comprehensive evaluation of the specified model on all
    three tasks from the paper, using a dataset at the provided path. It calculates
    all metrics mentioned in Section 3 of the paper and stores results in a database.
    
    The evaluation covers:
    1. Detection: Accuracy, Precision, Recall, F1
    2. Type Prediction: Accuracy, Macro-F1
    3. Segmentation (both guided and blind): Jaccard similarity, F1
    
    According to the paper's findings, models generally have higher precision than
    recall in contradiction detection, indicating they are more likely to miss
    contradictions than falsely identify them. Different model families also show
    varying responses to different prompting strategies.
    
    The prompt_strategy parameter controls the reasoning approach:
    - basic: Simple prompting
    - cot: Chain-of-Thought "Think step by step" reasoning
    - multi_agent: Multiple agents discussing the problem (experimental)
    
    Args:
        path: Path to dataset file for evaluation
        model: LLM to evaluate (optional)
        provider: Model provider (google, anthropic)
        prompt_strategy: Prompting strategy to use
        
    Returns:
        Dictionary with evaluation scores for all tasks
    """
    # Convert cot to boolean for backward compatibility
    use_cot = (prompt_strategy == "cot")
    ev = Evaluator(model=model, provider=provider, use_cot=use_cot, prompt_strategy=prompt_strategy, temperature=temperature)
    scores = await ev.evaluate(Path(path))  # Convert str to Path
    
    # Explicitly include sample_count in the response
    response = {
        "detection": scores["detection"],
        "type": scores["type"],
        "segmentation": scores["segmentation"],
        "segmentation_blind": scores["segmentation_blind"],
        "detection_details": scores.get("detection_details"),
        "type_details": scores.get("type_details"),
        "segmentation_details": scores.get("segmentation_details"),
        "segmentation_blind_details": scores.get("segmentation_blind_details"),
        "total_sample_count": scores.get("total_sample_count", 0)
    }
    
    # Save to DB
    save_evaluation_result(
        model=model,
        provider=provider,
        prompt_strategy=prompt_strategy,
        temperature=temperature,
        dataset_path=path,
        detection=scores["detection"],
        type_=scores["type"],
        segmentation=scores["segmentation"],
        details={
            "detection_details": scores.get("detection_details"),
            "type_details": scores.get("type_details"),
            "segmentation_details": scores.get("segmentation_details"),
            "segmentation_blind_details": scores.get("segmentation_blind_details"),
            "sample_count": scores.get("total_sample_count", 0)
        }
    )
    # Update aggregate scores as well
    update_aggregate(model, provider, prompt_strategy, scores.get("detection_details", []), temperature=temperature)
    return response

# The following endpoints are not displayed on OpenAPI (kept as internal implementation)
@router.post("/evaluate/{task}", response_model=Metrics, include_in_schema=False)
async def run_evaluation(
    task: str,
    dataset_path: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    prompt_strategy: PromptStrategyType = Query("basic", description="Prompt strategy")
):
    """
    Existing evaluation endpoint - Evaluate performance of a specific task
    
    The prompt_strategy parameter controls the reasoning approach:
    - basic: Simple prompting
    - cot: Chain-of-Thought "Think step by step" reasoning
    - multi_agent: Multiple agents discussing the problem (experimental)
    """
    # Check if dataset exists
    p = Path(dataset_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Convert cot to boolean for backward compatibility
    use_cot = (prompt_strategy == "cot")
    
    # Initialize evaluator
    evaluator = Evaluator(
        model=model,
        provider=provider,
        prompt_strategy=prompt_strategy
    )
    
    # Run evaluation
    results = await evaluator.evaluate(dataset_path)
    
    # Extract metrics for the requested task
    if task not in results:
        raise HTTPException(status_code=400, detail=f"Task '{task}' not supported")
    
    # Get task-specific metrics
    task_metrics = results[task]
    
    # If sample_count is not included, use the length of task-specific details
    if "sample_count" not in task_metrics:
        task_metrics["sample_count"] = len(results.get(f"{task}_details", []))
    
    # Save to database
    save_evaluation_result(
        task=task,
        model=model,
        provider=provider,
        prompt_strategy=prompt_strategy,
        metrics=task_metrics
    )
    
    return task_metrics

@router.get("/get_all_past_result", include_in_schema=False)
def get_all_past_result():
    """
    Retrieve all historical evaluation results from the database.
    
    This endpoint returns all past evaluation runs, allowing comparison
    between different models, providers, and prompt strategies over time.
    
    Returns:
        List of all evaluation results stored in the database
    """
    results = get_all_results()
    # Ensure sample_count is included
    for result in results:
        if "detection" in result and "sample_count" not in result["detection"]:
            details_count = len(result.get("details", {}).get("detection_details", []))
            result["detection"]["sample_count"] = details_count
        if "type" in result and "sample_count" not in result["type"]:
            details_count = len(result.get("details", {}).get("type_details", []))
            result["type"]["sample_count"] = details_count
        if "segmentation" in result and "sample_count" not in result["segmentation"]:
            details_count = len(result.get("details", {}).get("segmentation_details", []))
            result["segmentation"]["sample_count"] = details_count
            
    return results

@router.get("/get_all_aggregate", include_in_schema=False)
def get_all_aggregate_api():  # Function name changed
    """
    Retrieve aggregate scores across all evaluations.
    
    This endpoint returns aggregated performance metrics for each model,
    provider, and prompt strategy combination. This provides a higher-level
    view of overall model performance across multiple evaluation runs.
    
    Returns:
        Dictionary with aggregated scores by model, provider, and prompt strategy
    """
    return get_all_aggregate()  # This calls the get_all_aggregate function from DB

@router.get("/history/conflict_detection", include_in_schema=False)
def get_conflict_detection_history_api():
    """
    Retrieve historical data for conflict detection performance.
    
    This endpoint returns time series data showing how conflict detection
    performance has evolved over time. Useful for tracking improvements
    or regressions in detection capability.
    
    Returns:
        Time series data of conflict detection performance metrics
    """
    return get_conflict_detection_history()

@router.get("/history/type_detection", include_in_schema=False)
def get_type_detection_history_api():
    """
    Retrieve historical data for conflict type prediction performance.
    
    This endpoint returns time series data showing how type prediction
    performance has evolved over time. The paper found this task particularly
    challenging, with varying performance across contradiction types.
    
    Returns:
        Time series data of type prediction performance metrics
    """
    return get_type_detection_history()

@router.get("/history/segmentation", include_in_schema=False)
def get_segmentation_history_api():
    """
    Retrieve historical data for context segmentation performance.
    
    This endpoint returns time series data showing how segmentation
    performance has evolved over time, covering both guided and blind modes.
    The paper found significant performance differences between these two modes.
    
    Returns:
        Time series data of segmentation performance metrics
    """
    return get_segmentation_history()

@router.get("/dataset/summary", response_model=DatasetMetadata, include_in_schema=False)
async def get_dataset_summary(dataset_path: str):
    """
    Get summary statistics for the dataset.
    
    Returns statistics such as distribution of contradiction types, statement importance,
    document proximity, and length of conflicting text.
    Can be used to generate a table equivalent to Table 2 in the frontend.
    """
    # Check if dataset exists
    p = Path(dataset_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get dataset summary
    summary = Evaluator.summarize_dataset(dataset_path)
    return summary

@router.post("/analyze/metadata", response_model=MetadataAnalysisResponse, include_in_schema=False)
async def analyze_by_metadata(request: MetadataAnalysisRequest):
    """
    Analyze evaluation results by metadata field.
    
    Returns evaluation results grouped by a specific metadata field
    (statement_importance, conflicting_evidence_length, pair_proximity).
    Can be used to generate various graphs in the frontend.
    """
    # Check if dataset exists
    p = Path(request.dataset_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # First, evaluate results in the dataset
    evaluator = Evaluator()
    results = await evaluator.evaluate(request.dataset_path)
    
    # Analyze by metadata field
    if request.field:
        analysis = Evaluator.analyze_by_metadata(results, request.field, request.filter_values)
        return MetadataAnalysisResponse(
            analysis_type=f"by_{request.field}",
            results=analysis,
            sample_count=results.get("total_sample_count", 0)  # Include sample count
        )
    else:
        # Default to analysis by contradiction type
        analysis = Evaluator.analyze_by_contradiction_type(results)
        return MetadataAnalysisResponse(
            analysis_type="by_contradiction_type",
            results=analysis,
            sample_count=results.get("total_sample_count", 0)  # Include sample count
        )

@router.get("/analyze/contradiction-types", response_model=AnalysisByFieldResponse, include_in_schema=False)
async def analyze_contradiction_types(
    dataset_path: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    prompt_strategy: PromptStrategyType = Query("basic", description="Prompt strategy")
):
    """
    Analyze detection accuracy by contradiction type.
    
    Returns detection accuracy for each contradiction type (self, pair, conditional).
    Can be used to generate the "Accuracy by Contradiction Type" graph in the frontend.
    
    The prompt_strategy parameter controls the reasoning approach:
    - basic: Simple prompting
    - cot: Chain-of-Thought "Think step by step" reasoning
    - multi_agent: Multiple agents discussing the problem (experimental)
    """
    # Check if dataset exists
    p = Path(dataset_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Convert cot to boolean for backward compatibility
    use_cot = (prompt_strategy == "cot")
    
    # Run evaluation
    evaluator = Evaluator(model=model, provider=provider, use_cot=use_cot)
    results = await evaluator.evaluate(dataset_path)
    
    # Analyze by contradiction type
    analysis = Evaluator.analyze_by_contradiction_type(results)
    
    return AnalysisByFieldResponse(
        field="contradiction_type",
        values=analysis.get("detection", {}),
        sample_count=results.get("total_sample_count", 0)  # Include sample count
    )

@router.get("/analyze/statement-importance", response_model=AnalysisByFieldResponse, include_in_schema=False)
async def analyze_statement_importance(
    dataset_path: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    prompt_strategy: PromptStrategyType = Query("basic", description="Prompt strategy")
):
    """
    Analyze detection accuracy by statement importance.
    
    Returns detection accuracy for each statement importance level (most, least).
    Can be used to generate the "Impact of Statement Importance on Detection" graph in the frontend.
    
    The prompt_strategy parameter controls the reasoning approach:
    - basic: Simple prompting
    - cot: Chain-of-Thought "Think step by step" reasoning
    - multi_agent: Multiple agents discussing the problem (experimental)
    """
    # Check if dataset exists
    p = Path(dataset_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Convert cot to boolean for backward compatibility
    use_cot = (prompt_strategy == "cot")
    
    # Run evaluation
    evaluator = Evaluator(model=model, provider=provider, use_cot=use_cot)
    results = await evaluator.evaluate(dataset_path)
    
    # Analyze by statement importance
    analysis = Evaluator.analyze_by_metadata(results, "statement_importance", ["most", "least"])
    
    return AnalysisByFieldResponse(
        field="statement_importance",
        values=analysis.get("detection", {}),
        sample_count=results.get("total_sample_count", 0)  # Include sample count
    )

@router.get("/analyze/document-proximity", response_model=AnalysisByFieldResponse, include_in_schema=False)
async def analyze_document_proximity(
    dataset_path: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    prompt_strategy: PromptStrategyType = Query("basic", description="Prompt strategy")
):
    """
    Analyze detection accuracy by document proximity.
    
    Returns detection accuracy for each document proximity (near, far).
    Can be used to generate the "Impact of Document Proximity on Detection" graph in the frontend.
    
    The prompt_strategy parameter controls the reasoning approach:
    - basic: Simple prompting
    - cot: Chain-of-Thought "Think step by step" reasoning
    - multi_agent: Multiple agents discussing the problem (experimental)
    """
    # Check if dataset exists
    p = Path(dataset_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Convert cot to boolean for backward compatibility
    use_cot = (prompt_strategy == "cot")
    
    # Run evaluation
    evaluator = Evaluator(model=model, provider=provider, use_cot=use_cot)
    results = await evaluator.evaluate(dataset_path)
    
    # Analyze by document proximity
    analysis = Evaluator.analyze_by_metadata(results, "pair_proximity", ["near", "far"])
    
    return AnalysisByFieldResponse(
        field="pair_proximity",
        values=analysis.get("detection", {}),
        sample_count=results.get("total_sample_count", 0)  # Include sample count
    )

@router.get("/analyze/evidence-length", response_model=AnalysisByFieldResponse, include_in_schema=False)
async def analyze_evidence_length(
    dataset_path: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    prompt_strategy: PromptStrategyType = Query("basic", description="Prompt strategy"),
    bucket_size: int = 50
):
    """
    Analyze detection accuracy by length of conflicting text.
    
    Returns detection accuracy for text length buckets (e.g., every 50 words).
    Can be used to generate the "Impact of Conflicting Evidence Length on Detection" graph in the frontend.
    
    The prompt_strategy parameter controls the reasoning approach:
    - basic: Simple prompting
    - cot: Chain-of-Thought "Think step by step" reasoning
    - multi_agent: Multiple agents discussing the problem (experimental)
    """
    # Check if dataset exists
    p = Path(dataset_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Convert cot to boolean for backward compatibility
    use_cot = (prompt_strategy == "cot")
    
    # Run evaluation
    evaluator = Evaluator(model=model, provider=provider, use_cot=use_cot)
    results = await evaluator.evaluate(dataset_path)
    
    # Analyze by bucketing text length
    analysis = Evaluator.analyze_evidence_length(results, bucket_size)
    
    return AnalysisByFieldResponse(
        field="evidence_length",
        values=analysis.get("detection", {}),
        sample_count=results.get("total_sample_count", 0)  # Include sample count
    )

@router.get("/model-performance", response_model=ModelPerformanceResponse, include_in_schema=False)
async def get_model_performance(
    dataset_path: str,
    model: str,
    provider: str,
    prompt_strategy: PromptStrategyType = Query("basic", description="Prompt strategy"),
    with_metadata_analysis: bool = False
):
    """
    Get detailed model performance.
    
    Returns complete performance metrics for a specific model configuration,
    optionally including metadata analysis.
    Can be used to generate a model performance table equivalent to "Table 3" in the frontend.
    
    The prompt_strategy parameter controls the reasoning approach:
    - basic: Simple prompting
    - cot: Chain-of-Thought "Think step by step" reasoning
    - multi_agent: Multiple agents discussing the problem (experimental)
    """
    # Check if dataset exists
    p = Path(dataset_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Convert cot to boolean for backward compatibility
    use_cot = (prompt_strategy == "cot")
    
    # Run evaluation
    evaluator = Evaluator(model=model, provider=provider, use_cot=use_cot)
    results = await evaluator.evaluate(dataset_path)
    
    # Construct basic metrics
    metrics = {
        "detection": results["detection"],
        "type": results["type"],
        "segmentation": results["segmentation"],
        "segmentation_blind": results["segmentation_blind"]
    }
    
    # Include metadata analysis if requested
    metadata_analysis = None
    if with_metadata_analysis:
        metadata_analysis = {
            "contradiction_type": Evaluator.analyze_by_contradiction_type(results),
            "statement_importance": Evaluator.analyze_by_metadata(results, "statement_importance"),
            "pair_proximity": Evaluator.analyze_by_metadata(results, "pair_proximity"),
            "evidence_length": Evaluator.analyze_evidence_length(results)
        }
    
    return ModelPerformanceResponse(
        model_name=model,
        provider=provider,
        prompt_strategy=prompt_strategy,
        metrics=metrics,
        metadata_analysis=metadata_analysis,
        sample_count=results.get("total_sample_count", 0)  # Include sample count
    )

@router.post("/multiple_files")
async def evaluate_multiple_files(
    paths: List[str] = Query(..., description="List of dataset file paths for evaluation"),
    model: str = Query(
        None,
        description="Model name",
        enum=[
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022", 
            "claude-3-5-haiku-20241022",
            "gemini-2.5-flash",
            "gemini-2.5-pro"
        ]
    ),
    provider: str = Query(
        None,
        description="Provider",
        enum=["google", "anthropic", "openai"]
    ),
    temperature: float = Query(0, description="Sampling temperature (0, 0.5, 1.0)", enum=[0, 0.5, 1.0])
):
    """
    Evaluate model performance on multiple datasets and prompt strategies sequentially.
    
    This endpoint wraps the existing /evaluate endpoint to process multiple datasets
    with all three prompt strategies (basic, cot, multi_agent) in sequence (not parallel).
    It calls /evaluate for each combination of file path and prompt strategy.
    
    Args:
        paths: List of dataset file paths for evaluation
        model: LLM to evaluate (optional)
        provider: Model provider (google, anthropic)
        temperature: Sampling temperature
        
    Returns:
        Dictionary containing results for all evaluated files and prompt strategies
    """
    results = {}
    prompt_strategies: List[PromptStrategyType] = ["basic", "cot", "multi_agent"]
    
    total_combinations = len(paths) * len(prompt_strategies)
    processed_count = 0
    
    for path in paths:
        results[path] = {}
        
        # Validate file exists once per path
        file_path = Path(path)
        if not file_path.exists():
            for strategy in prompt_strategies:
                results[path][strategy] = {
                    "success": False,
                    "error": f"File not found: {path}",
                    "processed_order": processed_count + 1
                }
                processed_count += 1
            continue
        
        # Execute each prompt strategy sequentially for this path
        for strategy in prompt_strategies:
            try:
                processed_count += 1
                
                # Call the existing evaluate function
                result = await evaluate(
                    path=path,
                    model=model,
                    provider=provider,
                    prompt_strategy=strategy,
                    temperature=temperature
                )
                
                results[path][strategy] = {
                    "success": True,
                    "result": result,
                    "processed_order": processed_count
                }
                
            except Exception as e:
                results[path][strategy] = {
                    "success": False,
                    "error": str(e),
                    "processed_order": processed_count
                }
    
    # Calculate summary statistics
    total_evaluations = 0
    successful_evaluations = 0
    
    for path_results in results.values():
        for strategy_result in path_results.values():
            total_evaluations += 1
            if strategy_result.get("success", False):
                successful_evaluations += 1
    
    summary = {
        "total_files": len(paths),
        "total_prompt_strategies": len(prompt_strategies),
        "total_evaluations": total_evaluations,
        "successful_evaluations": successful_evaluations,
        "failed_evaluations": total_evaluations - successful_evaluations,
        "success_rate": successful_evaluations / total_evaluations if total_evaluations > 0 else 0,
        "prompt_strategies_used": prompt_strategies
    }
    
    return {
        "summary": summary,
        "results": results
    }