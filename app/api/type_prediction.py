"""
Contradiction Type Prediction API Router.

The type prediction endpoint classifies the type of contradiction present in a set
of documents, identifying whether the contradiction is:
1. Self: Within a single document
2. Pair: Between two documents
3. Conditional: Involving three documents with conditional exclusivity
4. None: No contradiction present
"""
from fastapi import APIRouter, Query
from ..schemas import TypeRequest, TypeResponse, PromptStrategyType
from ..core.llm_detection import detect_type_llm
from ..core.db import save_evaluation_result

router = APIRouter()

@router.post("/", include_in_schema=False)
async def predict_type(
    req: TypeRequest,
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
    temperature: float = Query(0.0, description="Sampling temperature (0, 0.5, 1.0)")
):
    """
    Predict the type of contradiction in documents.
    
    This endpoint implements the Conflict Type Prediction task from the paper.
    It takes a set of documents and identifies the specific type of contradiction
    present, according to the taxonomy outlined in Section 2.2 of the paper.
    
    The model outputs one of four classes:
    - self: Contradictions within a single document
    - pair: Contradictions between two documents
    - conditional: Contradictions involving three documents with conditional exclusivity
    - none: No contradiction detected
    
    In the paper's experiments, type prediction proved more challenging than binary
    detection, with performance differences between models and prompting strategies
    more pronounced.
    
    Args:
        req: Request object containing documents to analyze
        model: LLM to use for prediction (optional)
        provider: Model provider (google, anthropic)
        prompt_strategy: Prompting strategy to use
        
    Returns:
        TypeResponse with the predicted contradiction type
    """
    # Convert prompt_strategy to boolean for backward compatibility
    use_cot = (prompt_strategy == "cot")
    result = await detect_type_llm(req.documents, model=model, provider=provider, use_cot=use_cot, prompt_strategy=prompt_strategy, temperature=temperature)
    
    # Save result to database
    save_evaluation_result(
        model=model,
        provider=provider,
        prompt_strategy=prompt_strategy,
        task="type",
        metrics={"actual_type": None, "predicted_type": result.conflict_type},
        temperature=temperature
    )
    
    return result