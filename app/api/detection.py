"""
Contradiction Detection API Router.


The detection endpoint evaluates whether a set of documents contains
contradictory information, returning a boolean result (conflict=True/False).
This corresponds to the "Conflict Detection" task in the paper, where
the model acts as a binary classifier.

Performance metrics for this task include Accuracy, Precision, Recall, and F1.
"""
from fastapi import APIRouter, Query
from ..schemas import DetectRequest, DetectResponse, PromptStrategyType
from ..core.llm_detection import detect_conflict_llm
from ..core.db import save_evaluation_result

router = APIRouter()

# --- Pure logic function for both API and Evaluator ---
async def detect_logic(documents, model, provider, prompt_strategy, actual_conflict=None, temperature: float = 0.0):
    """
    Pure logic for contradiction detection. Used by both API and Evaluator.
    """
    use_cot = (prompt_strategy == "cot")
    is_conflict = await detect_conflict_llm(
        documents,
        model=model,
        provider=provider,
        use_cot=use_cot,
        prompt_strategy=prompt_strategy,
        temperature=temperature
    )
    # DB logging only if actual_conflict is provided (for API)
    if actual_conflict is not None:
        save_evaluation_result(
            model=model,
            provider=provider,
            prompt_strategy=prompt_strategy,
            task="detection",
            metrics={
                "actual_conflict": actual_conflict,
                "predicted_conflict": is_conflict
            },
            temperature=temperature
        )
    return is_conflict

# --- FastAPI endpoint ---
@router.post("/", include_in_schema=False)
async def detect(
    req: DetectRequest,
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
    Detect contradictions in a set of documents.
    
    This endpoint implements the Conflict Detection task from the paper.
    It takes a set of documents as input and returns whether they contain
    contradictory information.
    
    The model acts as a binary classifier for documents, responding with
    Yes/No to indicate contradiction presence. The prompt_strategy parameter
    controls the reasoning approach:
    - basic: Simple prompting
    - cot: Chain-of-Thought "Think step by step" reasoning
    - multi_agent: Multiple agents discussing the problem (experimental)
    
    According to the paper's findings, different prompting strategies may have varying effects
    on different model families.
    
    Args:
        req: Request object containing documents to analyze
        model: LLM to use for detection (optional)
        provider: Model provider (google, anthropic)
        prompt_strategy: Prompting strategy to use
        
    Returns:
        DetectResponse with boolean 'conflict' field
    """
    is_conflict = await detect_logic(
        req.documents,
        model=model,
        provider=provider,
        prompt_strategy=prompt_strategy,
        actual_conflict=req.actual_conflict,
        temperature=temperature
    )
    
    return DetectResponse(conflict=is_conflict)