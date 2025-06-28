"""
Contradiction Segmentation API Router.

The segmentation endpoint identifies which documents in a set contribute to
a contradiction. It operates in two modes:
1. Guided mode: Provided with the type of contradiction to help identification
2. Blind mode: No type information provided, more challenging for models
"""
from fastapi import APIRouter, Query
from ..schemas import SegmentRequest, SegmentResponse, PromptStrategyType
from ..core.llm_detection import segment_conflict_llm
from ..core.db import save_evaluation_result

router = APIRouter()

@router.post("/", include_in_schema=False)
async def segment(
    req: SegmentRequest,
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
    Identify documents that contribute to a contradiction.
    
    This endpoint implements the Conflicting Context Segmentation task from the paper.
    It takes a set of documents and identifies which specific documents are involved
    in creating the contradiction. It can operate in two modes:
    
    1. Guided segmentation (req.guided=True): The model is provided with the contradiction
       type to help it identify the relevant documents. This is an easier task.
       
    2. Blind segmentation (req.guided=False): The model must identify contributing
       documents without knowledge of the contradiction type. This is more challenging.
       
    The paper found significant performance differences between these modes, indicating
    that correctly identifying the contradiction type first can substantially improve
    document identification accuracy.
    
    Args:
        req: Request object containing documents to analyze
        model: LLM to use for segmentation (optional)
        provider: Model provider (google, anthropic)
        prompt_strategy: Prompting strategy to use
        
    Returns:
        SegmentResponse with list of document IDs that contribute to the contradiction
    """
    # Convert prompt_strategy to boolean for backward compatibility
    use_cot = (prompt_strategy == "cot")
    result = await segment_conflict_llm(
        req.documents,
        guided=req.guided,
        conflict_type=req.conflict_type if req.guided else None,
        model=model,
        provider=provider,
        use_cot=use_cot,
        prompt_strategy=prompt_strategy,
        temperature=temperature
    )
    
    # Save result to database
    segmentation_mode = "guided" if req.guided else "blind"
    save_evaluation_result(
        model=model,
        provider=provider,
        prompt_strategy=prompt_strategy,
        task="segmentation",
        metrics={
            segmentation_mode: {
                "doc_ids": result.doc_ids,
                "actual_doc_ids": None  # Actual answer is not included in the request
            }
        },
        temperature=temperature
    )
    
    return result