"""
Synthetic Dataset Generation API Router.

This module implements the synthetic contradiction dataset generation framework
described in "Contradiction Detection in RAG Systems" (Gokul et al., 2025).

The generation endpoint creates synthetic contradictions using Algorithm 1 from the paper,
which defines three types of contradictions:
- self: Contradictions within a single document
- pair: Contradictions between two documents
- conditional: Contradictions requiring three documents with mutually exclusive conditions

The implementation follows the paper's parameters:
- α (alpha): Salience probability (default 0.5) - controls whether to choose
  most important or least important statements
- λ (lambda): Context length (default 100) - target word length for generated contexts
- Dataset distribution: The paper used a distribution of approximately 
  37.5% none / 26.3% self / 19.1% pair / 17.1% conditional

The dataset is built using HotpotQA as the source corpus, as specified in the paper.
"""
from fastapi import APIRouter, Query
from pathlib import Path
from ..generation.dataset_builder import SyntheticDatasetBuilder

router = APIRouter()

@router.post("/")
async def generate(
    n: int = Query(5, description="Number of examples to generate"),
    alpha: float = Query(0.5, description="Salience probability (α)"),
    context_length: int = Query(100, description="Context length (λ)"),
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
    )
):
    """
    Generate synthetic contradiction dataset from HotpotQA source.
    
    This endpoint implements the synthetic dataset generation framework described
    in Algorithm 1 of the paper. It creates examples of the three contradiction types:
    self-contradiction, pair contradiction, and conditional contradiction.
    
    The implementation uses LLM prompts as described in Appendix A.1 of the paper:
    - ChooseStatement: Selects statements based on importance
    - ContradictStatement: Generates contradictions of selected statements
    - ContextGenerate: Creates surrounding context of specified length
    - GenerateConditionalContradiction: Creates three-document conditional contradictions
    
    The alpha parameter controls whether to choose most important (higher alpha)
    or least important (lower alpha) statements. The paper found that contradictions
    in important statements are easier to detect (RQ2).
    
    The generated dataset is saved as a JSONL file with timestamp in the data directory.
    
    Args:
        n: Number of examples to generate
        alpha: Salience probability (default 0.5)
        context_length: Target word count for contexts (default 100)
        model: LLM to use for generation
        provider: Model provider (google, anthropic)
        
    Returns:
        Dictionary with path to the generated dataset file
    """
    builder = SyntheticDatasetBuilder(alpha=alpha, λ=context_length, model=model, provider=provider)
    source_path = Path("data/hotpotqa_source.json")
    out_dir = Path("data")
    out_path = await builder.build_dataset(source_path, out_dir, n=n)
    return {"output_path": str(out_path)}

