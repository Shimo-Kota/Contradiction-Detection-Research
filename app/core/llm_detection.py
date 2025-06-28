"""
Core LLM Functions for Contradiction Detection Tasks.

1. Conflict Detection: Binary classification of contradiction presence
2. Conflict Type Prediction: Classification of contradiction types
3. Conflicting Context Segmentation: Identification of documents involved in contradictions

Each function handles a specific task, interpreting LLM responses and extracting
the required information from the model output using regex patterns.
"""
import re
from app.core.llm_clients import chat_completion
from app.prompts.tasks import TYPE_PROMPT, GUIDED_SEGMENT_PROMPT, BLIND_SEGMENT_PROMPT, DETECT_PROMPT
from app.prompts.cot_instructions import (
    COT_DETECT_PROMPT, COT_TYPE_PROMPT, COT_GUIDED_SEGMENT_PROMPT, 
    COT_BLIND_SEGMENT_PROMPT, EMPTY_COT_PROMPT
)
from app.schemas import TypeResponse, SegmentResponse
from app.api.multi_agent_discussion import discuss_with_evidence_collector

async def detect_conflict_llm(documents: list[str], model: str, provider: str, use_cot: bool, prompt_strategy: str, temperature: float = 0.0) -> bool:
    """
    Detect contradictions in a set of documents using LLMs.
    
    This function implements the Conflict Detection task from the paper,
    which evaluates whether a set of documents contains contradictory information.
    The model acts as a binary classifier, responding with Yes/No.
    
    Args:
        documents: List of document strings to check for contradictions
        model: Name of the LLM to use
        provider: Model provider (google, anthropic)
        use_cot: Whether to enable Chain-of-Thought reasoning (legacy, for backward compatibility)
        prompt_strategy: Prompting strategy to use ("basic", "cot", "multi_agent")
        temperature: Sampling temperature to use for the model (default: 0.0)
    
    Returns:
        Boolean indicating whether a contradiction is present
    
    Raises:
        ValueError: If the model output cannot be parsed as yes/no
    """
    print(f"prompt_strategy: {prompt_strategy}")
    if prompt_strategy == "multi_agent":
        from app.api.multi_agent_discussion import discuss_with_evidence_collector
        cot_instruction = COT_DETECT_PROMPT if (use_cot or prompt_strategy == "cot") else ""
        return await discuss_with_evidence_collector(documents, model=model, provider=provider, cot_instruction=cot_instruction, temperature=temperature)

    docs_str = "\\n".join(f"{i+1}. {doc}" for i, doc in enumerate(documents))
    cot_instruction = COT_DETECT_PROMPT if (use_cot or prompt_strategy == "cot") else EMPTY_COT_PROMPT
    prompt = DETECT_PROMPT.format(docs=docs_str, cot_instruction=cot_instruction)
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = await chat_completion(messages, model=model, provider=provider, temperature=temperature)
    answer = response.strip().lower()
    
    # Extract yes/no from response
    match = re.search(r"\\b(yes|no)\\b", answer)
    if match:
        return match.group(1) == "yes"

    if answer.startswith("yes"):
        return True
    if answer.startswith("no"):
        return False

    if "yes" in answer:
        return True
    if "no" in answer:
        return False
        
    raise ValueError(f"LLM output could not be parsed as yes/no: {response}")

async def detect_type_llm(documents: list[str], model: str, provider: str, use_cot: bool, prompt_strategy: str, temperature: float = 0.0) -> TypeResponse:
    """
    Classify the type of contradiction present in documents.
    
    This function implements the Conflict Type Prediction task from the paper,
    which classifies contradictions into three categories:
    - self: Contradictions within a single document
    - pair: Contradictions between two documents
    - conditional: Contradictions involving three documents with conditional exclusivity
    
    The function prioritizes responses in <type></type> tags as described in Appendix A.3
    but falls back to scanning the entire response if tags are not present.
    
    Args:
        documents: List of document strings containing a contradiction
        model: Name of the LLM to use
        provider: Model provider (google, anthropic)
        use_cot: Whether to enable Chain-of-Thought reasoning
        temperature: Sampling temperature to use for the model (default: 0.0)
        
    Returns:
        TypeResponse object containing the predicted contradiction type
        
    Raises:
        ValueError: If the model output cannot be parsed as a valid contradiction type
    """
    if prompt_strategy == "multi_agent":
        from app.api.multi_agent_discussion import discuss_with_evidence_collector_type
        cot_instruction = COT_TYPE_PROMPT if (use_cot or prompt_strategy == "cot") else ""
        return await discuss_with_evidence_collector_type(documents, model=model, provider=provider, cot_instruction=cot_instruction, temperature=temperature)

    cot_instruction = COT_TYPE_PROMPT if (use_cot or prompt_strategy == "cot") else EMPTY_COT_PROMPT
    prompt = TYPE_PROMPT.format(docs="\n\n".join(documents), cot_instruction=cot_instruction)
        
    reply = await chat_completion([{"role": "user", "content": prompt}], model=model, provider=provider, temperature=temperature)
    typ = None
    
    # Extract type from XML tags
    match = re.search(r"<type>(.*?)</type>", reply, re.IGNORECASE | re.DOTALL)
    if match:
        extracted_type = match.group(1).strip().lower()
        first_word = extracted_type.split()[0] if extracted_type else ""
        
        if first_word in ["self", "pair", "conditional"]:
            typ = first_word
        elif "self" in extracted_type:
            typ = "self"
        elif "pair" in extracted_type:
            typ = "pair"
        elif "conditional" in extracted_type:
            typ = "conditional"
        elif first_word in ["none", "no"] or "no contradiction" in extracted_type:
            typ = "none"
    else:
        # Fallback: scan response for type keywords
        reply_lower = reply.lower()
        if "self-contradiction" in reply_lower or "self contradiction" in reply_lower or '"self"' in reply_lower:
            typ = "self"
        elif "pair contradiction" in reply_lower or '"pair"' in reply_lower:
            typ = "pair"
        elif "conditional contradiction" in reply_lower or '"conditional"' in reply_lower:
            typ = "conditional"
        elif "no contradiction" in reply_lower or "none" in reply_lower:
            typ = "none"
            
    if typ not in ["self", "pair", "conditional", "none"]:
        raise ValueError(f"LLM output could not be parsed as a valid contradiction type: {reply}")
        
    return TypeResponse(conflict_type=typ)

async def segment_conflict_llm(documents: list[str], guided: bool, conflict_type: str, model: str, provider: str , use_cot: bool, prompt_strategy: str, temperature: float = 0.0) -> SegmentResponse:
    """Identify documents involved in a contradiction."""
    if prompt_strategy == "multi_agent":
        from app.api.multi_agent_discussion import discuss_with_evidence_collector_segment
        cot_instruction = COT_GUIDED_SEGMENT_PROMPT if ((use_cot or prompt_strategy == "cot") and guided) else (COT_BLIND_SEGMENT_PROMPT if (use_cot or prompt_strategy == "cot") else "")
        return await discuss_with_evidence_collector_segment(documents, guided=guided, conflict_type=conflict_type, model=model, provider=provider, cot_instruction=cot_instruction, temperature=temperature)

    # Return empty list if guided mode is requested but no type is provided
    if guided and not conflict_type:
        return SegmentResponse(doc_ids=[])
    
    # Select appropriate prompt and CoT instruction based on mode
    if guided:
        cot_instruction = COT_GUIDED_SEGMENT_PROMPT if (use_cot or prompt_strategy == "cot") else EMPTY_COT_PROMPT
        prompt = GUIDED_SEGMENT_PROMPT.format(ctype=conflict_type, docs=_join(documents), cot_instruction=cot_instruction)
    else:
        cot_instruction = COT_BLIND_SEGMENT_PROMPT if (use_cot or prompt_strategy == "cot") else EMPTY_COT_PROMPT
        prompt = BLIND_SEGMENT_PROMPT.format(docs=_join(documents), cot_instruction=cot_instruction)
        
    # Send request to LLM
    reply = await chat_completion([{"role": "user", "content": prompt}], model=model, provider=provider, temperature=temperature)
    ids = []
    
    # Primary extraction method: look for <documents></documents> tags
    match = re.search(r"<documents>(.*?)</documents>", reply, re.IGNORECASE | re.DOTALL)
    if match:
        extracted_ids_str = match.group(1).strip()
        # Parse document IDs, splitting by spaces or commas
        # Allow empty string in tags, which means no documents identified
        if extracted_ids_str: # Only try to parse if not empty
            ids = [int(tok) for tok in re.split(r"[\\s,]+", extracted_ids_str) if tok.isdigit()]
        # If extracted_ids_str is empty, ids will remain [], which is valid (no conflicting docs)
    else:
        # Fallback: check for "no contradiction" or extract numbers
        if "no contradiction" in reply.lower() or "does not appear to be any contradictory information" in reply.lower():
            return SegmentResponse(doc_ids=[])

        found_numbers = re.findall(r"\\b\\d+\\b", reply)
        ids = [int(num) for num in found_numbers]
        
    if not ids and not ("no contradiction" in reply.lower() or "does not appear to be any contradictory information" in reply.lower() or match):
        raise ValueError(f"LLM output could not be parsed as valid document IDs: {reply}")
        
    return SegmentResponse(doc_ids=ids)

def _join(docs):
    """Format documents with index numbers for prompt."""
    return "\n\n".join(f"({i+1}) {d}" for i, d in enumerate(docs))
