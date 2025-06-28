import asyncio
import re
from app.core.llm_clients import chat_completion
from app.schemas import TypeResponse, SegmentResponse
from app.prompts.multi_agent_prompts import (
    MULTI_AGENT_DETECT_PROMPT, EVIDENCE_COLLECTOR_DETECT_PROMPT,
    EVIDENCE_COLLECTOR_TYPE_PROMPT, EVIDENCE_COLLECTOR_GUIDED_SEGMENT_PROMPT,
    EVIDENCE_COLLECTOR_BLIND_SEGMENT_PROMPT,
)

async def discuss_with_evidence_collector(
    documents: list[str],
    model: str,
    provider: str,
    cot_instruction: str = "",
    temperature: float = 0.0,
) -> bool:
    """Return True if a contradiction exists, False otherwise (via <conflict> tag).

    Turn‑1  Evidence Collector
    Turn‑2  Reviewer – final decision
    """

    context = "\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))

    collector_prompt = EVIDENCE_COLLECTOR_DETECT_PROMPT.format(docs=context)
    messages = [
        {"role": "system", "content": collector_prompt},
        {"role": "user", "content": "Extract evidence relevant to contradiction detection following the required <evidence> format."},
    ]
    evidence = await chat_completion(messages, model=model, provider=provider, temperature=temperature)
    print(f"Evidence Collector output:\n{evidence}\n")

    reviewer_prompt = MULTI_AGENT_DETECT_PROMPT.format(docs=context, cot_instruction=cot_instruction)
    messages = [
        {"role": "assistant", "content": evidence},
        {"role": "system", "content": reviewer_prompt},
        {"role": "user", "content": "Using the evidence above, output your final conclusion."},
    ]
    final_answer = await chat_completion(messages, model=model, provider=provider, temperature=temperature)
    print(f"Reviewer output:\n{final_answer}\n")

    match = re.search(r"<conflict>(yes|no)</conflict>", final_answer.strip().lower())
    if match:
        return match.group(1) == "yes"
    raise ValueError(f"Could not parse <conflict> tag from response: {final_answer}")

async def discuss_with_evidence_collector_type(
    documents: list[str],
    model: str,
    provider: str,
    cot_instruction: str = "",
    temperature: float = 0.0,
) -> TypeResponse:
    """Return the contradiction type (via <type> tag).

    Turn‑1  Evidence Collector - extract evidence related to contradiction types
    Turn‑2  Reviewer - makes the final <type> decision based on those snippets
    """
    context = "\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))
    
    collector_prompt = EVIDENCE_COLLECTOR_TYPE_PROMPT.format(docs=context)
    messages = [
        {"role": "system", "content": collector_prompt},
        {"role": "user", "content": "Extract evidence relevant to determining the contradiction type following the required <evidence> format."},
    ]
    evidence = await chat_completion(messages, model=model, provider=provider, temperature=temperature)
    print(f"Type Evidence Collector output:\n{evidence}\n")
    
    from app.prompts.multi_agent_prompts import MULTI_AGENT_TYPE_REVIEW_PROMPT
    reviewer_prompt = MULTI_AGENT_TYPE_REVIEW_PROMPT.format(docs=context, cot_instruction=cot_instruction)
    messages = [
        {"role": "assistant", "content": evidence},
        {"role": "system", "content": reviewer_prompt},
        {"role": "user", "content": "Using the evidence above, output your final type classification with the required <type> tag format. Remember to use ONLY one of: <type>self</type>, <type>pair</type>, <type>conditional</type>, or <type>none</type>"},
    ]
    final_answer = await chat_completion(messages, model=model, provider=provider, temperature=temperature)
    print(f"Type Reviewer output:\n{final_answer}\n")
    
    # Extract type from XML tags
    match = re.search(r"<type>(.*?)</type>", final_answer, re.IGNORECASE | re.DOTALL)
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
            typ = None
        if typ in ["self", "pair", "conditional", "none"]:
            return TypeResponse(conflict_type=typ)
    
    # Fallback: extract keywords if no tags found
    reply_lower = final_answer.lower()
    if "self-contradiction" in reply_lower or "self contradiction" in reply_lower or '"self"' in reply_lower:
        return TypeResponse(conflict_type="self")
    elif "pair contradiction" in reply_lower or '"pair"' in reply_lower:
        return TypeResponse(conflict_type="pair")
    elif "conditional contradiction" in reply_lower or '"conditional"' in reply_lower:
        return TypeResponse(conflict_type="conditional")
    elif "no contradiction" in reply_lower or "none" in reply_lower:
        return TypeResponse(conflict_type="none")
    
    raise ValueError(f"Could not parse a final contradiction type from the response: {final_answer}")


async def discuss_with_evidence_collector_segment(
    documents: list[str],
    guided: bool,
    conflict_type: str,
    model: str,
    provider: str,
    cot_instruction: str = "",
    temperature: float = 0.0,
) -> SegmentResponse:
    """Return the document IDs involved in contradiction (via <documents> tag).

    Turn‑1  Evidence Collector - extract evidence related to which documents are involved
    Turn‑2  Reviewer - makes the final <documents> decision based on those snippets
    """
    context = "\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))
    
    if guided:
        collector_prompt = EVIDENCE_COLLECTOR_GUIDED_SEGMENT_PROMPT.format(docs=context, ctype=conflict_type)
    else:
        collector_prompt = EVIDENCE_COLLECTOR_BLIND_SEGMENT_PROMPT.format(docs=context)
        
    messages = [
        {"role": "system", "content": collector_prompt},
        {"role": "user", "content": "Extract evidence relevant to identifying which documents contain contradictions using the required <evidence> format."},
    ]
    evidence = await chat_completion(messages, model=model, provider=provider, temperature=temperature)
    print(f"Segment Evidence Collector output:\n{evidence}\n")
    
    from app.prompts.multi_agent_prompts import MULTI_AGENT_GUIDED_SEGMENT_REVIEW_PROMPT, MULTI_AGENT_BLIND_SEGMENT_REVIEW_PROMPT
    
    if guided:
        reviewer_prompt = MULTI_AGENT_GUIDED_SEGMENT_REVIEW_PROMPT.format(docs=context, ctype=conflict_type, cot_instruction=cot_instruction)
    else:
        reviewer_prompt = MULTI_AGENT_BLIND_SEGMENT_REVIEW_PROMPT.format(docs=context, cot_instruction=cot_instruction)
        
    messages = [
        {"role": "assistant", "content": evidence},
        {"role": "system", "content": reviewer_prompt},
        {"role": "user", "content": "Using the evidence above, output your final document IDs using ONLY the <documents> tag format. For example: <documents>1,3</documents> or <documents>2</documents> or <documents></documents> if no contradictions found."},
    ]
    final_answer = await chat_completion(messages, model=model, provider=provider, temperature=temperature)
    print(f"Segment Reviewer output:\n{final_answer}\n")
    
    # Extract document IDs from XML tags
    match = re.search(r"<documents>(.*?)</documents>", final_answer, re.IGNORECASE | re.DOTALL)
    ids = []
    if match:
        extracted_ids_str = match.group(1).strip()
        if extracted_ids_str:
            ids = [int(tok) for tok in re.split(r"[\s,]+", extracted_ids_str) if tok.isdigit()]
        return SegmentResponse(doc_ids=ids)
    
    # Fallback: extract numbers or handle no contradiction cases
    reply_lower = final_answer.lower()
    if "no contradiction" in reply_lower or "does not appear to be any contradictory information" in reply_lower:
        return SegmentResponse(doc_ids=[])
    
    found_numbers = re.findall(r"\b\d+\b", final_answer)
    ids = [int(num) for num in found_numbers]
    return SegmentResponse(doc_ids=ids)