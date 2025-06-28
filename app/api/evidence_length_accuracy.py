from fastapi import APIRouter
from app.schemas import EvidenceLengthAccuracyResponse, EvidenceLengthAccuracyRow
from app.core import db
from collections import defaultdict

router = APIRouter()

@router.get("/", response_model=EvidenceLengthAccuracyResponse, tags=["Analytics"])
def get_evidence_length_accuracy():
    """
    Analyze accuracy based on the length of conflicting evidence (50, 100, 200 words).
    """
    all_results = db.get_all_results()
    
    # {(model, prompt_strategy, temperature): {50: [correct, total], 100: [correct, total], 200: [correct, total]}}
    length_data = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    
    for record in all_results:
        model = record.get("model")
        prompt_strategy = record.get("prompt_strategy")
        temperature = float(record.get("temperature", 0))
        details_to_check = []
        details = record.get("details")
        if isinstance(details, dict):
            details_to_check = details.get("detection_details", [])
        if not details_to_check:
            continue
        for item in details_to_check:
            if not isinstance(item, dict):
                continue
            evidence_length = item.get("conflicting_evidence_length")
            if evidence_length not in [50, 100, 200]:
                continue
            true_value = item.get("true")
            pred_value = item.get("pred")
            if true_value is not None and pred_value is not None:
                length_data[(model, prompt_strategy, temperature)][evidence_length][1] += 1
                if true_value == pred_value:
                    length_data[(model, prompt_strategy, temperature)][evidence_length][0] += 1
    rows = []
    for (model, prompt_strategy, temperature), length_counts in length_data.items():
        acc_50_correct, acc_50_total = length_counts.get(50, [0, 0])
        acc_100_correct, acc_100_total = length_counts.get(100, [0, 0])
        acc_200_correct, acc_200_total = length_counts.get(200, [0, 0])
        acc_50 = round(acc_50_correct / acc_50_total * 100, 2) if acc_50_total > 0 else None
        acc_100 = round(acc_100_correct / acc_100_total * 100, 2) if acc_100_total > 0 else None
        acc_200 = round(acc_200_correct / acc_200_total * 100, 2) if acc_200_total > 0 else None
        rows.append(EvidenceLengthAccuracyRow(
            model=model,
            prompt_strategy=prompt_strategy,
            temperature=temperature,
            evidence_50_accuracy=acc_50,
            evidence_100_accuracy=acc_100,
            evidence_200_accuracy=acc_200
        ))
    return EvidenceLengthAccuracyResponse(rows=rows)