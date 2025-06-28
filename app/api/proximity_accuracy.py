from fastapi import APIRouter
from app.schemas import ProximityAccuracyResponse, ProximityAccuracyRow
from app.core import db
from collections import defaultdict

router = APIRouter()

@router.get("/", response_model=ProximityAccuracyResponse, tags=["Analytics"])
def get_proximity_accuracy():
    """
    Analyze contradiction detection accuracy based on document proximity.
    """
    all_results = db.get_all_results()
    
    # {(model, prompt_strategy, temperature): {"near": [correct, total], "far": [correct, total]}}
    proximity_data = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    
    for record in all_results:
        model = record.get("model")
        prompt_strategy = record.get("prompt_strategy")
        temperature = float(record.get("temperature", 0))
        # Only refer to detection_details (standard structure via evaluate)
        details_to_check = []
        details = record.get("details")
        if isinstance(details, dict):
            details_to_check = details.get("detection_details", [])
        if not details_to_check:
            continue
        for item in details_to_check:
            if not isinstance(item, dict):
                continue
            proximity = item.get("pair_proximity")
            if not proximity or proximity not in ["near", "far"]:
                continue
            true_value = item.get("true")
            pred_value = item.get("pred")
            if true_value is not None and pred_value is not None:
                proximity_data[(model, prompt_strategy, temperature)][proximity][1] += 1
                if true_value == pred_value:
                    proximity_data[(model, prompt_strategy, temperature)][proximity][0] += 1
    
    # Format results
    rows = []
    for (model, prompt_strategy, temperature), proximity_counts in proximity_data.items():
        near_correct, near_total = proximity_counts.get("near", [0, 0])
        far_correct, far_total = proximity_counts.get("far", [0, 0])
        
        near_accuracy = round(near_correct / near_total * 100, 2) if near_total > 0 else None
        far_accuracy = round(far_correct / far_total * 100, 2) if far_total > 0 else None
        
        rows.append(ProximityAccuracyRow(
            model=model,
            prompt_strategy=prompt_strategy,
            temperature=temperature,
            near_accuracy=near_accuracy,
            far_accuracy=far_accuracy
        ))
    
    return ProximityAccuracyResponse(rows=rows)