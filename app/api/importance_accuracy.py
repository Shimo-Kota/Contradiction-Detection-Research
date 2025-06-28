from fastapi import APIRouter
from app.schemas import ImportanceAccuracyResponse, ImportanceAccuracyRow
from app.core import db
from collections import defaultdict

router = APIRouter()

@router.get("/", response_model=ImportanceAccuracyResponse, tags=["Analytics"])
def get_importance_accuracy():
    """
    Analyze contradiction detection accuracy based on statement importance.
    """
    all_results = db.get_all_results()
    
    # {(model, prompt_strategy, temperature): {"most": [correct, total], "least": [correct, total]}}
    importance_data = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    
    for record in all_results:
        model = record.get("model")
        prompt_strategy = record.get("prompt_strategy")
        temperature = float(record.get("temperature", 0))
        # Only refer to detection_details (standard structure via evaluate)
        details_to_check = []
        details = record.get("details")
        if isinstance(details, dict):
            details_to_check = details.get("detection_details", [])
        # Skip if no detection_details
        if not details_to_check:
            continue
        for item in details_to_check:
            if not isinstance(item, dict):
                continue
            # Check importance metadata
            importance = item.get("statement_importance")
            
            # Skip if importance is not defined
            if not importance or importance not in ["most", "least"]:
                continue
                
            # Check if prediction is correct
            true_value = item.get("true")
            pred_value = item.get("pred")
            
            if true_value is not None and pred_value is not None:
                importance_data[(model, prompt_strategy, temperature)][importance][1] += 1
                if true_value == pred_value:
                    importance_data[(model, prompt_strategy, temperature)][importance][0] += 1
    
    # Format results
    rows = []
    for (model, prompt_strategy, temperature), importance_counts in importance_data.items():
        most_correct, most_total = importance_counts.get("most", [0, 0])
        least_correct, least_total = importance_counts.get("least", [0, 0])
        
        most_accuracy = round(most_correct / most_total * 100, 2) if most_total > 0 else None
        least_accuracy = round(least_correct / least_total * 100, 2) if least_total > 0 else None
        
        rows.append(ImportanceAccuracyRow(
            model=model,
            prompt_strategy=prompt_strategy,
            temperature=temperature,
            most_important_accuracy=most_accuracy,
            least_important_accuracy=least_accuracy
        ))
    
    return ImportanceAccuracyResponse(rows=rows)