from fastapi import APIRouter
from app.schemas import SensitivityAnalysisResponse, SensitivityAnalysisRow
from app.core import db
from collections import defaultdict

router = APIRouter()

@router.get("/", response_model=SensitivityAnalysisResponse, tags=["Analytics"])
def get_sensitivity_analysis():
    """
    Perform sensitivity analysis by contradiction type.
    """
    all_results = db.get_all_results()
    # {(model, prompt_strategy, temperature): {"self": [true_positives, all_positives], "pair": [true_positives, all_positives]}}
    sensitivity_data = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    
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
        # Count by contradiction type
        typewise_true_positives = defaultdict(int)
        typewise_all_actual_positives = defaultdict(int)
        for item in details_to_check:
            if not isinstance(item, dict):
                continue
            conflict_type = str(item.get("conflict_type", "none")).lower()
            if conflict_type not in ["self", "pair"]:
                continue
            if item.get("true") is True:
                typewise_all_actual_positives[conflict_type] += 1
                if item.get("pred") is True:
                    typewise_true_positives[conflict_type] += 1
        for conflict_type in ["self", "pair"]:
            if typewise_all_actual_positives[conflict_type] > 0:
                sensitivity_data[(model, prompt_strategy, temperature)][conflict_type][0] += typewise_true_positives[conflict_type]
                sensitivity_data[(model, prompt_strategy, temperature)][conflict_type][1] += typewise_all_actual_positives[conflict_type]
    
    # Format results
    rows = []
    for (model, prompt_strategy, temperature), sensitivity_counts in sensitivity_data.items():
        self_tp, self_total = sensitivity_counts.get("self", [0, 0])
        pair_tp, pair_total = sensitivity_counts.get("pair", [0, 0])
        
        self_sensitivity = round(self_tp / self_total * 100, 2) if self_total > 0 else None
        pair_sensitivity = round(pair_tp / pair_total * 100, 2) if pair_total > 0 else None
        
        rows.append(SensitivityAnalysisRow(
            model=model,
            prompt_strategy=prompt_strategy,
            temperature=temperature,
            self_contradiction_sensitivity=self_sensitivity,
            pair_contradiction_sensitivity=pair_sensitivity
        ))
    
    return SensitivityAnalysisResponse(rows=rows)