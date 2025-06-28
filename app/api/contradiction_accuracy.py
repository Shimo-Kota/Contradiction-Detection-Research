from fastapi import APIRouter
from app.schemas import ContradictionAccuracyResponse, ContradictionAccuracyRow
from app.core import db
from collections import defaultdict

router = APIRouter()

@router.get("/", response_model=ContradictionAccuracyResponse, tags=["Contradiction Accuracy"])
def get_contradiction_accuracy():
    """
    Analyze detection accuracy for each contradiction type.
    """
    all_results = db.get_all_results()
    # {(model, prompt_strategy, temperature): {"self": [correct, total], ...}}
    from collections import defaultdict
    aggregated_data = defaultdict(lambda: defaultdict(lambda: [0, 0]))
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
            conflict_type = str(item.get("conflict_type", "none")).lower()
            if conflict_type not in ["self", "pair", "conditional"]:
                continue
            true_val = item.get("true")
            pred_val = item.get("pred")
            if true_val is True:
                aggregated_data[(model, prompt_strategy, temperature)][conflict_type][1] += 1
                if pred_val is True:
                    aggregated_data[(model, prompt_strategy, temperature)][conflict_type][0] += 1
    output_rows = []
    for (model_key, prompt_strategy_key, temperature_key), type_accuracies in aggregated_data.items():
        self_accuracy = 0.0
        if "self" in type_accuracies and type_accuracies["self"][1] > 0:
            self_accuracy = round((type_accuracies["self"][0] / type_accuracies["self"][1]) * 100, 2)
        pair_accuracy = 0.0
        if "pair" in type_accuracies and type_accuracies["pair"][1] > 0:
            pair_accuracy = round((type_accuracies["pair"][0] / type_accuracies["pair"][1]) * 100, 2)
        conditional_accuracy = 0.0
        if "conditional" in type_accuracies and type_accuracies["conditional"][1] > 0:
            conditional_accuracy = round((type_accuracies["conditional"][0] / type_accuracies["conditional"][1]) * 100, 2)
        output_rows.append(ContradictionAccuracyRow(
            model=model_key,
            prompt_strategy=prompt_strategy_key,
            temperature=temperature_key,
            self_contradiction_accuracy=self_accuracy,
            pair_contradiction_accuracy=pair_accuracy,
            conditional_contradiction_accuracy=conditional_accuracy
        ))
    return ContradictionAccuracyResponse(rows=output_rows)