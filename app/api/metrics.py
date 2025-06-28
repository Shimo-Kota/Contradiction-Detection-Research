from fastapi import APIRouter
from app.schemas import (
    MetricsResponse, MetricsRow, ConflictDetectionMetrics, 
    TypeDetectionMetrics, SegmentationMetrics, SegmentationSubMetrics
)
from app.core import db
from collections import Counter

router = APIRouter()

@router.get("/", response_model=MetricsResponse, tags=["Metrics"])
def get_metrics():
    """Get performance metrics by model, prompt strategy, and temperature."""
    aggregate_results = db.get_all_aggregate()
    all_results = db.get_all_results()

    rows = []
    individual_results_map = {}
    for r in all_results:
        key = (r.get("model"), r.get("prompt_strategy"), float(r.get("temperature", 0)))
        if key not in individual_results_map:
            individual_results_map[key] = []
        individual_results_map[key].append(r)

    for agg_row in aggregate_results:
        model_name = agg_row.get("model")
        prompt_strategy = agg_row.get("prompt_strategy")
        temperature = float(agg_row.get("temperature", 0))
        
        conflict_detection_metrics = ConflictDetectionMetrics(
            accuracy=agg_row.get("accuracy"),
            precision=agg_row.get("precision"),
            recall=agg_row.get("recall"),
            f1=agg_row.get("f1"),
            sample_count=agg_row.get("total_samples", 0)  # Use total_samples as sample_count
        )

        type_detection_metrics_data = {"accuracy": 0.0, "macro_f1": 0.0}
        segmentation_metrics_data = {
            "guided": {"jaccard": 0.0, "f1": 0.0}, 
            "blind": {"jaccard": 0.0, "f1": 0.0}
        }

        relevant_results = individual_results_map.get((model_name, prompt_strategy, temperature), [])
        type_true_positives = Counter()
        type_false_positives = Counter()
        type_false_negatives = Counter()
        type_correct_predictions = 0
        type_total_predictions = 0
        type_actual_types = []
        type_predicted_types = []

        for r_indiv in relevant_results:
            # Safely get the value of the details field, even if None
            details_dict = r_indiv.get("details")
            
            # Get type_details
            details_for_type = []
            if isinstance(details_dict, dict):
                details_for_type = details_dict.get("type_details", [])

            type_data = r_indiv.get('type') # Fallback for single API calls

            if details_for_type: # Data from evaluate
                for item in details_for_type:
                    if isinstance(item, dict):
                        actual = item.get('true') 
                        predicted = item.get('pred')
                        if actual and predicted:
                            type_total_predictions += 1
                            type_actual_types.append(str(actual).lower())
                            type_predicted_types.append(str(predicted).lower())
                            if actual == predicted:
                                type_correct_predictions += 1
                                type_true_positives[str(actual).lower()] += 1
                            else:
                                type_false_positives[str(predicted).lower()] += 1
                                type_false_negatives[str(actual).lower()] += 1
            elif type_data and isinstance(type_data, dict): # Data from single API
                actual = type_data.get('actual_type') # single API uses actual_type/predicted_type
                predicted = type_data.get('predicted_type')
                
                if actual and predicted:
                    type_total_predictions += 1
                    type_actual_types.append(str(actual).lower())
                    type_predicted_types.append(str(predicted).lower())
                    
                    if actual == predicted:
                        type_correct_predictions += 1
                        type_true_positives[str(actual).lower()] += 1
                    else:
                        type_false_positives[str(predicted).lower()] += 1
                        type_false_negatives[str(actual).lower()] += 1
        
        if type_total_predictions > 0:
            type_detection_metrics_data["accuracy"] = round(type_correct_predictions / type_total_predictions, 3)
            
            f1_scores = []
            all_unique_types = set(type_actual_types + type_predicted_types) # Already lowercased
            
            for t in all_unique_types: # t is already lowercased
                tp = type_true_positives[t]
                fp = type_false_positives[t]
                fn = type_false_negatives[t]
                
                precision_class = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_class = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_class = 2 * (precision_class * recall_class) / (precision_class + recall_class) if (precision_class + recall_class) > 0 else 0
                
                f1_scores.append(f1_class)
                
            if f1_scores:
                type_detection_metrics_data["macro_f1"] = round(sum(f1_scores) / len(f1_scores), 3)
        
        # Calculate segmentation metrics
        guided_jaccards, guided_f1s, blind_jaccards, blind_f1s = [], [], [], []
        guided_total, blind_total = 0, 0
        guided_items, blind_items = 0,0

        for r_indiv in relevant_results:
            details_dict = r_indiv.get("details")
            # Get segmentation_details
            segmentation_details_list = []
            if isinstance(details_dict, dict):
                segmentation_details_list = details_dict.get("segmentation_details", [])
            # Get segmentation_blind_details
            segmentation_blind_details_list = []
            if isinstance(details_dict, dict):
                segmentation_blind_details_list = details_dict.get("segmentation_blind_details", [])

            # guided segmentation
            if segmentation_details_list:
                guided_items += len(segmentation_details_list)
                for item in segmentation_details_list:
                    if isinstance(item, dict):
                        true_ids = item.get("true")
                        pred_ids = item.get("pred")
                        if true_ids is not None and pred_ids is not None:
                            true_set = set(true_ids)
                            pred_set = set(pred_ids)
                            intersection = len(true_set.intersection(pred_set))
                            union = len(true_set.union(pred_set))
                            
                            jaccard = intersection / union if union > 0 else 0
                            guided_jaccards.append(jaccard)
                            
                            # F1
                            tp = intersection
                            fp = len(pred_set - true_set)
                            fn = len(true_set - pred_set)
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            guided_f1s.append(f1)
            # fallback to top-level segmentation (for single API calls)
            elif r_indiv.get("segmentation") and isinstance(r_indiv["segmentation"], dict) and r_indiv["segmentation"].get("guided"):
                guided_total +=1
                seg_data = r_indiv["segmentation"]["guided"]
                if seg_data.get("jaccard") is not None: guided_jaccards.append(seg_data["jaccard"])
                if seg_data.get("f1") is not None: guided_f1s.append(seg_data["f1"])
                

            # blind segmentation
            if segmentation_blind_details_list:
                blind_items += len(segmentation_blind_details_list)
                for item in segmentation_blind_details_list:
                    if isinstance(item, dict):
                        true_ids = item.get("true")
                        pred_ids = item.get("pred")
                        if true_ids is not None and pred_ids is not None:
                            true_set = set(true_ids)
                            pred_set = set(pred_ids)
                            intersection = len(true_set.intersection(pred_set))
                            union = len(true_set.union(pred_set))

                            jaccard = intersection / union if union > 0 else 0
                            blind_jaccards.append(jaccard)

                            # F1
                            tp = intersection
                            fp = len(pred_set - true_set)
                            fn = len(true_set - pred_set)
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            blind_f1s.append(f1)
            elif r_indiv.get("segmentation") and isinstance(r_indiv["segmentation"], dict) and r_indiv["segmentation"].get("blind"):
                blind_total += 1
                seg_data = r_indiv["segmentation"]["blind"]
                if seg_data.get("jaccard") is not None: blind_jaccards.append(seg_data["jaccard"])
                if seg_data.get("f1") is not None: blind_f1s.append(seg_data["f1"])

        if guided_jaccards: segmentation_metrics_data["guided"]["jaccard"] = round(sum(guided_jaccards) / len(guided_jaccards), 3)
        if guided_f1s: segmentation_metrics_data["guided"]["f1"] = round(sum(guided_f1s) / len(guided_f1s), 3)
        if blind_jaccards: segmentation_metrics_data["blind"]["jaccard"] = round(sum(blind_jaccards) / len(blind_jaccards), 3)
        if blind_f1s: segmentation_metrics_data["blind"]["f1"] = round(sum(blind_f1s) / len(blind_f1s), 3)
        
        # Create final metrics objects for type detection and segmentation
        type_detection_final_metrics = TypeDetectionMetrics(**type_detection_metrics_data)
        segmentation_final_metrics = SegmentationMetrics(
            guided=SegmentationSubMetrics(**segmentation_metrics_data["guided"]),
            blind=SegmentationSubMetrics(**segmentation_metrics_data["blind"])
        )

        # Add a row with all metrics to the result
        rows.append(MetricsRow(
            model=model_name,
            prompt_strategy=prompt_strategy,
            temperature=temperature,
            conflict_detection=conflict_detection_metrics,
            type_detection=type_detection_final_metrics,
            segmentation=segmentation_final_metrics
        ))
    
    return MetricsResponse(rows=rows)