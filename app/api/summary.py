from fastapi import APIRouter
from app.schemas import SummaryResponse, SummaryRow
from app.core import db
from collections import Counter

router = APIRouter()

@router.get("/", response_model=SummaryResponse, tags=["Summary"])
def get_summary():
    """
    Get summary statistics by contradiction type.
    """
    all_results = db.get_all_results()
    
    type_counts = Counter()
    
    for record in all_results:
        # For evaluate results, refer to details.detection_details
        details_to_check = []
        if isinstance(record.get("details"), dict):
            details_to_check = record["details"].get("detection_details", [])
        elif isinstance(record.get("details"), list): # For old format or single API
            details_to_check = record["details"]
        
        # If no detection_details, refer to record.type (for type_prediction)
        if not details_to_check and isinstance(record.get("type"), dict):
            conflict_type_val = record["type"].get("predicted_type") # For type_prediction, look at predicted
            if conflict_type_val: # Check not None
                 type_counts[str(conflict_type_val).lower()] += 1
            continue # If no detection_details, skip further processing for this record

        for item in details_to_check:
            if isinstance(item, dict):
                # Get conflict_type from each item in detection_details
                conflict_type_val = item.get("conflict_type")
                if conflict_type_val: # Check not None
                    type_counts[str(conflict_type_val).lower()] += 1
            
    total_valid_records = sum(type_counts.values())
            
    rows = []
    # Standard type mapping for output
    output_type_mapping = {
        "none": "None",
        "self": "Self",
        "pair": "Pair",
        "conditional": "Conditional"
    }
    
    # Ensure all possible types appear in output (even if count is 0)
    for db_type, display_type in output_type_mapping.items():
        count = type_counts[db_type]
        pct = (count / total_valid_records * 100) if total_valid_records > 0 else 0
        rows.append(SummaryRow(type=display_type, count=count, pct=round(pct, 2)))
        
    rows.append(SummaryRow(type="Total", count=total_valid_records, pct=100.0 if total_valid_records > 0 else 0))
    
    return SummaryResponse(rows=rows)