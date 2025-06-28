"""
Evaluation Metrics for Contradiction Detection Tasks.

1. Classification metrics: Accuracy, Precision, Recall, F1
   - Used for Contradiction Detection and Type Prediction tasks
   - Supports both binary classification (detection) and multi-class (type prediction)

2. Multi-label metrics: Jaccard similarity, F1 score
   - Used for Conflicting Context Segmentation task
   - Evaluates the model's ability to identify which documents contribute to contradictions
"""

from typing import Sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # type: ignore

def classification_metrics(y_true: Sequence[int | str], y_pred: Sequence[int | str], average_method: str = "binary"):
    """
    Calculate classification metrics for contradiction detection or type prediction.
    
    This function computes the standard classification metrics specified in Section 3
    of the paper: accuracy, precision, recall, and F1 score. It handles both:
    
    - Binary classification: For contradiction detection (yes/no)
    - Multi-class classification: For type prediction (self/pair/conditional/none)
    
    Args:
        y_true: Sequence of true labels
        y_pred: Sequence of predicted labels
        average_method: Method for averaging metrics in multi-class case
                       'binary' for detection task, 'macro' for type prediction
    
    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    # Convert all labels to strings for consistency in scikit-learn
    y_true_str = [str(x) for x in y_true]
    y_pred_str = [str(x) for x in y_pred]

    # Handle different averaging methods based on task requirements
    # For binary classification (detection task), use binary averaging
    # For multi-class classification (type prediction), use macro averaging
    if average_method == "binary":
        # Attempt to infer positive label if not standard 0/1, True/False
        unique_labels = sorted(list(set(y_true_str + y_pred_str)))
        pos_label = unique_labels[-1] if len(unique_labels) > 1 else unique_labels[0] if unique_labels else "1"
        
        # Handle cases where labels might only appear in predictions or true values
        # This ensures scikit-learn treats them as valid labels
        labels = unique_labels if len(unique_labels) >=2 else None

        # Calculate precision, recall, F1 with binary averaging
        # zero_division=0 ensures we get 0.0 instead of errors when no positive samples exist
        precision = precision_score(y_true_str, y_pred_str, average=average_method, pos_label=pos_label, zero_division=0, labels=labels)
        recall = recall_score(y_true_str, y_pred_str, average=average_method, pos_label=pos_label, zero_division=0, labels=labels)
        f1 = f1_score(y_true_str, y_pred_str, average=average_method, pos_label=pos_label, zero_division=0, labels=labels)
    else: # For 'macro', 'weighted' etc. (used in type prediction)
        # Calculate metrics with specified averaging method
        # For type prediction, 'macro' averaging treats all classes equally
        precision = precision_score(y_true_str, y_pred_str, average=average_method, zero_division=0)
        recall = recall_score(y_true_str, y_pred_str, average=average_method, zero_division=0)
        f1 = f1_score(y_true_str, y_pred_str, average=average_method, zero_division=0)

    # Return all metrics in a standardized dictionary
    return dict(
        accuracy=float(accuracy_score(y_true_str, y_pred_str)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )

def multilabel_metrics(y_true, y_pred):
    """
    Calculate multi-label metrics for document segmentation task.
    
    This function computes the Jaccard similarity and F1 score for the
    segmentation task as described in Section 3 of the paper.
    
    The metrics evaluate how well the model identifies which documents
    contribute to contradictions, with both guided and blind approaches.
    
    Args:
        y_true: List of lists containing true document IDs
        y_pred: List of lists containing predicted document IDs
        
    Returns:
        Dictionary containing Jaccard similarity and F1 score
    """
    def jaccard(a, b):
        """
        Calculate Jaccard similarity between two sets.
        
        Jaccard = |intersection| / |union|
        
        Args:
            a: Set of true document IDs
            b: Set of predicted document IDs
            
        Returns:
            Jaccard similarity score (0.0-1.0)
        """
        a, b = set(a), set(b)
        # Handle edge cases (empty sets)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        # Standard Jaccard formula: intersection / union
        return len(a & b) / len(a | b)
    
    def f1(a, b):
        """
        Calculate F1 score for multi-label prediction.
        
        F1 = 2 * (precision * recall) / (precision + recall)
        
        Args:
            a: Set of true document IDs
            b: Set of predicted document IDs
            
        Returns:
            F1 score (0.0-1.0)
        """
        a, b = set(a), set(b)
        # Handle edge cases (empty sets)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        # Calculate precision and recall
        p = len(a & b) / len(b) if b else 0.0
        r = len(a & b) / len(a) if a else 0.0
        # Handle division by zero
        if p + r == 0:
            return 0.0
        # Standard F1 formula
        return 2 * p * r / (p + r)
    
    # Calculate metrics for each example
    jaccard_scores = [jaccard(t, p) for t, p in zip(y_true, y_pred)]
    f1_scores = [f1(t, p) for t, p in zip(y_true, y_pred)]
    
    # Calculate averages across all examples
    avg_jaccard = float(sum(jaccard_scores) / len(jaccard_scores)) if jaccard_scores else 0.0
    avg_f1 = float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0
    
    return dict(
        jaccard=avg_jaccard,
        f1=avg_f1,
    )