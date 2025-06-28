"""
Database Utilities for Evaluation Results Storage.

The database schema stores:
1. Individual evaluation runs with detailed metrics
2. Aggregate statistics across multiple runs to track model performance trends
3. Historical data to enable time-series analysis of performance

The implementation uses SQLite for simplicity and portability, storing results
in a database file located in the data directory.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from app.schemas import PromptStrategyType

DB_PATH = Path("data/eval_results.sqlite3")

def init_db():
    """Initialize the database with required tables if they don't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            provider TEXT,
            use_cot INTEGER,
            prompt_strategy TEXT DEFAULT 'basic',
            dataset_path TEXT,
            date TEXT,
            detection TEXT,
            type TEXT,
            segmentation TEXT,
            details TEXT,
            temperature REAL DEFAULT 0
        )
        ''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_aggregate (
            model TEXT,
            provider TEXT,
            use_cot INTEGER,
            prompt_strategy TEXT DEFAULT 'basic',
            temperature REAL DEFAULT 0,
            dataset_count INTEGER,
            total_correct INTEGER,
            total_samples INTEGER,
            total_tp INTEGER,
            total_fp INTEGER,
            total_fn INTEGER,
            prec_valid INTEGER,
            rec_valid INTEGER,
            f1_valid INTEGER,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            PRIMARY KEY (model, provider, use_cot, prompt_strategy, temperature)
        )
        ''')

def save_evaluation_result(model, provider, dataset_path=None, detection=None, type_=None, segmentation=None, details=None, 
                          use_cot=False, prompt_strategy='basic', task=None, metrics=None, temperature=0):
    """Save evaluation results from a single run to the database."""
    if use_cot and prompt_strategy == 'basic':
        prompt_strategy = 'cot'
    
    with sqlite3.connect(DB_PATH) as conn:
        if task and metrics:
            conn.execute(
                f'''INSERT INTO evaluation_results (model, provider, use_cot, prompt_strategy, date, {task}, temperature)
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (
                    model,
                    provider,
                    int(use_cot),
                    prompt_strategy,
                    datetime.utcnow().isoformat(),
                    json.dumps(metrics, ensure_ascii=False),
                    temperature
                )
            )
        else:
            conn.execute(
                '''INSERT INTO evaluation_results (model, provider, use_cot, prompt_strategy, dataset_path, date, detection, type, segmentation, details, temperature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    model,
                    provider,
                    int(use_cot),
                    prompt_strategy,
                    str(dataset_path),
                    datetime.utcnow().isoformat(),
                    json.dumps(detection, ensure_ascii=False) if detection else None,
                    json.dumps(type_, ensure_ascii=False) if type_ else None,
                    json.dumps(segmentation, ensure_ascii=False) if segmentation else None,
                    json.dumps(details, ensure_ascii=False) if details else None,
                    temperature
                )
            )

def update_aggregate(model, provider, prompt_strategy: Union[PromptStrategyType, bool], detection_details, temperature=0):
    """Update aggregate statistics for a model/provider/prompt combination."""
    # Backward compatibility handling
    use_cot = False
    if isinstance(prompt_strategy, bool):
        use_cot = prompt_strategy
        prompt_strategy = "cot" if use_cot else "basic"
    else:
        # For string type, set use_cot based on prompt strategy
        use_cot = (prompt_strategy == "cot")
    
    # Calculate metrics for this evaluation run
    correct = sum(1 for d in detection_details if d["true"] == d["pred"])
    total = len(detection_details)
    
    # Calculate true positives, false positives, and false negatives
    # These are used for precision, recall, and F1 calculations
    tp = sum(1 for d in detection_details if d["true"] == True and d["pred"] == True)
    fp = sum(1 for d in detection_details if d["true"] == False and d["pred"] == True)
    fn = sum(1 for d in detection_details if d["true"] == True and d["pred"] == False)
    
    # Track whether denominators are valid (non-zero) for precision/recall/F1
    # This prevents division by zero when calculating metrics
    prec_den = tp + fp
    rec_den = tp + fn
    f1_den = (tp + fp) + (tp + fn)
    prec_valid = 1 if prec_den > 0 else 0
    rec_valid = 1 if rec_den > 0 else 0
    f1_valid = 1 if (prec_den > 0 and rec_den > 0) else 0
    
    with sqlite3.connect(DB_PATH) as conn:
        # Check if there are existing records for this model/provider/prompt/temperature
        cur = conn.execute(
            'SELECT dataset_count, total_correct, total_samples, total_tp, total_fp, total_fn, prec_valid, rec_valid, f1_valid ' +
            'FROM evaluation_aggregate WHERE model=? AND provider=? AND prompt_strategy=? AND temperature=?', 
            (model, provider, prompt_strategy, temperature)
        )
        row = cur.fetchone()
        
        if row:
            # Update existing aggregate record
            dataset_count, prev_correct, prev_total, prev_tp, prev_fp, prev_fn, prev_prec_valid, prev_rec_valid, prev_f1_valid = row
            new_count = dataset_count + 1
            new_correct = prev_correct + correct
            new_total = prev_total + total
            new_tp = prev_tp + tp
            new_fp = prev_fp + fp
            new_fn = prev_fn + fn
            new_prec_valid = prev_prec_valid + prec_valid
            new_rec_valid = prev_rec_valid + rec_valid
            new_f1_valid = prev_f1_valid + f1_valid
            
            # Calculate updated aggregate metrics
            accuracy = new_correct / new_total if new_total else None
            precision = new_tp / (new_tp + new_fp) if new_prec_valid else None
            recall = new_tp / (new_tp + new_fn) if new_rec_valid else None
            f1 = 2 * precision * recall / (precision + recall) if (precision is not None and recall is not None and (precision + recall) > 0) else None
            
            conn.execute(
                'UPDATE evaluation_aggregate SET dataset_count=?, total_correct=?, total_samples=?, ' +
                'total_tp=?, total_fp=?, total_fn=?, prec_valid=?, rec_valid=?, f1_valid=?, ' +
                'accuracy=?, precision=?, recall=?, f1=?, use_cot=? ' +
                'WHERE model=? AND provider=? AND prompt_strategy=? AND temperature=?',
                (new_count, new_correct, new_total, new_tp, new_fp, new_fn, new_prec_valid, 
                 new_rec_valid, new_f1_valid, accuracy, precision, recall, f1, int(use_cot),
                 model, provider, prompt_strategy, temperature)
            )
        else:
            # Create new aggregate record
            accuracy = correct / total if total else None
            precision = tp / (tp + fp) if prec_valid else None
            recall = tp / (tp + fn) if rec_valid else None
            f1 = 2 * precision * recall / (precision + recall) if (precision is not None and recall is not None and (precision + recall) > 0) else None
            
            conn.execute(
                'INSERT INTO evaluation_aggregate ' +
                '(model, provider, use_cot, prompt_strategy, dataset_count, total_correct, total_samples, ' +
                'total_tp, total_fp, total_fn, prec_valid, rec_valid, f1_valid, accuracy, precision, recall, f1, temperature) ' +
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (model, provider, int(use_cot), prompt_strategy, 1, correct, total, tp, fp, fn, 
                 prec_valid, rec_valid, f1_valid, accuracy, precision, recall, f1, temperature)
            )

# Functions below maintain minimal changes for result retrieval
def get_all_results():
    """
    Retrieve all evaluation results from the database.
    
    Returns:
        List of dictionaries, each containing a complete evaluation record.
        JSON fields are parsed back into Python objects.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute('SELECT * FROM evaluation_results ORDER BY id DESC')
        cols = [d[0] for d in cur.description]
        results = []
        for row in cur.fetchall():
            rec = dict(zip(cols, row))
            # Convert JSON strings back to Python objects
            for k in ["detection", "type", "segmentation", "details"]:
                if k in rec and rec[k]:
                    rec[k] = json.loads(rec[k])
            # Default temperature to 0 if missing
            if "temperature" not in rec or rec["temperature"] is None:
                rec["temperature"] = 0
            results.append(rec)
        return results

def get_all_aggregate():
    """
    Retrieve all aggregate metrics by model/provider/prompt combination.
    
    Returns:
        List of dictionaries, each containing aggregated metrics for a
        specific model, provider, and prompt strategy combination.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute('SELECT * FROM evaluation_aggregate ORDER BY model, provider, prompt_strategy, temperature')
        cols = [d[0] for d in cur.description]
        results = [dict(zip(cols, row)) for row in cur.fetchall()]
        # Default temperature to 0 if missing
        for rec in results:
            if "temperature" not in rec or rec["temperature"] is None:
                rec["temperature"] = 0
        return results

def get_conflict_detection_history():
    """
    Retrieve historical data for conflict detection performance.
    
    Returns:
        List of dictionaries containing evaluation records with
        detection metrics, ordered by evaluation ID (chronological).
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute('SELECT id, model, provider, use_cot, prompt_strategy, dataset_path, date, detection FROM evaluation_results ORDER BY id ASC')
        cols = [d[0] for d in cur.description]
        results = []
        for row in cur.fetchall():
            rec = dict(zip(cols, row))
            if "detection" in rec and rec["detection"]:
                rec["detection"] = json.loads(rec["detection"])
            results.append(rec)
        return results

def get_type_detection_history():
    """
    Retrieve historical data for contradiction type prediction performance.
    
    Returns:
        List of dictionaries containing evaluation records with
        type prediction metrics, ordered by evaluation ID (chronological).
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute('SELECT id, model, provider, use_cot, prompt_strategy, dataset_path, date, type FROM evaluation_results ORDER BY id ASC')
        cols = [d[0] for d in cur.description]
        results = []
        for row in cur.fetchall():
            rec = dict(zip(cols, row))
            if "type" in rec and rec["type"]:
                rec["type"] = json.loads(rec["type"])
            results.append(rec)
        return results

def get_segmentation_history():
    """
    Retrieve historical data for document segmentation performance.
    
    Returns:
        List of dictionaries containing evaluation records with
        segmentation metrics, ordered by evaluation ID (chronological).
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute('SELECT id, model, provider, use_cot, prompt_strategy, dataset_path, date, segmentation FROM evaluation_results ORDER BY id ASC')
        cols = [d[0] for d in cur.description]
        results = []
        for row in cur.fetchall():
            rec = dict(zip(cols, row))
            if "segmentation" in rec and rec["segmentation"]:
                rec["segmentation"] = json.loads(rec["segmentation"])
            results.append(rec)
        return results

# Initialize the database when the module is imported
init_db()
