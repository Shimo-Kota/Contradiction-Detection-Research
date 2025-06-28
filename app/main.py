from fastapi import FastAPI
from .api import evaluate, generation
from .api import summary, metrics, contradiction_accuracy
from .api import importance_accuracy, proximity_accuracy, evidence_length_accuracy, sensitivity_analysis

app = FastAPI(title="Contradiction‑Detection‑API", version="1.0.0")

app.include_router(generation.router, prefix="/generate", tags=["core"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["core"])

app.include_router(summary.router, prefix="/summary", tags=["analytics"])
app.include_router(metrics.router, prefix="/metrics", tags=["analytics"])
app.include_router(contradiction_accuracy.router, prefix="/contradiction_accuracy", tags=["analytics"])
app.include_router(importance_accuracy.router, prefix="/importance_accuracy", tags=["analytics"]) 
app.include_router(proximity_accuracy.router, prefix="/proximity_accuracy", tags=["analytics"])
app.include_router(evidence_length_accuracy.router, prefix="/evidence_length_accuracy", tags=["analytics"])
app.include_router(sensitivity_analysis.router, prefix="/sensitivity_analysis", tags=["analytics"])