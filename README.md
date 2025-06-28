# Beyond One-Size-Fits-All: Model-Specific Multi-Agent Architectures for Contradiction Detection

This repository contains the source code for the paper "Beyond One-Size-Fits-All: Model-Specific Multi-Agent Architectures for Contradiction Detection" submitted to WI-IAT 2025.

## Overview

This FastAPI-based application implements a comprehensive contradiction detection pipeline for Retrieval-Augmented Generation (RAG) systems. The system provides both synthetic dataset generation and model evaluation capabilities for detecting contradictions in retrieved contexts.

## Features

### Core Functionality

- **Synthetic Dataset Generation**: Creates contradiction datasets following Algorithm 1 from the paper
- **Model Evaluation**: Comprehensive evaluation framework for three context validator tasks:
  1. **Conflict Detection**: Binary classification (contradiction present/absent)
  2. **Conflict Type Prediction**: Classification among self/pair/conditional types
  3. **Conflicting Context Segmentation**: Identifying documents involved in contradictions

### Supported Models

- **Claude**: 3 Haiku, 3.5 Sonnet, 3.5 Haiku
- **Gemini**: 2.5 Flash, 2.5 Pro
- **GPT**: GPT-4o, GPT-4o Mini

### Evaluation Strategies

- Basic prompting
- Chain-of-Thought (CoT) prompting
- ECR architecture, which is introduced in the paper

## Installation

### Prerequisites

- Python ≥ 3.11
- HotpotQA dataset (see Dataset Setup section)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/Contradiction-Detection-Research.git
cd Contradiction-Detection-Research
```

1. Install dependencies:

```bash
pip install -e .
```

1. Set up environment variables for API keys:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

### Dataset Setup

Due to file size constraints, the HotpotQA dataset is not included in this repository. To use the system:

1. Download the HotpotQA dataset
2. Place it as `/data/hotpotqa_source.json` in the repository root
3. The system will automatically detect and use this dataset

## Usage

### Starting the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### API Endpoints

#### Dataset Generation

- `POST /generate/` - Generate synthetic contradiction datasets
  - Parameters: `n` (number of examples), `alpha` (salience probability), `context_length`, `model`, `provider`

#### Model Evaluation

- `POST /evaluate/` - Evaluate models on contradiction detection tasks
  - Parameters: `path` (dataset path), `model`, `provider`, `prompt_strategy`, `repeat`

#### Analytics

- `GET /summary/` - Dataset distribution and performance summaries
- `GET /metrics/` - Detailed performance metrics
- `GET /contradiction_accuracy/` - Contradiction detection accuracy analysis
- `GET /importance_accuracy/` - Performance by contradiction importance
- `GET /proximity_accuracy/` - Performance by context proximity
- `GET /evidence_length_accuracy/` - Performance by evidence length
- `GET /sensitivity_analysis/` - Parameter sensitivity analysis

## Architecture

```text
app/
├── main.py                 # FastAPI application entry point
├── schemas.py             # Pydantic data models
├── api/                   # API route handlers
│   ├── generation.py      # Dataset generation endpoints
│   ├── evaluate.py        # Model evaluation endpoints
│   └── ...               # Analytics endpoints
├── core/                  # Core system components
│   ├── config.py          # Configuration management
│   ├── db.py             # Database operations
│   ├── llm_clients.py    # LLM provider clients
│   └── llm_detection.py  # Detection algorithms
├── evaluation/            # Evaluation framework
├── generation/            # Dataset generation
└── prompts/              # Prompt templates and strategies
```

## Dataset Format

The system expects datasets in JSON format with the following structure:

```json
{
  "id": "unique_identifier",
  "query": "question or query",
  "contexts": ["context1", "context2", "context3"],
  "contradiction_type": "none|self|pair|conditional",
  "conflicting_contexts": [0, 1],
  "metadata": {
    "importance": "",
    "proximity": "",
    "evidence_length": ""
  }
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.