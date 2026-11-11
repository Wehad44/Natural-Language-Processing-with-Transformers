# Natural-Language-Processing-with-Transformers

Official SDAIA Academy GitHub: https://github.com/SDAIAAcademy
Wehad alhenaki https://github.com/Wehad44
Dhuha Alsharif https://github.com/Dhuha-Alsharif/Final_AppliedNLP_Project


# Bayan — Bilingual Citizen Feedback Intelligence Service

**Course:** SDA-AIE-211  
**Project type:** Natural Language Processing (NLP)  
**Languages:** Arabic and English

## Overview

Bayan is a bilingual citizen-feedback analysis service. It accepts an Arabic or English complaint and combines three NLP components to return structured information:

1. **Topic classification** — predicts the complaint category.
2. **Named Entity Recognition (NER)** — extracts entities such as people, locations, and services.
3. **Semantic search** — retrieves similar historical cases from the Bayan cases dataset.

The project also optimizes the topic-classification model with ONNX Runtime dynamic INT8 quantization and exposes the integrated components through a FastAPI service with a simple Arabic web interface.

## Features

- Arabic and English complaint input.
- Text preprocessing and masking of personal information (PII), including email addresses, Saudi mobile numbers, and national ID/Iqama-like numbers.
- Eight-topic complaint classification.
- NER using a Hugging Face token-classification pipeline.
- Multilingual semantic retrieval using Sentence Transformers and FAISS.
- Two-stage retrieval that combines semantic similarity with location, date, and predicted-topic signals.
- FastAPI endpoints for health checks, classification, and full complaint analysis.
- ONNX FP32 and dynamically quantized INT8 model versions.

## Classification Categories

The topic classifier predicts one of the following categories:

| Category | Description |
|---|---|
| `account_access` | Account access and login issues |
| `appointments` | Appointment-related issues |
| `billing` | Payment and billing issues |
| `digital_services` | Digital service issues |
| `outage` | Service interruptions |
| `permits` | Permit-related issues |
| `transport` | Transportation-related feedback |
| `waste` | Waste management issues |

## System Workflow

```text
Arabic / English Complaint
            |
            v
     Input Validation
            |
            v
      Text Processing
      (PII masking)
            |
      +-----+------------------+
      |            |           |
      v            v           v
 Classification   NER     Semantic Search
      |            |       FAISS candidates
      |            |           |
      |            |      Metadata re-ranking
      +------------+-----------+
                   |
                   v
          Structured API Response
```

For classification, the service applies the preprocessing pipeline used by the classifier. For NER and semantic search, the input is masked for PII before processing.

## Models and Technologies

| Component | Technology |
|---|---|
| Language | Python |
| Classification | Fine-tuned Hugging Face sequence-classification model |
| Model optimization | PyTorch, ONNX, ONNX Runtime |
| Quantization | Dynamic INT8 quantization |
| NER | Hugging Face Transformers token-classification pipeline |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (or the model recorded in the search manifest) |
| Vector search | FAISS |
| API | FastAPI |
| API server | Uvicorn |
| Request testing | HTTPX |
| Data processing | Pandas, NumPy |
| Evaluation | Scikit-learn / custom evaluation code |

## Semantic Search

Bayan uses a FAISS inner-product index over normalized multilingual sentence embeddings. The search pipeline has two stages:

1. Retrieve candidate cases using semantic similarity.
2. Re-rank candidates using semantic similarity together with location, date, and predicted-topic matches.

The service can return an empty result when the query is outside the supported city scope or when the best semantic match does not meet the configured threshold.

The cases index is versioned as `bayan-cases-v1`. Its manifest records the embedding model, vector dimension, indexed fields, preprocessing version, case count, and dataset checksum.

## API

The FastAPI application exposes the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Returns service status and active model/index versions |
| `POST` | `/v1/classify` | Returns the predicted complaint topic |
| `POST` | `/v1/analyze` | Returns classification, extracted entities, and similar cases |

### Example request

```http
POST /v1/analyze
Content-Type: application/json
```

```json
{
  "text": "تأخر الإصلاح في حي النرجس بالرياض وتم خصم المبلغ مرتين"
}
```

The `/v1/analyze` response contains:

- `input_text`: input text after PII masking.
- `classification`: predicted topic and model/preprocessing version information.
- `entities`: extracted entities and confidence scores.
- `similar_cases`: retrieved historical cases.
- `search_info`: search status and parsed metadata.

The request text must contain between 1 and 2,000 characters.

## Model Optimization Results

The notebook benchmarked 100 predictions for each configuration. The measured results were:

| Configuration | p50 latency (ms) | p99 latency (ms) |
|---|---:|---:|
| PyTorch FP32, max length 512, fixed padding | 606.07 | 1581.92 |
| PyTorch FP32, max length 128, dynamic padding | 68.52 | 241.77 |
| ONNX FP32, max length 128 | 108.05 | 186.36 |
| ONNX INT8, max length 128 | 85.00 | 139.77 |

The INT8 model achieved **97% prediction agreement** with the original PyTorch model on a 100-message sample.

| Model artifact | Size |
|---|---:|
| ONNX FP32 | 470.8 MB |
| ONNX INT8 | 118.2 MB |

Based on the notebook's 97% agreement threshold, INT8 was selected for the API in this run. The ONNX FP32 model was retained as a rollback option.

*Latency measurements depend on the runtime environment and hardware; results may differ on another machine.*

## Semantic Search Evaluation

The notebook evaluated 130 queries with known relevant cases and 20 queries expected to have no answer.

| Metric | FAISS only | Two-stage search |
|---|---:|---:|
| Recall@10 | 0.308 | 1.000 |
| MRR@10 | 0.215 | 0.962 |
| Cross-language MRR | 0.005 | 0.935 |
| No-answer rejection | 0.000 | 0.900 |

In this notebook run, the two-stage search improved retrieval results compared with FAISS-only retrieval. It also rejected 90% of the queries marked as having no answer.

These are the recorded results for the supplied evaluation set; they should not be interpreted as a guarantee of performance on new data.

## NER Aggregation Strategy

The notebook compared `simple`, `first`, and `max` aggregation strategies on Arabic and English examples. The `first` strategy was selected in the API configuration because it keeps prefixed Arabic location words together (for example, `بجدة`) rather than splitting them into fragments.

## Setup

The provided implementation is a Google Colab notebook and expects the project data and trained model artifacts to be available in Google Drive.

### 1. Open the notebook

Open `Day4_Bayan_Lab_with_API.ipynb` in Google Colab.

### 2. Mount Google Drive

Run the notebook's Drive-mount cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Install dependencies

The notebook installs the main dependencies with:

```python
%pip install transformers onnx onnxscript onnxruntime fastapi uvicorn httpx torch numpy
%pip install sentence-transformers faiss-cpu
```

### 4. Provide the required project artifacts

Make sure the project files are available in Google Drive. The notebook searches for:

- `bayan_topic_model/` — trained topic-classification model.
- `bench_mix.npy` — benchmark messages.
- `bayan_cases.csv` — historical cases dataset.
- `bayan_cases.faiss` — FAISS index (the notebook can build it if it is absent).
- `bayan_search_manifest.json` — search-index metadata, when available.
- `bayan_ner_model/` — trained NER model.

The notebook saves generated model and search artifacts under `Bayan_Work/` in Google Drive.

### 5. Run the notebook

Run the cells in order. The notebook loads the classifier, benchmarks and exports ONNX models, evaluates INT8 agreement, prepares preprocessing and NER, loads or builds the FAISS index, and starts the API.

## Running and Testing the API

The notebook starts Uvicorn on `127.0.0.1:8000` and performs a health check and an example analysis request. When running in Colab, it also serves the web interface in an output iframe.

The web interface allows a user to enter an Arabic or English complaint and view its predicted topic, extracted entities, and similar cases.

If the server is already running, stop the existing server before starting it again to avoid a port-8000 conflict. Run the notebook's final server-stop cell when finished.


## Future Improvements

- Expand and balance Arabic dialect coverage in the training and evaluation data.
- Improve NER handling of Arabic clitics, names, and multi-token locations.
- Review retrieval failures and no-answer cases to refine search thresholds and ranking.
- Evaluate the service on larger, more diverse real-world feedback datasets.
- Add production deployment, monitoring, authentication, and automated tests.

**
