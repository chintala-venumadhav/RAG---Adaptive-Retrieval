
| Method | Retrieval Accuracy | Precision@5 | Recall@5 | Faithfulness | Answer Relevance | Groundedness | Hallucination | Latency (s) | Explainability | Clinical Reliability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline 1 (BM25) | 0.74 | 0.78 | 0.81 | 0.46 | 0.67 | 0.58 | 0.24 | **1.10** | No | 0.60 |
| Baseline 2 (Dense FAISS) | 0.81 | 0.84 | 0.86 | 0.61 | 0.76 | 0.71 | 0.17 | 1.42 | No | 0.72 |
| Baseline 3 (Hybrid RAG) | 0.86 | 0.89 | 0.91 | 0.74 | 0.83 | 0.80 | 0.11 | 1.95 | Partial | 0.81 |
| Baseline 4 (Flat RAG + CrossEncoder) | 0.89 | 0.91 | 0.93 | 0.84 | 0.88 | 0.86 | 0.07 | 2.54 | Partial | 0.87 |
| Baseline 5 (Hierarchical + CrossEncoder) | 0.92 | 0.94 | 0.96 | 0.90 | 0.91 | 0.89 | 0.05 | 2.92 | Yes | 0.91 |
| **Proposed Method** | **0.9766** | **0.9394** | **0.9486** | **0.9678** | **0.9663** | **0.9639** | **0.0383** | **84.45** | **0.9812** | **0.9662** |
