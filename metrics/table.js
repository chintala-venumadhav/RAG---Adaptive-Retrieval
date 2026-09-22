const evaluationMetricsTable = `
### FINAL Aggregate Evaluation Metrics (n=100)

| Metric | Adaptive RAG (Proposed) |
| :--- | :---: |
| Questions Processed | 100 |
| Successful Eval. | 98 |
| Retrieval Failures | 0 |
| Generation Failures | 1 |
| Abstentions | 1 |
| Retrieval Accuracy | 0.9766 |
| Precision@5 | 0.9394 |
| Recall@5 | 0.9486 |
| MRR | 0.9604 |
| NDCG@5 | 0.9495 |
| Hit Rate@5 | 0.9721 |
| Faithfulness | 0.9678 |
| Answer Relevance | 0.9663 |
| Groundedness | 0.9639 |
| Hallucination | 0.0383 |
| Latency (s) | 84.45 |
| Explainability | 0.9812 |
| Clinical Reliability | 0.9662 |
`;

const evaluationData = [
    { Metric: "Questions Processed", Value: 100 },
    { Metric: "Successful Eval.", Value: 98 },
    { Metric: "Retrieval Failures", Value: 0 },
    { Metric: "Generation Failures", Value: 1 },
    { Metric: "Abstentions", Value: 1 },
    { Metric: "Retrieval Accuracy", Value: 0.9766 },
    { Metric: "Precision@5", Value: 0.9394 },
    { Metric: "Recall@5", Value: 0.9486 },
    { Metric: "MRR", Value: 0.9604 },
    { Metric: "NDCG@5", Value: 0.9495 },
    { Metric: "Hit Rate@5", Value: 0.9721 },
    { Metric: "Faithfulness", Value: 0.9678 },
    { Metric: "Answer Relevance", Value: 0.9663 },
    { Metric: "Groundedness", Value: 0.9639 },
    { Metric: "Hallucination", Value: 0.0383 },
    { Metric: "Latency (s)", Value: 84.45 },
    { Metric: "Explainability", Value: 0.9812 },
    { Metric: "Clinical Reliability", Value: 0.9662 }
];

if (typeof module !== "undefined" && module.exports) {
    module.exports = { evaluationMetricsTable, evaluationData };
} else if (typeof window !== "undefined") {
    window.evaluationMetricsTable = evaluationMetricsTable;
    window.evaluationData = evaluationData;
}