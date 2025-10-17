from pydantic import BaseModel


class EvaluationResult(BaseModel):
    """Evaluation metrics for a test case"""

    query: str
    precision_at_5: float
    recall_at_5: float
    f1_score: float
    retrieved_docs: list[str]
    relevant_docs: list[str]


class BenchmarkResult(BaseModel):
    """Benchmark results across test cases"""

    config_name: str
    avg_precision: float
    avg_recall: float
    avg_f1: float
    mrr: float
    map_score: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_test_cases: int
