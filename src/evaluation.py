"""Architecture-independent intelligence evaluation system for Model Culi."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

EVALUATION_CATEGORIES = [
    "vietnamese_language",
    "english_language",
    "rust_code",
    "dioxus_code",
    "reasoning",
    "planning",
    "tool_selection",
]

SCORING_MODES = {"exact", "contains", "manual"}


class ModelAdapter(Protocol):
    name: str
    version: str

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class TestCase:
    id: str
    category: str
    input: str
    expected: Any = None
    scoring: str = "exact"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("test case id must not be empty")
        if self.category not in EVALUATION_CATEGORIES:
            raise ValueError(f"invalid category: {self.category}")
        if self.scoring not in SCORING_MODES:
            raise ValueError(f"invalid scoring mode: {self.scoring}")


@dataclass
class EvaluationDataset:
    dataset_id: str
    version: str
    cases: list[TestCase]

    def validate(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        if not self.version:
            raise ValueError("version must not be empty")
        if not self.cases:
            raise ValueError("dataset must contain at least one test case")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("test case ids must be unique")
        for case in self.cases:
            case.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationDataset":
        cases = [TestCase(**case) for case in data.get("cases", [])]
        dataset = cls(
            dataset_id=data.get("dataset_id", ""),
            version=data.get("version", ""),
            cases=cases,
        )
        dataset.validate()
        return dataset

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationDataset":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset_id": self.dataset_id,
                    "version": self.version,
                    "cases": [asdict(case) for case in self.cases],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    def cases_for(self, category: str) -> list[TestCase]:
        return [case for case in self.cases if case.category == category]


@dataclass
class CaseResult:
    case_id: str
    category: str
    input: str
    expected: Any
    output: str | None
    score: float | None
    passed: bool | None
    scoring: str
    status: str
    error: str | None
    latency_seconds: float


@dataclass
class EvaluationResult:
    evaluation_id: str
    timestamp_utc: str
    model_name: str
    model_version: str
    dataset_id: str
    dataset_version: str
    case_results: list[CaseResult]
    category_scores: dict[str, float | None]
    overall_score: float | None
    total_runtime_seconds: float
    peak_rss_bytes: int
    manual_review_count: int
    implemented_measurements: list[str]
    planned_measurements: list[str]


IMPLEMENTED_MEASUREMENTS = [
    "per_case_latency_seconds",
    "total_runtime_seconds",
    "peak_rss_bytes",
    "automatic_scores",
    "manual_review_flags",
]

PLANNED_MEASUREMENTS = [
    "human_reviewed quality scores",
    "throughput per task",
]


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).lower().split())


def score_case(case: TestCase, output: str | None) -> tuple[float | None, bool | None, str]:
    if case.scoring == "manual":
        return None, None, "needs_review"
    if case.expected is None:
        return None, None, "missing_expected"
    if output is None or output == "":
        return 0.0, False, "empty_output"
    if case.scoring == "exact":
        passed = _normalize(output) == _normalize(case.expected)
        return (1.0 if passed else 0.0), passed, "scored"
    if case.scoring == "contains":
        passed = _normalize(case.expected) in _normalize(output)
        return (1.0 if passed else 0.0), passed, "scored"
    return None, None, "unsupported_scoring"


def _peak_rss_bytes() -> int:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except (ImportError, AttributeError, OSError):
        return 0


class EvaluationRunner:
    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter

    def run(self, dataset: EvaluationDataset) -> EvaluationResult:
        dataset.validate()
        start = time.perf_counter()
        case_results: list[CaseResult] = []
        for case in dataset.cases:
            case_start = time.perf_counter()
            output: str | None = None
            error: str | None = None
            status = "completed"
            try:
                raw_output = self.adapter.generate(case.input)
                output = "" if raw_output is None else str(raw_output)
            except Exception as exc:  # noqa: BLE001 - model failures must be recorded
                status = "model_error"
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - case_start
            if status == "completed":
                score, passed, score_status = score_case(case, output)
                status = score_status
            else:
                score, passed = None, None
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    category=case.category,
                    input=case.input,
                    expected=case.expected,
                    output=output,
                    score=score,
                    passed=passed,
                    scoring=case.scoring,
                    status=status,
                    error=error,
                    latency_seconds=latency,
                )
            )

        category_scores: dict[str, float | None] = {}
        for category in EVALUATION_CATEGORIES:
            scores = [
                result.score
                for result in case_results
                if result.category == category and result.score is not None
            ]
            category_scores[category] = (sum(scores) / len(scores)) if scores else None

        numeric_overall = [result.score for result in case_results if result.score is not None]
        overall_score = (sum(numeric_overall) / len(numeric_overall)) if numeric_overall else None
        manual_review_count = sum(1 for result in case_results if result.scoring == "manual")

        return EvaluationResult(
            evaluation_id=f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{id(self) % 100000}",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model_name=getattr(self.adapter, "name", type(self.adapter).__name__),
            model_version=getattr(self.adapter, "version", "unknown"),
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            case_results=case_results,
            category_scores=category_scores,
            overall_score=overall_score,
            total_runtime_seconds=time.perf_counter() - start,
            peak_rss_bytes=_peak_rss_bytes(),
            manual_review_count=manual_review_count,
            implemented_measurements=list(IMPLEMENTED_MEASUREMENTS),
            planned_measurements=list(PLANNED_MEASUREMENTS),
        )

    def save(self, result: EvaluationResult, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(result), handle, indent=2, ensure_ascii=False)


def compare_evaluations(baseline: EvaluationResult, candidate: EvaluationResult) -> dict[str, Any]:
    categories = sorted(set(baseline.category_scores) | set(candidate.category_scores))
    deltas: dict[str, float | None] = {}
    for category in categories:
        base_score = baseline.category_scores.get(category)
        candidate_score = candidate.category_scores.get(category)
        if base_score is None or candidate_score is None:
            deltas[category] = None
        else:
            deltas[category] = candidate_score - base_score

    if baseline.overall_score is None or candidate.overall_score is None:
        overall_delta = None
    else:
        overall_delta = candidate.overall_score - baseline.overall_score

    return {
        "baseline_evaluation_id": baseline.evaluation_id,
        "candidate_evaluation_id": candidate.evaluation_id,
        "baseline_model": f"{baseline.model_name} {baseline.model_version}",
        "candidate_model": f"{candidate.model_name} {candidate.model_version}",
        "category_score_deltas": deltas,
        "overall_score_delta": overall_delta,
    }
