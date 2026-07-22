"""A.R.I.A. Evaluation Harness (F13).

Runs labeled gold-standard test cases through the pipeline and computes
metrics: WER, entity F1, code accuracy. Results persisted to JSON for
the EvalDashboard.

Usage:
    python -m services.eval_harness          # run full eval
    python -m services.eval_harness case_001 # run single case
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from jiwer import wer as jiwer_wer, process_characters

from data_loader import DATA_DIR

logger = logging.getLogger(__name__)

GOLD_DIR = DATA_DIR / "gold"
RESULTS_DIR = DATA_DIR / "eval_results"


@dataclass
class CaseResult:
    """Metrics for a single gold test case."""
    case_id: str
    description: str
    wer: float = 0.0
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    code_accuracy: float = 0.0
    codes_matched: int = 0
    codes_expected: int = 0
    codes_predicted: int = 0
    soap_similarity: float = 0.0
    passed: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    run_id: str = ""
    timestamp: str = ""
    total_cases: int = 0
    passed_cases: int = 0
    avg_wer: float = 0.0
    avg_entity_f1: float = 0.0
    avg_code_accuracy: float = 0.0
    avg_soap_similarity: float = 0.0
    case_results: list[CaseResult] = field(default_factory=list)
    duration_seconds: float = 0.0


class EvalHarness:
    """Evaluation harness for A.R.I.A. pipeline.

    Loads gold test cases, runs them through the pipeline, and computes
    metrics against expected outputs.
    """

    def __init__(self) -> None:
        self.gold_dir = GOLD_DIR
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_gold_cases(self) -> list[dict]:
        """Load all gold test cases from data/gold/."""
        cases = []
        if not self.gold_dir.exists():
            logger.warning(f"Gold directory not found: {self.gold_dir}")
            return cases

        for case_file in sorted(self.gold_dir.glob("*.json")):
            try:
                with open(case_file, encoding="utf-8") as f:
                    case = json.load(f)
                cases.append(case)
            except Exception as e:
                logger.error(f"Failed to load {case_file}: {e}")

        logger.info(f"Loaded {len(cases)} gold test cases")
        return cases

    def run_single_case(self, case: dict) -> CaseResult:
        """Run a single gold test case through the pipeline and compute metrics."""
        case_id = case.get("id", "unknown")
        description = case.get("description", "")
        transcript = case.get("transcript", "")
        expected = case.get("expected", {})

        start_time = time.time()

        try:
            # Import here to avoid circular imports
            from agent_graph import process_transcript

            # Run pipeline
            result = process_transcript(transcript)

            # Compute metrics
            case_result = CaseResult(
                case_id=case_id,
                description=description,
            )

            # WER: compare transcript vs normalized transcript from scribe
            predicted_transcript = result.get("normalized_transcript", transcript)
            case_result.wer = self._compute_wer(transcript, predicted_transcript)

            # Entity F1: compare expected entities vs extracted entities
            expected_entities = expected.get("entities", [])
            predicted_entities = result.get("entities", [])
            ep, er, ef1 = self._compute_entity_f1(expected_entities, predicted_entities)
            case_result.entity_precision = ep
            case_result.entity_recall = er
            case_result.entity_f1 = ef1

            # Code accuracy: compare expected codes vs predicted codes
            expected_codes = expected.get("codes", [])
            predicted_codes = result.get("codes", [])
            code_acc, matched, exp_count, pred_count = self._compute_code_accuracy(
                expected_codes, predicted_codes
            )
            case_result.code_accuracy = code_acc
            case_result.codes_matched = matched
            case_result.codes_expected = exp_count
            case_result.codes_predicted = pred_count

            # SOAP similarity: compare expected SOAP sections
            expected_soap = expected.get("soap", {})
            predicted_soap = result.get("soap", {})
            case_result.soap_similarity = self._compute_soap_similarity(
                expected_soap, predicted_soap
            )

            # Pass/fail: all metrics above threshold
            case_result.passed = (
                case_result.entity_f1 >= 0.5
                and case_result.code_accuracy >= 0.5
                and case_result.wer <= 0.5
            )

        except Exception as e:
            logger.error(f"Case {case_id} failed: {e}")
            case_result = CaseResult(
                case_id=case_id,
                description=description,
                error=str(e),
            )

        case_result.duration_seconds = time.time() - start_time
        return case_result

    def run_eval(self, case_ids: Optional[list[str]] = None) -> EvalResult:
        """Run evaluation on all or selected gold cases."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()

        cases = self.load_gold_cases()
        if case_ids:
            cases = [c for c in cases if c.get("id") in case_ids]

        case_results = []
        for case in cases:
            logger.info(f"Running case {case.get('id')}...")
            result = self.run_single_case(case)
            case_results.append(result)

        # Aggregate metrics
        eval_result = EvalResult(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            total_cases=len(case_results),
            passed_cases=sum(1 for r in case_results if r.passed),
            case_results=case_results,
            duration_seconds=time.time() - start_time,
        )

        if case_results:
            eval_result.avg_wer = sum(r.wer for r in case_results) / len(case_results)
            eval_result.avg_entity_f1 = sum(r.entity_f1 for r in case_results) / len(case_results)
            eval_result.avg_code_accuracy = sum(r.code_accuracy for r in case_results) / len(case_results)
            eval_result.avg_soap_similarity = sum(r.soap_similarity for r in case_results) / len(case_results)

        # Persist results
        self._save_results(eval_result)

        return eval_result

    def _compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate between reference and hypothesis."""
        if not reference or not hypothesis:
            return 1.0
        try:
            return jiwer_wer(reference, hypothesis)
        except Exception:
            return 1.0

    def _compute_entity_f1(
        self, expected: list[dict], predicted: list[dict]
    ) -> tuple[float, float, float]:
        """Compute entity-level precision, recall, F1.

        Matches entities by type and text similarity.
        """
        if not expected and not predicted:
            return 1.0, 1.0, 1.0
        if not expected or not predicted:
            return 0.0, 0.0, 0.0

        # Normalize entities to comparable format
        expected_set = {
            (e.get("text", "").lower().strip(), e.get("type", "").lower())
            for e in expected
        }
        predicted_set = {
            (e.get("text", "").lower().strip(), e.get("type", "").lower())
            for e in predicted
        }

        # Exact match
        tp = len(expected_set & predicted_set)
        fp = len(predicted_set - expected_set)
        fn = len(expected_set - predicted_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def _compute_code_accuracy(
        self, expected: list[dict], predicted: list[dict]
    ) -> tuple[float, int, int, int]:
        """Compute ICD-10 code accuracy (exact code match)."""
        if not expected:
            return 1.0, 0, 0, len(predicted)

        expected_codes = {c.get("code", "").upper() for c in expected}
        predicted_codes = {c.get("code", "").upper() for c in predicted}

        matched = len(expected_codes & predicted_codes)
        accuracy = matched / len(expected_codes) if expected_codes else 0.0

        return accuracy, matched, len(expected_codes), len(predicted_codes)

    def _compute_soap_similarity(
        self, expected: dict[str, str], predicted: dict[str, str]
    ) -> float:
        """Compute SOAP section similarity (character-level)."""
        if not expected:
            return 1.0

        similarities = []
        for section in ["subjective", "objective", "assessment", "plan"]:
            exp_text = expected.get(section, "")
            pred_text = predicted.get(section, "")

            if not exp_text:
                similarities.append(1.0 if not pred_text else 0.0)
                continue

            try:
                result = process_characters(exp_text, pred_text)
                # Normalize: 1 - (errors / reference_length)
                max_len = max(len(exp_text), 1)
                sim = 1.0 - (result.hits / max_len) if hasattr(result, 'hits') else 0.5
                similarities.append(max(0.0, min(1.0, sim)))
            except Exception:
                similarities.append(0.0)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _save_results(self, result: EvalResult) -> None:
        """Persist evaluation results to JSON."""
        output_file = self.results_dir / f"eval_{result.run_id}.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main() -> None:
    """CLI entry point for running evaluation."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    harness = EvalHarness()

    case_ids = None
    if len(sys.argv) > 1:
        case_ids = sys.argv[1:]

    result = harness.run_eval(case_ids if case_ids else None)

    print(f"\n{'='*60}")
    print(f"A.R.I.A. Evaluation Results — Run {result.run_id}")
    print(f"{'='*60}")
    print(f"Total cases:   {result.total_cases}")
    print(f"Passed:        {result.passed_cases}")
    print(f"Avg WER:       {result.avg_wer:.3f}")
    print(f"Avg Entity F1: {result.avg_entity_f1:.3f}")
    print(f"Avg Code Acc:  {result.avg_code_accuracy:.3f}")
    print(f"Avg SOAP Sim:  {result.avg_soap_similarity:.3f}")
    print(f"Duration:      {result.duration_seconds:.1f}s")
    print(f"{'='*60}")

    for cr in result.case_results:
        status = "PASS" if cr.passed else "FAIL"
        print(f"  [{status}] {cr.case_id}: WER={cr.wer:.3f} F1={cr.entity_f1:.3f} CodeAcc={cr.code_accuracy:.3f}")
        if cr.error:
            print(f"         Error: {cr.error}")


if __name__ == "__main__":
    main()
