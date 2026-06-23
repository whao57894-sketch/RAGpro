from scripts.day12.run_integration_test import run_integration_test


def test_day12_system_integration_report_is_generated():
    results = run_integration_test()
    summary = results["summary"]

    assert summary["document_count"] >= 5
    assert summary["question_count"] == 20
    assert summary["retrieval_accuracy"] >= 0.85
    assert summary["answer_accuracy"] >= 0.85
    assert summary["p95_response_ms"] < 1000
