from scripts.day15.code_quality_check import run_quality_check


def test_day15_code_quality_check_passes_without_running_nested_pytest():
    results = run_quality_check(run_tests=False)

    assert results["python_file_count"] > 0
    assert results["compile_errors"] == []
    assert results["secret_findings"] == []
    assert results["passed"] is True
