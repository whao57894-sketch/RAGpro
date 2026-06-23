import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import api.main as main_module


def test_run_starts_uvicorn_with_default_host_and_port():
    with patch("api.main.uvicorn.run") as mock_run:
        main_module.run()

    mock_run.assert_called_once_with(main_module.app, host="127.0.0.1", port=8000)


def test_main_script_can_import_src_when_executed_directly():
    main_path = Path("api/main.py").resolve()
    command = (
        "import runpy; "
        f"result = runpy.run_path(r'{main_path}', run_name='__test__'); "
        "print(result['parse_document'].__module__)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "src.document_parser"
