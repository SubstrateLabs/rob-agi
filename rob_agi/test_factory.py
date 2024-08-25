import json
import subprocess
import sys
import traceback
from pathlib import Path
from pprint import pformat
from typing import Optional

from rob_agi.computed_result import ComputedResult
from rob_agi.grid_problem import GridProblem


to_replace = "============================= test session starts ==============================\ncollecting ..."


def run_pytest(test_file: Path):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-vv", "--no-header", "--random-order", str(test_file.resolve())],
            # [sys.executable, "-m", "pytest", "-vv", "--no-header", "--random-order", str(test_file.resolve())],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.replace(to_replace, "")

        return {
            "success": result.returncode == 0,
            "output": output,
            "error": result.stderr,
            "returncode": result.returncode,
        }
    except Exception as e:
        trace_str = traceback.format_exc()
        return {"success": False, "output": "", "error": str(e) + trace_str, "returncode": -1}


def get_pytest_error(test_file):
    result = run_pytest(test_file)
    if not result["success"]:
        err = result["error"] or ""
        out = result["output"] or ""
        return err + out
    return result["error"]


def setup_files(gp: GridProblem, cr: Optional[ComputedResult], path: Path):
    path.mkdir(parents=True, exist_ok=True)
    write_test_file(gp, path, cr)
    write_main_file(gp, path)
    write_meta_file(path)
    # write_init_file(path)


def write_init_file(path: Path):
    with open(path / "__init__.py", "w") as f:
        f.write("")


def write_meta_file(path: Path, solved: bool = False, latest_plan: str = None, total_attempts: int = 0):
    target_file = path / "meta.py"

    content = f"""solved = {solved}
latest_plan = {repr(latest_plan)}
total_attempts = {total_attempts}
"""

    with open(target_file, "w") as f:
        f.write(content)


def read_meta_file(base_path: Path) -> tuple[bool, Optional[str], int]:
    target_file = base_path / "meta.py"
    if not target_file.exists():
        write_meta_file(base_path)

    with open(target_file, "r") as f:
        content = f.read()
    namespace = {}
    exec(content, namespace)
    solved = namespace.get("solved", False)
    latest_plan = namespace.get("latest_plan", None)
    total_attempts = namespace.get("total_attempts", 0)
    return solved, latest_plan, total_attempts


def write_main_file(gp: GridProblem, path: Path):
    target_file = path / "main.py"
    if target_file.exists():
        return
    template = f"""from rob_agi.colored_grid import ColoredGrid

def solve_{gp.id}(input_grid: ColoredGrid) -> ColoredGrid:
    pass
"""
    with open(target_file, "w") as f:
        f.write(template)


def write_test_file(gp: GridProblem, path: Path, cr: Optional[ComputedResult] = None):
    examples = gp.examples

    main_import_path = f"rob_agi.attempts.c_{gp.id}.main"

    example_assertions = [
        f"""
def test_{gp.id}_example_{i}():
    input_grid = ColoredGrid(values=
{json_to_python_string(examples[i].input.values)}
    )
    expected = ColoredGrid(values=
{json_to_python_string(examples[i].output.values)}
)
    actual = solve_{gp.id}(input_grid)
    assert actual == expected
"""
        for i, example in enumerate(examples)
    ]
    ex_assertion_list = "\n".join(example_assertions)

    test_assertion_list = ""
    if cr is not None:
        test_cases = gp.test_cases
        test_case_solutions = cr.outputs
        test_case_assertions = [
            f"""
def test_{gp.id}_test_case_{i}():
    input_grid = ColoredGrid(values=
{json_to_python_string(test_case.values)}
    )
    expected = ColoredGrid(values=
{json_to_python_string(test_case_solutions[i].values)}
    )
    actual = solve_{gp.id}(input_grid)
    assert actual == expected
"""
            for i, test_case in enumerate(test_cases)
        ]
        test_assertion_list = "\n".join(test_case_assertions)
    template = f"""import pytest
from rob_agi.colored_grid import ColoredGrid
from {main_import_path} import solve_{gp.id}

{ex_assertion_list}

{test_assertion_list}
"""
    with open(path / "test.py", "w") as f:
        f.write(template)


def json_to_python_string(json_obj):
    json.dumps(json_obj)
    return pformat(json_obj, sort_dicts=False)
