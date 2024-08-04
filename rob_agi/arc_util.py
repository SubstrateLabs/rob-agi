import json
from importlib import resources
from pathlib import Path

from colored_grid import ColoredGrid
from computed_result import ComputedResult
from grid_problem import GridProblem


def get_data_path(filename):
    try:
        # Try to get the path using importlib.resources (Python 3.7+)
        with resources.path("arcagi.data", filename) as path:
            return str(path)
    except ImportError:
        # Fallback for older Python versions or if the above fails
        return str(Path(__file__).parent.parent / "data" / "inputs" / filename)


task_sets = {
    "training": {
        "challenges": "arc-agi_training_challenges.json",
        "solutions": "arc-agi_training_solutions.json",
    },
    "evaluation": {
        "challenges": "arc-agi_evaluation_challenges.json",
        "solutions": "arc-agi_evaluation_solutions.json",
    },
}


def load_tasks_from_file(task_set_name) -> tuple[dict, dict]:
    """Return all the challenges and solutions from a task set name, e.g. training or evaluation"""
    challenges_path = get_data_path(task_sets[task_set_name]["challenges"])
    solutions_path = get_data_path(task_sets[task_set_name]["solutions"])

    with open(challenges_path, "r") as tasks:
        challenges = json.load(tasks)
    with open(solutions_path, "r") as tasks:
        solutions = json.load(tasks)
    return challenges, solutions


def load_task_set(task_set_name) -> tuple[dict[str, GridProblem], dict[str, ComputedResult]]:
    challenges_json, solutions_json = load_tasks_from_file(task_set_name)
    challenges = {k: GridProblem.parse(id=k, **v) for k, v in challenges_json.items()}
    solutions = {
        k: ComputedResult(outputs=[ColoredGrid(values=g) for g in v], task_id=k) for k, v in solutions_json.items()
    }
    return challenges, solutions
