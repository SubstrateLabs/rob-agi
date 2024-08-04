import json

from colored_grid import ColoredGrid
from computed_result import ComputedResult
from grid_problem import GridProblem

task_sets = {
    "training": {
        "challenges": "data/inputs/arc-agi_training_challenges.json",
        "solutions": "data/inputs/arc-agi_training_solutions.json",
    },
    "evaluation": {
        "challenges": "data/inputs/arc-agi_evaluation_challenges.json",
        "solutions": "data/inputs/arc-agi_evaluation_solutions.json",
    },
}


def load_tasks_from_file(task_set_name) -> tuple[dict, dict]:
    """return all the challenges and solutions from a task set name, e.g. training or evaluation"""
    with open(task_set_name["challenges"], "r") as tasks:
        challenges = json.load(tasks)
    with open(task_set_name["solutions"], "r") as tasks:
        solutions = json.load(tasks)
    return challenges, solutions


def load_task_set(task_set_name) -> tuple[dict[str, GridProblem], dict[str, ComputedResult]]:
    challenges_json, solutions_json = load_tasks_from_file(task_set_name)
    challenges = {k: GridProblem.parse(id=k, **v) for k, v in challenges_json.items()}
    solutions = {
        k: ComputedResult(outputs=[ColoredGrid(values=g) for g in v], task_id=k) for k, v in solutions_json.items()
    }
    return challenges, solutions
