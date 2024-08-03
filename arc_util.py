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

"""
Example Challenge:
"007bbfb7": {
    "test": [
        {
            "input": [[7, 0, 7], [7, 0, 7], [7, 7, 0]]
        }
    ],
    "train": [
        {
            "input": [[0, 7, 7], [7, 7, 7], [0, 7, 7]],
            "output": [
                [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7],
                [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7],
                [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7]
            ]
        }, {
            "input": [[4, 0, 4], [0, 0, 0], [0, 4, 0]],
            "output": [
                [4, 0, 4, 0, 0, 0, 4, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 4, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0]
            ]
        }, {
            "input": [[0, 0, 0], [0, 0, 2], [2, 0, 2]],
            "output": [
                [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 2, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 2], [2, 0, 2, 0, 0, 0, 2, 0, 2]
            ]
        }, {
            "input": [[6, 6, 0], [6, 0, 0], [0, 6, 6]],
            "output": [
                [6, 6, 0, 6, 6, 0, 0, 0, 0], [6, 0, 0, 6, 0, 0, 0, 0, 0], [0, 6, 6, 0, 6, 6, 0, 0, 0],
                [6, 6, 0, 0, 0, 0, 0, 0, 0], [6, 0, 0, 0, 0, 0, 0, 0, 0], [0, 6, 6, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 6, 6, 0, 6, 6, 0], [0, 0, 0, 6, 0, 0, 6, 0, 0], [0, 0, 0, 0, 6, 6, 0, 6, 6]
            ]
        }, {
            "input": [[2, 2, 2], [0, 0, 0], [0, 2, 2]],
            "output": [
                [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 2, 2, 0, 2, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2, 0, 2, 2]
            ]
        }
    ]
},
"""

"""
Example Solution:
  "007bbfb7": [
    [
      [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0],
      [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0],
      [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 7, 0, 7, 7, 0, 0, 0, 0]
    ]
  ],
"""


def load_tasks_from_file(task_set_name) -> tuple[dict, dict]:
    """return all the challenges and solutions from a task set name, e.g. training or evaluation"""
    with open(task_set_name["challenges"], "r") as tasks:
        challenges = json.load(tasks)
    with open(task_set_name["solutions"], "r") as tasks:
        solutions = json.load(tasks)
    return challenges, solutions


def load_task_set(task_set_name) -> tuple[dict, dict]:
    challenges_json, solutions_json = load_tasks_from_file(task_set_name)
    challenges = {k: GridProblem.parse(id=k, **v) for k, v in challenges_json.items()}
    solutions = {k: ComputedResult(outputs=[ColoredGrid(values=g) for g in v]) for k, v in solutions_json.items()}
    return challenges, solutions


def json_task_to_string(challenge_tasks: dict, task_id: str, test_input_index: int) -> str:
    json_task = challenge_tasks[task_id]
    train_tasks = json_task["train"]
    test_task = json_task["test"]
    final_output = f"{len(train_tasks)} Known Examples:\n"
    for i, task in enumerate(train_tasks):
        final_output += f"Known {i + 1} Input:\n["
        for row in task["input"]:
            final_output += f"\n  {str(row)},"
        final_output += "\n]\n\n"
        final_output += f"Known {i + 1} Output:\n["
        for row in task["output"]:
            final_output += f"\n  {str(row)},"
        final_output += "\n]\n\n"

    final_output += "Unknown Test Case:\n["
    for row in test_task[test_input_index]["input"]:
        final_output += f"\n  {str(row)}"

    final_output += "\n]\n"

    return final_output


def compare_result(result, expected):
    for i, row in enumerate(result):
        if row != expected[i]:
            return False
    return True


def show_result(result, expected):
    print("\nOutput:")
    for row in result:
        print(row)
    print("\nExpected:")
    for row in expected:
        print(row)
    print("\nMATCH", compare_result(result, expected))
