import json

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

# output_json_schema = {
#     "type": "object",
#     "properties": {
#         "result": {
#             "type": "array",
#             "items": {"type": "array", "items": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}},
#         }
#     },
#     "required": ["result"],
# }


def load_tasks_from_file(task_set):
    with open(task_set["challenges"], "r") as tasks:
        challenges = json.load(tasks)

    with open(task_set["solutions"], "r") as tasks:
        solutions = json.load(tasks)

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
