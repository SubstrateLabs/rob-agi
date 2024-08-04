import json

task_sets = {
    "training": {
        "challenges": "../data/inputs/arc-agi_training_challenges.json",
        "solutions": "../data/inputs/arc-agi_training_solutions.json",
    },
    "evaluation": {
        "challenges": "../data/inputs/arc-agi_evaluation_challenges.json",
        "solutions": "../data/inputs/arc-agi_evaluation_solutions.json",
    },
}


def load_tasks_from_file(task_set):
    """
    Loads the tasks from the file and returns the challenges and solutions tasks
    """
    with open(task_set["challenges"], "r") as tasks:
        challenges = json.load(tasks)

    with open(task_set["solutions"], "r") as tasks:
        solutions = json.load(tasks)

    return challenges, solutions


def load_training_tasks():
    """
    Loads the training tasks from the file and returns the challenges and solutions tasks
    """
    return load_tasks_from_file(task_sets["training"])


def load_evaluation_tasks():
    """
    Loads the evaluation tasks from the file and returns the challenges and solutions tasks
    """
    return load_tasks_from_file(task_sets["evaluation"])


def json_task_to_string(challenge_tasks: dict, task_id: str, test_input_index: int = 0) -> str:
    """
    challenge_tasks: dict a list of tasks
    task_id: str the id of the task we want to convert to a string

    Convert your json task into a string so you can pass it to LLM.
    """
    json_task = challenge_tasks[task_id]

    final_output = ""

    train_tasks = json_task["train"]
    test_task = json_task["test"]

    final_output = "Training Examples\n"

    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n["
        for row in task["input"]:
            final_output += f"\n{str(row)},"

        final_output += "]\n\n"
        final_output += f"Example {i + 1}: Output\n["

        for row in task["output"]:
            final_output += f"\n{str(row)},"

        final_output += "]\n\n"

    final_output += "Test\n["
    for row in test_task[test_input_index]["input"]:
        final_output += f"\n{str(row)}"

    final_output += "]\n\nYour Response:"

    return final_output
