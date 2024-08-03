import json
import os
from substrate import Substrate, ComputeText, ComputeJSON, sb
from arc_util import load_tasks_from_file, task_sets, json_task_to_string, show_result
from colored_grid import ColoredGrid
from grid_problem import GridProblem
from solver_functions import get_initial_impression

api_key = os.environ.get("SUBSTRATE_API_KEY")
substrate = Substrate(api_key=api_key, timeout=60 * 5)

challenges, solutions = load_tasks_from_file(task_set=task_sets["training"])
id = "0520fde7"
example = challenges[id]
task_string = json_task_to_string(challenges, id, 0)


json_prompt = f"""You have been tasked to solve a spatial reasoning test.
You are given a few examples of a transform that you need to find.
The goal is to find the function that fits the transform, then apply it to the test case.

You were given the task:
{task_string}

Your evaluation was:
"""
final_task = "Respond with the JSON output for the test case."
# write a python function that solves the test case. The function should validate correctly on all the train cases."""

challenge = GridProblem.parse(id=id, examples=example["train"], test_cases=example["test"])

reason = ComputeText(prompt=get_initial_impression(challenge), model="Llama3Instruct70B")
result = ComputeJSON(
    prompt=sb.concat(json_prompt, reason.future.text),
    json_schema=ColoredGrid.model_json_schema(),
    model="Llama3Instruct8B",
)
#
# # a = RunPython(function=print_time)
res = substrate.run(result)
print(json.dumps(res.json, indent=2))
#
# # [[[2, 0, 2], [0, 0, 0], [0, 0, 0]]]
print(show_result(result=res.get(result).json_object["values"], expected=solutions[id]))
