import json
import os
from substrate import Substrate, ComputeText, ComputeJSON, sb
from arc_util import load_tasks_from_file, task_sets, json_task_to_string, show_result, load_task_set
from colored_grid import ColoredGrid
from computed_result import ComputedResult
from grid_problem import GridProblem
from solver_functions import get_initial_impression

api_key = os.environ.get("SUBSTRATE_API_KEY")
substrate = Substrate(api_key=api_key, timeout=60 * 5)
id = "0520fde7"

# challenges, solutions = load_tasks_from_file(task_set_name=task_sets["training"])
challenges, solutions = load_task_set(task_set_name=task_sets["training"])
challenge: GridProblem = challenges[id]

# task_string = json_task_to_string(challenges, id, 0)


json_prompt = f"""You have been tasked to solve a spatial reasoning test.
You are given a few examples of a transform that you need to find.
The goal is to find the function that fits the transform, then apply it to the test case.

You were given this challenge:
{challenge.to_task_description()}

Your evaluation was:
"""

if len(challenge.test_cases) > 1:
    final_task = f"Respond in JSON with the solutions for the {len(challenge.test_cases)} test cases in this task."
else:
    final_task = "Respond in JSON with the solution for the test case in this task."

# write a python function that solves the test case. The function should validate correctly on all the train cases."""
# challenge = GridProblem.parse(id=id, train=example["train"], test=example["test"])

reason = ComputeText(prompt=get_initial_impression(challenge), model="Llama3Instruct70B")
result = ComputeJSON(
    prompt=sb.concat(json_prompt, reason.future.text),
    json_schema=ComputedResult.json_schema(max_outputs=len(challenge.test_cases)),
    model="Llama3Instruct8B",
)
#
# # a = RunPython(function=print_time)
res = substrate.run(result)
print(json.dumps(res.json, indent=2))

solution = ComputedResult(outputs=[ColoredGrid(**j) for j in res.get(result).json_object["outputs"]])
print(solution.comparison_report(solutions[id]))
# "007bbfb7": [
#     [
#         [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0],
#         [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0],
#         [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 7, 0, 7, 7, 0, 0, 0, 0]
#     ]
# ],
