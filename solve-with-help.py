import json
import os
from substrate import Substrate, ComputeJSON, sb, ComputeText, If, Box

from arc_util import task_sets, load_task_set
from computed_result import ComputedResult
from grid_problem import GridProblem



api_key = os.environ.get("SUBSTRATE_API_KEY")
substrate = Substrate(api_key=api_key, timeout=60 * 5)
ids = ["0520fde7", "0a938d79", "007bbfb7", "6a1e5592"]

challenges, solutions = load_task_set(task_set_name=task_sets["training"])

hint_0520fde7 = "The transform is that the input is divide into two 3x3 grids. The divider is the number 5. If both 3x3 grids have the number 1 in the same position then the output must have a number 2 in that location, other wise the output must have a number 0 in that location." 

hint_0a938d79 = "The transform is that two different numbers are at two edges of the grid. The output then extends each number in a straight line to the opposite edge of the grid. Then continue alternating the lines while maintaining the exact same distance between the lines all the way to the end of the grid in the direction where there is more space. The output must be the same dimensions as the input."

hint_007bbfb7 = "The transform is that the output is a 9x9 grid where each correspond 3x3 grid of the output corresponds to the a grid in the 3x3 input. If the input grid has a color, the same input pattern is placed in that corresponding 3x3 grid of the output."

hint_6a1e5592 = "The transform is that the number at the top has gaps and the number at the bottom have shapes that can be pushed up to fill the gap. Move the number at the bottom upwards to fill the gaps of the number at the top. The bottom shapes must retain its shape and are simply moved up to gap that it is able to fill. Change the number of the shapes that moved up."

hints = [hint_0520fde7, hint_0a938d79, hint_007bbfb7, hint_6a1e5592]

hint_prompt = "Your task is to select the transforms among the given list of transforms that fits the examples. Do not explain your reasoning, just pick amoung these given transforms.\ntransforms:\n{transforms}\nexamples:\n{examples}"

q_prompt = """You have been tasked to solve a spatial reasoning test.
The goal is to output a matrix in the form of JSON list that fits the transform.
I will give you some hints.
{hint}

{description}

What is the output matrix for the test case:
"""

def generate_output(task_id: str):
    challenge: GridProblem = challenges[task_id]
    description = challenge.to_task_description()
    examples = challenge.example_string()

    hint = ComputeText(
        prompt=hint_prompt.format(transforms="\n\n".join(hints), examples=examples),
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
    )

    solution = ComputeText(
        prompt=sb.format(q_prompt, description=description, hint=hint.future.text),
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
    )
    extract_prompt = "Given a text that describes an output matrix, extract the output matrix as a JSON list\nInput:\n{input_text}."
    result = ComputeJSON(
        prompt=sb.format(extract_prompt, input_text=solution.future.text),
        json_schema=ComputedResult.json_schema(max_outputs=len(challenge.test_cases)),
        model="Llama3Instruct8B",
    )
#
    res = substrate.run(result)
    print(json.dumps(res.json, indent=2))

    out = res.get(result).json_object["outputs"][0]["values"]
    print(out)
    display(out)
    expected = solutions[task_id]
    expected_out = expected.outputs[0].values
    print('#######')
    display(expected_out)
    print('MATCHED', out == expected_out)

description_prompt = "Your task is to describe the transformation of the given examples:\n{examples}"
hint_from_description = "Your task is to select the transforms among the given list of transforms that fits the desription. Do not explain your reasoning, just pick amoung these given transforms.\ntransforms:\n{transforms}\ndescription:\n{description}"
def description_to_hint_to_out(task_id: str):
    challenge: GridProblem = challenges[task_id]
    challenge_description = challenge.to_task_description()
    examples = challenge.example_string()

    description = ComputeText(
        prompt=description_prompt.format(examples=examples),
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
    )
    hint = ComputeText(
        prompt=sb.format(hint_from_description, transforms="\n\n".join(hints), description=description.future.text),
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
    )

    solution = ComputeText(
        prompt=sb.format(q_prompt, description=challenge_description, hint=hint.future.text),
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
    )
    extract_prompt = "Given a text that describes an output matrix, extract the output matrix as a JSON list\nInput:\n{input_text}."
    result = ComputeJSON(
        prompt=sb.format(extract_prompt, input_text=solution.future.text),
        json_schema=ComputedResult.json_schema(max_outputs=len(challenge.test_cases)),
        model="Llama3Instruct8B",
    )

    res = substrate.run(result)
    print(json.dumps(res.json, indent=2))

    out = res.get(result).json_object["outputs"][0]["values"]
    print(out)
    display(out)
    expected = solutions[task_id]
    expected_out = expected.outputs[0].values
    print('#######')
    display(expected_out)
    print('MATCHED', out == expected_out)


def improve_instructions(task_id: str):
    challenge: GridProblem = challenges[task_id]
    challenge_description = challenge.to_task_description()
    examples = challenge.example_string()
    expected = solutions[task_id]
    expected_out = expected.outputs[0].values
    techniquest_list = [
        "resize output",
        "move numbers",
        "change numbers",
    ]
    for _ in range(10):
        hint_generator = ComputeText(

        )

        solution = ComputeText(
            prompt=sb.format(q_prompt, description=challenge_description, hint=hint.future.text),
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
        )
        extract_prompt = "Given a text that describes an output matrix, extract the output matrix as a JSON list\nInput:\n{input_text}."
        result = ComputeJSON(
            prompt=sb.format(extract_prompt, input_text=solution.future.text),
            json_schema=ComputedResult.json_schema(max_outputs=len(challenge.test_cases)),
            model="Llama3Instruct8B",
        )
        box = Box(
            value={
                "output": result.future.json_object["outputs"][0]["values"],
                "expected": expected.outputs[0].values,
            },
        )
        if_condition = If(
            condition=sb.jq(result.future.json_object["outputs"][0]["values"], ""),
            value_if_true=hint_generator.future.text,
            value_if_false="",
        )



def display(grid):
    for row in grid:
        print(" ".join(str(x) for x in row))
#
# for task_id in ids:
#     description_to_hint_to_out(task_id)

challenge: GridProblem = challenges["0a938d79"]
challenge_description = challenge.to_task_description()
print(challenge_description)

