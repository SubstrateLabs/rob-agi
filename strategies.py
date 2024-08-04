import json
import os
from substrate import Substrate, ComputeJSON, sb, ComputeText, If, Box

from arc_util import task_sets, load_task_set
from computed_result import ComputedResult, MultipleComputedResults
from grid_problem import GridProblem

strategies = """
These are some example strategies for transforming:
1. Element-wise positional matching
2. Grid segmentation and alignment
3. Bilateral grid comparison
4. Ignoring central divider elements
5. Positional equivalence checking
6. Coordinate-based element pairing
7. Comparative pattern transfer
8. Dimensional mapping between input and output
9. Selective element processing
"""
solution_prompt = """
{strategies}
Your task is to come up with a description of the transform based on the pattern from the examples below:
{challenge_description}
"""
transform_prompt = """
{strategies}
Your task is to come up with a description of the transform based on the pattern from the examples below:
{examples}
"""

evaluate_prompt = """
Given the examples and a description of the transform can you evaluate whether the description is correct?
Examples:
{examples}

Transform:
{transform}
"""

improve_prompt = """
Based on the evaluation, the transform and the examples can you improve the transform description?
Evaluation:
{evaluation}

Transform:
{transform}

Examples:
{examples}
"""

solution_from_transform_prompt = """
Given the transform and the examples can you generate the output for the test case?
Transform:
{transform}

Challenge:
{challenge_description}
"""


api_key = os.environ.get("SUBSTRATE_API_KEY")
substrate = Substrate(api_key=api_key, timeout=60 * 5)
ids = ["0520fde7", "0a938d79", "007bbfb7", "6a1e5592"]

challenges, solutions = load_task_set(task_set_name=task_sets["training"])


def generate_output(task_id: str):
    challenge: GridProblem = challenges[task_id]
    description = challenge.to_task_description()
    examples = challenge.example_string()

    # solution = ComputeText(
    #     prompt=solution_prompt.format(strategies=strategies, challenge_description=description),
    #     model="claude-3-5-sonnet-20240620",
    #     max_tokens=4096,
    # )

    transform = ComputeText(
        prompt=transform_prompt.format(strategies=strategies, examples=examples),
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
    )
    evaluate = ComputeText(
        prompt=sb.format(evaluate_prompt, examples=examples, transform=transform.future.text),
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
    )
    improve = ComputeText(
        prompt=sb.format(improve_prompt, evaluation=evaluate.future.text, transform=transform.future.text, examples=examples),
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
    )
    solution_from_transform = ComputeText(
        prompt=sb.format(solution_from_transform_prompt, transform=improve.future.text, challenge_description=description),
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
    )
    extract_prompt = "Given a text that describes an output matrix, extract the output matrix as a JSON list\nInput:\n{input_text}."
    result = ComputeJSON(
        prompt=sb.format(extract_prompt, input_text=solution_from_transform.future.text),
        json_schema=ComputedResult.json_schema(max_outputs=len(challenge.test_cases)),
        model="Llama3Instruct8B",
    )
    compare_prompt = """
    This is the challenge description:
    {challenge_description}
    
    This is the description of the transformation based on observation:
    {transform}
    
    The output generated from attempting to apply the transformation to the test case:
    {output}
    
    The expected output from the test case:
    {expected}
    
    If the output and the expected output do not match can you explain what may be going wrong here?
    """
    compare = ComputeText(
        prompt=sb.format(compare_prompt, challenge_description=description, transform=improve.future.text, output=result.future.json_object["outputs"][0]["values"], expected=solutions[task_id].outputs[0].values),
    )
    res = substrate.run(compare)
    print(res.json)
    out = res.get(compare).text
    print(out)


def display(grid):
    for row in grid:
        print(" ".join(str(x) for x in row))


for task_id in ids:
    generate_output(task_id)
