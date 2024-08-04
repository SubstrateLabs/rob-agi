import base64
import json
import os
from typing import List, Literal, Optional

from substrate import Substrate, ComputeText, ComputeJSON, sb, FindOrCreateVectorStore
from arc_util import task_sets, load_task_set
from arc_vec import ResearchEvent
from colored_grid import ColoredGrid
from computed_result import ComputedResult
from grid_problem import GridProblem
from solver_functions import get_initial_impression, extract_result, gather_research

api_key = os.environ.get("SUBSTRATE_API_KEY")
substrate = Substrate(api_key=api_key, timeout=60 * 5)
collection = FindOrCreateVectorStore(collection_name="arc_events_raw", model="jina-v2")

ModelType = Literal[
    "Mistral7BInstruct",
    "Mixtral8x7BInstruct",
    "Llama3Instruct8B",
    "Llama3Instruct70B",
    "Llama3Instruct405B",
    "Firellava13B",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-5-sonnet-20240620",
]
smart_model: ModelType = "Llama3Instruct70B"
json_model: ModelType = "Llama3Instruct8B"

challenges, solutions = load_task_set(task_set_name=task_sets["training"])


def local_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        data = image_file.read()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
        # return f"data:image/jpeg;base64,"
        # return


def visual_parse(challenge: GridProblem):
    path = "data/task_images"
    image_path = os.path.join(path, f"{challenge.id}.png")
    if not os.path.exists(image_path):
        raise f"Image not found for task {challenge.id}"
    uri = local_image_to_base64(image_path)
    look_at = ComputeText(
        prompt="This is an ARC reasoning challenge. The goal is to find the transform that takes the input grids to the output grids. The description of the transform is often simple to express in words. Look at these example input (top) output (bottom) pairs and suggest a high level solution. Concepts like symmetry, rotation, masking, and color patterns are often useful. Your final solution should be short, only a sentence or two.",
        # model="Firellava13B",
        model="claude-3-5-sonnet-20240620",
        # model="gpt-4o",
        image_uris=[uri],
    )
    res = substrate.run(look_at)
    print(json.dumps(res.json, indent=2))


def first_try(challenge: GridProblem):
    if len(challenge.test_cases) > 1:
        final_task = f"Respond in JSON with the solutions for the {len(challenge.test_cases)} test cases in this task."
    else:
        final_task = "Respond in JSON with the solution for the test case in this task."
    reason = ComputeText(prompt=get_initial_impression(challenge), model=smart_model)
    result = ComputeJSON(
        prompt=sb.concat(extract_result(challenge), reason.future.text, final_task),
        json_schema=ComputedResult.json_schema(max_outputs=len(challenge.test_cases)),
        model=json_model,
    )
    res = substrate.run(result)
    print(json.dumps(res.json, indent=2))
    solution = ComputedResult(outputs=[ColoredGrid(**j) for j in res.get(result).json_object["outputs"]])
    print(solution.comparison_report(solutions[challenge.id]))


def research_pass(challenge_list: List[GridProblem], prev_event: Optional[ResearchEvent] = None, passes=1):
    prev_str = json.dumps(prev_event.model_dump()) if prev_event is not None else None
    think = ComputeText(prompt=gather_research(challenge_list, previous_research=prev_str), model=smart_model)

    research = ComputeJSON(
        prompt=gather_research(challenge_list, think.future.text),
        json_schema=ResearchEvent.model_json_schema(),
        model=json_model,
    )
    tail = research
    if passes > 1:
        for i in range(passes - 1):
            next_pass = ComputeJSON(
                prompt=gather_research(challenge_list, sb.jq(tail.future.json_object, "@json")),
                json_schema=ResearchEvent.model_json_schema(),
                model=json_model,
            )
            tail = next_pass

    res = substrate.run(tail)
    print(json.dumps(res.json, indent=2))
    return ResearchEvent.model_validate(res.get(tail).json_object)


def main():
    # id = "0520fde7"
    # 3bd67248
    id = "3bd67248"
    challenge: GridProblem = challenges[id]
    # first_try(challenge)

    # ids = list(challenges.keys())[0:5]
    # challenge_list = [challenges[id] for id in ids]
    # research_pass(challenge_list)

    visual_parse(challenge)


if __name__ == "__main__":
    main()
