import os

from pydantic import ValidationError, BaseModel, json_schema
from pydantic_core.core_schema import chain_schema
from task_loader import load_training_tasks, load_evaluation_tasks, json_task_to_string

from substrate import ComputeJSON, Substrate, ComputeText, SubstrateResponse, sb

prompt_format = """
You are a bot that is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
Identify the pattern, then apply that pattern to the test input to give a final output.
Just give valid json list of lists response back, nothing else. Do not explain your thoughts.
{0}
"""
improve_output_prompt = """
Given the following string, please give me a valid JSON list.
{input_text}
"""
class JSONListOutput(BaseModel):
    output: list[list[int]]

def init_substrate_client():
    api_key = os.environ['SUBSTRATE_API_KEY']
    return Substrate(
        api_key=api_key,
    )

def llm(prompt: str, model: str) -> dict:
    substrate = init_substrate_client()
    # print(prompt)
    text_node = ComputeText(
        prompt=prompt,
        model=model,
        max_tokens=4096,
    )
    improve_output_node = ComputeJSON(
        prompt=sb.format(
            improve_output_prompt,
            input_text=text_node.future.text,
        ),
        model="Llama3Instruct8B",
        json_schema=JSONListOutput.model_json_schema(),
    )
    res = substrate.run(improve_output_node)
    #print(res.json)
    return res.get(improve_output_node).json_object

def generate_output(model: str, challenges: dict, task_id: str, retries: int = 2):
    task_prompt = json_task_to_string(
        challenge_tasks=challenges,
        task_id=task_id,
    )
    prompt = prompt_format.format(task_prompt)
    out = llm(prompt, model)
    print(out)
    return out['output']

def define_node(model: str, challenges: dict, task_id: str):
    task_prompt = json_task_to_string(
        challenge_tasks=challenges,
        task_id=task_id,
    )
    prompt = prompt_format.format(task_prompt)
    text_node = ComputeText(
        prompt=prompt,
        model=model,
        max_tokens=4096,
    )
    improve_output_node = ComputeJSON(
        prompt=sb.format(
            improve_output_prompt,
            input_text=text_node.future.text,
        ),
        model="Llama3Instruct8B",
        json_schema=JSONListOutput.model_json_schema(),
    )
    return improve_output_node

def run_nodes(nodes: list) -> SubstrateResponse:
    substrate = init_substrate_client()
    return substrate.run(*nodes)

def evaluate_training_concurrent(num_tasks: int = -1, model: str = "claude-3-5-sonnet-20240620"):
    challenges, solutions = load_training_tasks()
    count = 0
    right = 0
    if num_tasks == -1:
        num_tasks = len(challenges)

    nodes = []
    task_id_to_node = {}
    for task_id in challenges:
        if count == num_tasks:
            break
        node = define_node(
            challenges=challenges,
            task_id=task_id,
            model=model,
        )
        nodes.append(node)
        task_id_to_node[task_id] = node
        count += 1

    res = run_nodes(nodes)

    failed_response = 0
    for task_id in task_id_to_node:
        print(f"Task: {task_id}")
        expected = solutions[task_id][0]
        try:
            out = res.get(task_id_to_node[task_id]).json_object['output']
        except ValidationError as e:
            print("Failed to pydanic output validaiton")
            failed_response += 1
            continue
        match = expected == out
        print(f"   Correct: {match}")
        if not match:
            num_rows = len(out)
            num_cols = len(out[0]) if num_rows > 0 else 0
            expected_num_rows = len(expected)
            expected_num_cols = len(expected[0])
            print(f"   Expected Grid Size: {expected_num_rows}x{expected_num_cols}")
            print(f"   Prediction Grid Size: {num_rows}x{num_cols}")
            print(f"   Expected: {expected}")
            print(f"   Prediction: {out}")
        else:
            right += 1
    print(f"Failed to get response for {failed_response} outputs")
    return right, num_tasks

def evaluate_training_synchronous(num_tasks: int = -1, model: str = "claude-3-5-sonnet-20240620"):
    challenges, solutions = load_training_tasks()
    count = 0
    right = 0
    if num_tasks == -1:
        num_tasks = len(challenges)

    for task_id in challenges:
        if count == num_tasks:
            break
        print(f"Task: {task_id}")
        out = generate_output(
            challenges=challenges,
            task_id=task_id,
            model=model,
        )
        expected = solutions[task_id][0]
        match = expected == out
        print(f"   Correct: {match}")
        if not match:
            num_rows = len(out)
            num_cols = len(out[0]) if num_rows > 0 else 0
            expected_num_rows = len(expected)
            expected_num_cols = len(expected[0])
            print(f"   Expected Grid Size: {expected_num_rows}x{expected_num_cols}")
            print(f"   Prediction Grid Size: {num_rows}x{num_cols}")
            print(f"   Expected: {expected}")
            print(f"   Prediction: {out}")
        else:
            right += 1
        count += 1
    return right, num_tasks

def evaluate_training(num_tasks: int = -1, model: str = "claude-3-5-sonnet-20240620", concurrent: bool = False):
    print('##############################################')
    print("EVALUATING MODEL: ", model)
    print('##############################################')
    if concurrent:
        right, total = evaluate_training_concurrent(num_tasks=num_tasks, model=model)
    else:
        right, total = evaluate_training_synchronous(num_tasks=num_tasks, model=model)
    print(f"Right: {right}, Total: {total}")
    print(f"Accuracy: {right / total}")
    print()

def main():
    num_tasks = 10
    concurrent = True
    # evaluate_training(num_tasks=num_tasks, model="claude-3-5-sonnet-20240620", concurrent=concurrent)
    # evaluate_training(num_tasks=num_tasks, model="Llama3Instruct405B", concurrent=concurrent)
    evaluate_training(num_tasks=num_tasks, model="Llama3Instruct70B", concurrent=concurrent)
    # evaluate_training(num_tasks=num_tasks, model="Llama3Instruct8B", concurrent=concurrent)
    # evaluate_training(num_tasks=num_tasks, model="gpt-4o", concurrent=concurrent)


if __name__ == '__main__':
    main()
