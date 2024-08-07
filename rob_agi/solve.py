import asyncio
import base64
import json
import os
import random
import time
import traceback
from typing import List, Literal, Optional

from substrate import (
    Substrate,
    ComputeText,
    ComputeJSON,
    sb,
    FindOrCreateVectorStore,
    EmbedText,
    QueryVectorStore,
    Box,
    RunPython,
)
from rob_agi.arc_util import load_task_set
from rob_agi.arc_vec import ResearchEvent, SolveAttempt
from rob_agi.computed_result import ComputedResult
from rob_agi.grid_problem import GridProblem
from rob_agi.moa import moa
from rob_agi.solver_functions import (
    get_initial_impression,
    extract_result,
    gather_research,
    explain_research,
    arc_intro,
    attempt_challenge,
    run_eval,
)

api_key = os.environ.get("SUBSTRATE_API_KEY")
substrate = Substrate(
    api_key=api_key, timeout=60 * 5, additional_headers={"x-substrate-fp": "1", "x-substrate-debug": "1"}
)

col_attempts = FindOrCreateVectorStore(collection_name="arc_attempts", model="jina-v2")
col_research = FindOrCreateVectorStore(collection_name="arc_research_events", model="jina-v2")
col_solves = FindOrCreateVectorStore(collection_name="arc_solves", model="jina-v2")
col_knowledge = FindOrCreateVectorStore(collection_name="arc_problems", model="jina-v2")

# smart_model = "Llama3Instruct70B"
smart_model = "claude-3-5-sonnet-20240620"
json_model = "Llama3Instruct8B"
# json_model = "Mixtral8x7BInstruct"

task_set = "training"
# task_set = "evaluation"
challenges, solutions = load_task_set(task_set_name=task_set)

max_concurrent = 20
all_challenges = list(challenges.values())
random.shuffle(all_challenges)


def ensure_db():
    res = substrate.run(col_attempts, col_research, col_knowledge)
    print(json.dumps(res.json, indent=2))


def local_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        data = image_file.read()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


def visual_parse(challenge: GridProblem):
    path = "../data/task_images"
    image_path = os.path.join(path, f"{challenge.id}.png")
    if not os.path.exists(image_path):
        raise f"Image not found for task {challenge.id}"
    uri = local_image_to_base64(image_path)
    look_at = ComputeText(
        prompt="This is an ARC reasoning challenge. The goal is to find the transform that takes the input grids to the output grids. The description of the transform is often simple to express in words. Look at these example input (top) output (bottom) pairs and suggest a high level solution. Concepts like symmetry, rotation, masking, and color patterns are often useful. Your final solution should be short, only a sentence or two.",
        model="claude-3-5-sonnet-20240620",
        # model="Firellava13B",
        # model="gpt-4o",
        image_uris=[uri],
    )
    res = substrate.run(look_at)
    print(json.dumps(res.json, indent=2))


attempted = 0
successful = 0


async def attempt(challenge: GridProblem, run_remote=False, with_solution=False, verbose=False):
    print(f"Attempting {challenge.id}")
    global attempted, successful
    debug_io = {}

    solution_str = f"SOLUTION:\n\n{solutions[challenge.id].result_description()}" if with_solution else ""
    impression_q = get_initial_impression(challenge)
    # reason = ComputeText(prompt=impression_q, model=smart_model, temperature=0.25)
    reason = moa(impression_q)

    debug_io["impression"] = {"in": impression_q, "out": reason.future.value.text}
    if with_solution:
        think_with_solution = sb.format(
            "{impression_q}\nHere is your first impression:\n\n{impression}\n\nNow, here is the solution:\n\n{solution_str}\n\nRevise your approach if necessary to incorporate what you learned from the solution. The important part here is to identify the key concepts and procedures that are necessary to solve this problem. Be comprehensive, detailed, but also concise.",
            impression_q=impression_q,
            impression=reason.future.value.text,
            solution_str=solution_str,
        )
        # reason = ComputeText(prompt=think_with_solution, model=smart_model, temperature=0.2)
        reason = moa(think_with_solution)
        debug_io["think_with_solution"] = {"in": think_with_solution, "out": reason.future.value.text}
    first_try = attempt_challenge(challenge, reason.future.value.text)
    make_attempt = ComputeText(
        prompt=first_try,
        model=smart_model,
        temperature=0.2,
        max_tokens=3000,
    )
    debug_io["attempt"] = {"in": first_try, "out": make_attempt.future.text}
    parse_query = sb.concat(
        "Extract this result to valid JSON:\n\n",
        make_attempt.future.text,
        "\n\n",
    )
    parse_attempt = ComputeJSON(
        prompt=parse_query,
        # json_schema=SolveAttempt.simple_json_schema(),
        json_schema=SolveAttempt.model_json_schema(),
        model="Mixtral8x7BInstruct",
        # model=json_model,
        _max_retries=2,
        temperature=0.25,
        max_tokens=4000,
    )
    if run_remote:
        py_args = {"id": challenge.id, "fn_code": parse_attempt.future.json_object.python_function}
        run_py = RunPython(
            function=run_eval,
            kwargs=py_args,
            pip_install=["pydantic==2.8.2", "substrate", "git+https://github.com/SubstrateLabs/rob-agi.git@b6ff97c"],
        )
        debug_io["run_py"] = {"in": py_args, "out": run_py.future}
    debug_io["parse_attempt"] = {"in": parse_query, "out": parse_attempt.future.json_object}

    if len(challenge.test_cases) > 1:
        final_task = f"Respond in JSON with the solutions for the {len(challenge.test_cases)} test cases in this task."
    else:
        final_task = "\n\nRespond in JSON with the solution for the test case in this task."
    compute_result_query = sb.concat(
        extract_result(challenge), sb.jq(parse_attempt.future.json_object, "@json"), final_task
    )
    result = ComputeJSON(
        prompt=compute_result_query,
        json_schema=ComputedResult.json_schema(max_outputs=len(challenge.test_cases)),
        model=json_model,
        temperature=0.25,
        _max_retries=2,
    )
    debug_io["computed_result"] = {"in": compute_result_query, "out": result.future.json_object}
    has_solution = json.dumps({"with_solution": with_solution}, indent=None, separators=(",", ":"))
    add_py = '{"py_test": (if .out.output.examples == null and .out.output.test_cases == null then "unknown" elif (.out.output.examples | all) and (.out.output.test_cases | all) then "pass" elif (.out.output.examples | all(. == false)) and (.out.output.test_cases | all(. == false)) then "fail" elif (.out.output.examples | any) or (.out.output.test_cases | any) then "partial" else "unknown" end)}'
    jq_q = f". + {has_solution} + {add_py}"

    emb = EmbedText(
        text=sb.concat(challenge.to_task_description(), "\n\nComputed:\n", sb.jq(result.future.json_object, "@json")),
        collection_name=col_attempts.future.collection_name,
        metadata=sb.jq(parse_attempt.future.json_object, jq_q),
        embedded_metadata_keys=[
            "concepts_used",
            "approach",
            "python_function",
            "solution",
            "stdout",
            "error_message",
            "computed_result",
        ],
        # hide=True,
        _max_retries=2,
    )
    input_space = Box(value=debug_io)
    try:
        res = await substrate.async_run(emb, result, input_space)
        if verbose:
            print(json.dumps(res.json, indent=2))
            print("----------------------------------")
            print(res.get(input_space).value)
            print("----------------------------------")
        solution = ComputedResult.model_validate(res.get(result).json_object)
        print(solution.comparison_report(solutions[challenge.id]))
        attempted += 1
    except Exception as e:
        print("Error running", e)
        traceback.print_exc()
        return
    if solution.validate(solutions[challenge.id]):
        successful += 1
        print(f"Solve Rate: {successful} of {attempted} ({successful / attempted:.2%})")
        try:
            await substrate.async_run(
                EmbedText(
                    text=challenge.to_task_description(),
                    collection_name=col_solves.future.collection_name,
                    metadata=res.get(parse_attempt).json_object,
                    doc_id=challenge.id,
                    _max_retry=2,
                )
            )
        except Exception as e:
            print("Error embedding solve", e)
            traceback.print_exc()


async def attempt_python(challenge: GridProblem, verbose=False):
    print(f"Attempting {challenge.id}")
    global attempted, successful
    if len(challenge.test_cases) > 1:
        final_task = f"Respond in JSON with the solutions for the {len(challenge.test_cases)} test cases in this task."
    else:
        final_task = "\n\nRespond in JSON with the solution for the test case in this task."
    reason = ComputeText(prompt=get_initial_impression(challenge), model=smart_model, temperature=0.2)
    make_attempt = ComputeText(
        prompt=attempt_challenge(challenge, reason.future.text),
        model=smart_model,
        temperature=0.2,
        max_tokens=3000,
    )
    parse_attempt = ComputeJSON(
        prompt=sb.concat(
            "Extract this result to valid JSON:\n\n",
            make_attempt.future.text,
            "\n\n",
        ),
        json_schema=SolveAttempt.simple_json_schema(),
        # model="Mixtral8x7BInstruct",
        model=json_model,
        _max_retries=2,
        temperature=0.25,
        max_tokens=4000,
    )
    runit = RunPython(
        function=run_eval,
        kwargs={"id": challenge.id, "fn_code": parse_attempt.future.json_object.python_function},
        pip_install=["pydantic==2.8.2", "substrate", "git+https://github.com/SubstrateLabs/rob-agi.git@b6ff97c"],
    )
    # emb = EmbedText(
    #     text=challenge.to_task_description(),
    #     collection_name=col_attempts.future.collection_name,
    #     metadata=parse_attempt.future.json_object,
    #     embedded_metadata_keys=["concepts_used", "approach", "python_function", "solution", "stdout", "error_message"],
    #     _max_retries=2,
    # )
    try:
        res = substrate.run(parse_attempt)
        print(res.get(runit))
        # print(solution.comparison_report(solutions[challenge.id]))
        attempted += 1
    except Exception as e:
        print("Error running", e)
        return
    False and print(json.dumps(res.json, indent=2))
    # if solution.validate(solutions[challenge.id]):
    #     successful += 1
    #     print(f"Solve Rate: {successful} of {attempted} ({successful / attempted:.2%})")
    #     try:
    #         await substrate.async_run(
    #             EmbedText(
    #                 text=challenge.to_task_description(),
    #                 collection_name=col_solves.future.collection_name,
    #                 metadata=res.get(parse_attempt).json_object,
    #                 doc_id=challenge.id,
    #                 _max_retry=2,
    #             )
    #         )
    #     except Exception as e:
    #         print("Error embedding solve", e)


# def save_research(event: ResearchEvent):
#     res = substrate.run(event)
#     print(json.dumps(res.json, indent=2))
#     return ResearchEvent.model_validate(res.get(event).json_object)


def research_pass(challenge_list: List[GridProblem], prev_event: Optional[ResearchEvent] = None, passes=1):
    if prev_event:
        pretty_fields = f"""

Current Total Knowledge: {prev_event.current_total_knowledge}
Ordered Concept List: {prev_event.ordered_concept_list}
New Knowledge: {prev_event.new_knowledge}
        """
        prev_str = pretty_fields + explain_research()
    else:
        prev_str = ""
    relevant_solutions = {id: solutions[id] for id in [c.id for c in challenge_list]}
    challenge_solutions = "\n- ".join([s.result_description() for id, s in relevant_solutions.items()])
    think = ComputeText(
        prompt=gather_research(challenge_list, previous_research=challenge_solutions + prev_str),
        model=smart_model,
        temperature=0.2,
    )
    research = ComputeJSON(
        prompt=gather_research(challenge_list, think.future.text),
        json_schema=ResearchEvent.model_json_schema(),
        model=json_model,
        temperature=0.25,
        _max_retries=3,
    )
    tail = research
    if passes > 1:
        for i in range(passes - 1):
            next_pass = ComputeJSON(
                prompt=gather_research(challenge_list, sb.jq(tail.future.json_object, "@json")),
                json_schema=ResearchEvent.model_json_schema(),
                model=json_model,
                temperature=0.25,
                _max_retries=3,
            )
            tail = next_pass

    # res = substrate.run(tail)
    # print(json.dumps(res.json, indent=2))
    # return ResearchEvent.model_validate(res.get(tail).json_object)
    return tail


def research_loop(prev_event: Optional[ResearchEvent] = None, passes=1):
    chunk_size = 5
    time_ns = None
    for i in range(0, len(all_challenges), chunk_size):
        challenge_list = all_challenges[i : i + chunk_size]
        evt = research_pass(challenge_list, prev_event, passes)
        extra_metadata = {"previous_id": str(time_ns)} if prev_event is not None and time_ns is not None else {}
        time_ns = time.time_ns()
        embed = EmbedText(
            text=f"Research Event over {chunk_size} challenges",
            collection_name="arc_research_events",
            metadata=sb.jq(
                evt.future.json_object, f". + {json.dumps(extra_metadata, indent=None, separators=(',', ':'))}"
            ),
            # metadata=evt.future.json_object,
            embedded_metadata_keys=["current_total_knowledge", "new_knowledge"],
            doc_id=str(time_ns),
            _max_retries=2,
        )
        res = substrate.run(evt, embed)
        print(json.dumps(res.json, indent=2))
        print("\nWrote EMB", res.get(embed).embedding.doc_id)
        prev_event = ResearchEvent.model_validate(res.get(evt).json_object)
        print(prev_event.current_total_knowledge)


def distill_research():
    researches = QueryVectorStore(
        collection_name="arc_research_events",
        query_strings=[
            arc_intro(short=True)
            + "We need to find and distill all knowledge learned so far about solving these challenges. Prefer recent, comprehensive, quality research items"
        ],
        top_k=12,
        include_metadata=True,
        include_values=False,
        model="jina-v2",
    )
    summarize = ComputeText(
        model=smart_model,
        prompt=sb.format(
            """Listed below is a collection of research logs after exploring a set of spatial reasoning problems.
    The goal is to distill the most important knowledge learned from all the research sessions.
    The research logs contain summaries of the current total knowledge, ordered concept list, and new knowledge gained.
    The distilled knowledge should capture the most important concepts and insights that will be necessary to solve the challenges in the dataset. 
    The distillation process is also time to organize and prioritize the concepts and ideas that have been gathered.
    
    <RESEARCH_LOGS>
    {logs}
    </RESEARCH_LOGS>
    
    Based on the research logs, distill the most important knowledge learned so far.
    Respond with a single new object with keys: current_total_knowledge, ordered_concept_list, new_knowledge
        """,
            logs=sb.jq(researches.future.results[0], "map(.metadata) | @json"),
        ),
    )
    parse_json = ComputeJSON(
        prompt=sb.concat(
            summarize.future.text,
            "\n\nRespond with a single new JSON object with keys: current_total_knowledge, ordered_concept_list, new_knowledge",
        ),
        json_schema=ResearchEvent.model_json_schema(),
        _max_retries=2,
    )
    time_ns = time.time_ns()
    extra_metadata = {"distillation": True}
    embed = EmbedText(
        text=f"Research Logs",
        collection_name="arc_research_events",
        metadata=sb.jq(
            parse_json.future.json_object, f". + {json.dumps(extra_metadata, indent=None, separators=(',', ':'))}"
        ),
        embedded_metadata_keys=["current_total_knowledge", "new_knowledge"],
        doc_id=str(time_ns),
        _max_retries=2,
    )
    res = substrate.run(embed)
    print(json.dumps(res.json, indent=2))


async def process_challenge(semaphore, challenge):
    async with semaphore:
        return await attempt(challenge, with_solution=True)


async def solve_loop():
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [process_challenge(semaphore, challenge) for challenge in all_challenges]
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        t0 = time.perf_counter()
        try:
            await task
            print(f"Finished {i} of {len(all_challenges)} [{time.perf_counter() - t0:.2f}s]")
        except Exception as e:
            traceback.print_exc()
            print(f"Error on task {i}: {e}")
    print("\n\n===============================================")
    print("FINAL STATS")
    print(f"Attempted: {attempted}")
    print(f"Successful: {successful}")
    print(f"Solve Rate: {successful / attempted:.2%}")
    print("===============================================\n\n")


async def main():
    # ensure_db()
    # id = "0520fde7"
    id = "3bd67248"
    # id = "1f876c06"
    # challenge: GridProblem = challenges[id]

    # random_challenge = random.choice(all_challenges)
    # await attempt(random_challenge, with_solution=True, verbose=True, run_remote=True)

    # ids = list(challenges.keys())[0:5]
    # challenge_list = [challenges[id] for id in ids]
    # tail = research_pass(challenge_list)
    # res = substrate.run(tail)
    # print(json.dumps(res.json, indent=2))
    # return ResearchEvent.model_validate(res.get(tail).json_object)

    # visual_parse(challenge)

    # last = ResearchEvent(**latest_research)
    # research_loop(prev_event=last)

    # distill_research()

    await solve_loop()


if __name__ == "__main__":
    asyncio.run(main())
