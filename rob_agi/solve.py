import asyncio
import base64
import json
import os
import random
import time
import traceback
from typing import List, Optional

from openai.lib._pydantic import to_strict_json_schema
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
    If,
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
# json_model = "Llama3Instruct8B"
json_model = "Mixtral8x7BInstruct"
gpt = "gpt-4o"

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


async def get_previous_tries(challenge: GridProblem):
    prev_attempts = QueryVectorStore(
        collection_name="arc_attempts",
        model="jina-v2",
        query_strings=["correct response"],
        top_k=10,
        include_metadata=True,
        include_values=False,
        filters={"task_id": {"$eq": challenge.id}},
    )
    solved = QueryVectorStore(
        collection_name="arc_solves",
        model="jina-v2",
        query_strings=["correct response"],
        top_k=1,
        include_metadata=True,
        include_values=False,
        filters={"task_id": {"$eq": challenge.id}},
    )
    has_attempts = sb.jq(prev_attempts.future.results[0], "length > 0")
    not_solved = sb.jq(solved.future.results[0], "length == 0")
    ir = Box(value={"has_attempts": has_attempts, "unsolved": not_solved})
    should_summ = sb.jq(ir.future.value, ".has_attempts and .unsolved")

    past_attempts = sb.jq(
        prev_attempts.future.results[0],
        'group_by(.task_id) | map(.[0]) | map(.metadata.approach) | flatten | join("\n")',
    )
    best_past_attempt = sb.jq(
        prev_attempts.future.results[0],
        'first? | .metadata.approach? | if type == "array" then join("\n") else tostring end | . // ""',
    )
    past_solve_str = sb.jq(solved.future.results[0], 'map(.metadata.approach) | flatten | join("\n")')

    past_str = If(has_attempts, best_past_attempt, challenge.to_task_description()).future.result

    query_str = If(not_solved, past_str, past_solve_str).future.result
    similar_unsolved = QueryVectorStore(
        collection_name="arc_attempts",
        model="jina-v2",
        query_strings=[query_str],
        top_k=1,
        include_metadata=True,
        include_values=False,
        filters={"task_id": {"$ne": challenge.id}},
    )
    similar_solved = QueryVectorStore(
        collection_name="arc_solves",
        model="jina-v2",
        query_strings=[query_str],
        top_k=1,
        include_metadata=True,
        include_values=False,
        filters={"task_id": {"$ne": challenge.id}},
    )
    summary = If(
        should_summ,
        ComputeText(
            prompt=sb.format(
                "{intro}\n\nBelow are learnings from past failed attempts at solving a particular problem. Summarize these failed attempts, including what seems to be important, what seems to work, what seems to not work, and what to investigate before the next attempt.\n\n<PAST_ATTEMPTS>{past_attempts}</PAST_ATTEMPTS>",
                intro=arc_intro(short=True),
                past_attempts=past_attempts,
            ),
            model=gpt,
            max_tokens=800,
        ).future.text,
        "",
    )
    related = Box(
        value={"unsolved": similar_unsolved.future.results[0][0], "solved": similar_solved.future.results[0][0]}
    )
    res = await substrate.async_run(summary, related)
    solution = res.get(solved).results[0][0] if res.get(solved).results[0] else None
    most_recent_attempt = None
    for att in res.get(prev_attempts).results[0]:
        if att.metadata.get("time"):
            if most_recent_attempt is None:
                most_recent_attempt = att
            elif att.metadata["time"] > most_recent_attempt.metadata["time"]:
                most_recent_attempt = att

    try:
        final = solution, most_recent_attempt, res.get(summary).result, res.get(related).value
    except Exception as e:
        print("Error getting past attempts", e)
        traceback.print_exc()
        final = None, None, None, {}
    return final


async def attempt(challenge: GridProblem, run_remote=False, with_solution=False, verbose=False):
    global attempted, successful
    print(f"Checking past for {challenge.id}")
    prev_solution, recent_attempt, summarize_learnings, related = await get_previous_tries(challenge)
    if prev_solution:
        print("Previous Solution Found")
        # return

    print(f"Attempting {challenge.id}")
    debug_io = {}

    solution_str = f"SOLUTION:\n\n{solutions[challenge.id].result_description()}" if with_solution else ""
    impression_q = get_initial_impression(challenge)
    if related.get("unsolved"):
        impression_q = sb.concat(
            impression_q, "\n\nRelated unsolved task (possibly low signal):", related["unsolved"]["metadata"]["doc"]
        )
    if related.get("solved"):
        impression_q = sb.concat(
            impression_q, "\n\nRelated solved task (maybe relevant):", related["solved"]["metadata"]["doc"]
        )
    if recent_attempt:
        impression_q = sb.concat(
            impression_q,
            "\n\nResults from most recent attempt:",
            recent_attempt.metadata["doc"],
        )
    if summarize_learnings:
        impression_q = sb.concat(impression_q, "\n\nLearnings from past attempts:", summarize_learnings)

    # reason = ComputeText(prompt=impression_q, model=smart_model, temperature=0.25)
    reason = moa(impression_q, max_tokens=1300)

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
        "From the following message, extract structured JSON:\n\n<RESPONSE>",
        make_attempt.future.text,
        "</RESPONSE>\n\n",
    )
    parse_attempt = ComputeJSON(
        prompt=parse_query,
        # json_schema=SolveAttempt.simple_json_schema(),
        # json_schema=SolveAttempt.model_json_schema(),
        json_schema=to_strict_json_schema(SolveAttempt),
        # model="Llama3Instruct70B",
        model=gpt,
        _max_retries=2,
        temperature=0.25,
        max_tokens=4000,
    )
    run_py = None
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
        json_schema=to_strict_json_schema(ComputedResult),
        model=gpt,
        temperature=0.25,
        _max_retries=2,
    )
    debug_io["computed_result"] = {"in": compute_result_query, "out": result.future.json_object}
    input_space = Box(value=debug_io)
    try:
        res = await substrate.async_run(result, input_space)
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
    did_pass = solution.validate(solutions[challenge.id])
    try:
        finish_ns = str(time.time_ns())
        extra_meta = {"with_solution": with_solution, "time": int(time.time())}
        if bool(run_py and res.get(run_py)):
            pytest_out = res.get(run_py).output
            if pytest_out is not None and pytest_out.get("examples") and pytest_out.get("test_cases"):
                if all(pytest_out["examples"]) and all(pytest_out["test_cases"]):
                    pytest_res = "pass"
                elif all([not x for x in pytest_out["examples"]]) and all([not x for x in pytest_out["test_cases"]]):
                    pytest_res = "fail"
                else:
                    pytest_res = "partial"
            else:
                pytest_res = "unknown"
            extra_meta["py_test"] = pytest_res
            extra_meta["py_run_error"] = res.get(run_py).stderr
            extra_meta["py_run_logs"] = res.get(run_py).stdout
        if did_pass:
            successful += 1
            await substrate.async_run(
                EmbedText(
                    text=challenge.to_task_description(),
                    collection_name="arc_solves",
                    metadata={**res.get(parse_attempt).json_object, **extra_meta},
                    doc_id=challenge.id,
                    _max_retry=2,
                )
            )
        else:
            await substrate.async_run(
                EmbedText(
                    text=sb.concat(
                        challenge.to_task_description(),
                        "\n\nComputed:\n",
                        solution.comparison_report(solutions[challenge.id]),
                    ),
                    collection_name="arc_attempts",
                    metadata={**res.get(parse_attempt).json_object, **extra_meta},
                    embedded_metadata_keys=[
                        "concepts_used",
                        "approach",
                        "python_function",
                        "solution",
                        "stdout",
                        "error_message",
                        "computed_result",
                        "py_test",
                        "py_run_error",
                    ],
                    # hide=True,
                    _max_retries=2,
                    doc_id=finish_ns,
                )
            )
        print(f"\n\nWrote to {'solves' if did_pass else 'attempts'}")
        print(f"Solve Rate: {successful} of {attempted} ({successful / attempted:.2%})")
    except Exception as e:
        print("Error embedding solve", e)
        traceback.print_exc()


# def save_research(event: ResearchEvent):
#     res = substrate.run(event)
#     print(json.dumps(res.json, indent=2))
#     return ResearchEvent.model_validate(res.get(event).json_object)


def research_pass(challenge_list: List[GridProblem], prev_event: Optional[ResearchEvent] = None, passes=1):
    if prev_event:
        pretty_fields = f"""Current Total Knowledge: {prev_event.current_total_knowledge}
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
Respond with a single new object with keys: current_total_knowledge, ordered_concept_list, new_knowledge""",
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
        text="Research Logs",
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
        return await attempt(challenge, with_solution=True, run_remote=True)


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
    # id = "3bd67248"
    # id = "1f876c06"
    # challenge: GridProblem = challenges[id]

    # random_challenge = random.choice(all_challenges)
    # await attempt(random_challenge, with_solution=True, verbose=True, run_remote=True)

    # so, rec, su, rel = await get_previous_tries(random_challenge)
    # print("Previous Solution:", so.metadata if so else "None")
    # print("Recent Attempt:", rec.metadata if rec else "None")
    # print("Summary:", su)
    # print("Related:", rel)

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

    for i in range(4):
        await solve_loop()
    # await solve_loop()


if __name__ == "__main__":
    asyncio.run(main())
