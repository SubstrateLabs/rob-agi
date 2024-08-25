import asyncio
import base64
import json
import os
import random
import time
import traceback
import logging
from typing import List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(threadName)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

import cloudpickle
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
from substrate.core.models import RunPythonOut

from rob_agi.arc_util import load_task_set, report_results
from rob_agi.arc_vec import ResearchEvent, SolveAttempt
from rob_agi.colored_grid import ColoredGrid
from rob_agi.computed_result import ComputedResult
from rob_agi.grid_problem import GridProblem
from rob_agi.moa import moa, run_moa
from rob_agi.solver_functions import (
    problem_setup,
    gather_research,
    explain_research,
    arc_intro,
    attempt_challenge,
    run_eval,
)
from rob_agi.progressive_solve import Solver

api_key = os.environ.get("SUBSTRATE_API_KEY")
substrate = Substrate(api_key=api_key, timeout=60 * 4, additional_headers={})

col_attempts = FindOrCreateVectorStore(collection_name="arc_attempts", model="jina-v2")
col_research = FindOrCreateVectorStore(collection_name="arc_research_events", model="jina-v2")
col_solves = FindOrCreateVectorStore(collection_name="arc_solves", model="jina-v2")
col_knowledge = FindOrCreateVectorStore(collection_name="arc_problems", model="jina-v2")

# smart_model = "Llama3Instruct70B"
smart_model = "claude-3-5-sonnet-20240620"
# smart_model = "gpt-4o"
# json_model = "Llama3Instruct8B"
json_model = "Mixtral8x7BInstruct"
gpt = "gpt-4o"

# task_set = "training"
task_set = "evaluation"

challenges, solutions = load_task_set(task_set_name=task_set)
with_solution = task_set == "training"
remote_pip_deps = [
    "pydantic==2.8.2",
    "substrate",
    "git+https://github.com/SubstrateLabs/rob-agi.git@673d3e5",
    "numpy",
]
max_python_tries = 4

all_challenges = list(challenges.values())
random.shuffle(all_challenges)


def ensure_db():
    res = substrate.run(col_attempts, col_research, col_knowledge)
    logger.info(json.dumps(res.json, indent=2))


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
    logger.info(json.dumps(res.json, indent=2))


attempted = 0
successful = 0
errored_count = 0


async def get_all_verified():
    solved = QueryVectorStore(
        collection_name="arc_solves",
        model="jina-v2",
        query_strings=["correct response"],
        top_k=1000,
        include_metadata=True,
        include_values=False,
        filters={"py_test": {"$eq": "pass"}},
    )
    res = await substrate.async_run(solved)
    return res.get(solved).results[0]


async def get_previous_tries(challenge: GridProblem):
    prev_attempts = QueryVectorStore(
        collection_name="arc_attempts",
        model="jina-v2",
        query_strings=["recent correct response"],
        top_k=12,
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
    passes = {"py_test": {"$eq": "pass"}}
    is_not_task = {"task_id": {"$ne": challenge.id}}
    similar_solved = QueryVectorStore(
        collection_name="arc_solves",
        model="jina-v2",
        query_strings=[query_str],
        top_k=1,
        include_metadata=True,
        include_values=False,
        filters={"$and": [passes, is_not_task]},
    )
    summary = If(
        should_summ,
        ComputeText(
            prompt=sb.format(
                "{intro}\n\nBelow are learnings from past failed attempts at solving a particular problem. Summarize these failed attempts, including what seems to be important, what seems to work, what seems to not work, and what to investigate before the next attempt.\n\n<PAST_ATTEMPTS>{past_attempts}</PAST_ATTEMPTS>",
                intro=arc_intro(short=True),
                past_attempts=past_attempts,
            ),
            model=smart_model,
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
    all_prev_attempts = res.get(prev_attempts).results[0]
    for att in all_prev_attempts:
        if att.metadata.get("time"):
            if most_recent_attempt is None:
                most_recent_attempt = att
            elif att.metadata["time"] > most_recent_attempt.metadata["time"]:
                most_recent_attempt = att

    try:
        final = {
            "prev_solution": solution,
            "prev_attempts": all_prev_attempts,
            "recent_attempt": most_recent_attempt,
            "learnings": res.get(summary).result,
            "related": res.get(related).value,
        }
    except Exception as e:
        print("Error getting past attempts", e)
        traceback.print_exc()
        final = {}
    return final


async def get_initial_thoughts(challenge: GridProblem, verbose=False) -> Tuple[str, List[str]]:
    logger.info(f"Checking past for {challenge.id}")
    check_past = await get_previous_tries(challenge)

    prev_solution = check_past.get("prev_solution")
    recent_attempt = check_past.get("recent_attempt")
    summarize_learnings = check_past.get("learnings") or ""
    related = check_past.get("related") or {}
    prev_attempts = check_past.get("prev_attempts") or []

    if verbose:
        print(" > CHECK_PAST\n", check_past)

    impression_q = problem_setup(challenge)

    if prev_solution:
        py_solved = prev_solution.metadata.get("py_test")
        solve_time = prev_solution.metadata.get("time")
        solve_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(solve_time)) if solve_time else "Unknown"
        print(f"Previous Solution Found[{solve_time_str}]:", py_solved)
        if py_solved == "fail" or py_solved == "partial":
            impression_q = sb.concat(
                impression_q,
                "\n\nYour previous solutions were able to produce the right answer to the test, but the python function did not work generally. Please consider this in your approach.",
                f"<PREVIOUS_SUBMISSION>{prev_solution.metadata.get('python_function')}\n\n{prev_solution.metadata.get('py_run_error')}\n\n{prev_solution.metadata.get('py_run_logs')}</PREVIOUS_SUBMISSION>",
            )
            if len(prev_attempts) > 2:
                impression_q = sb.concat(
                    impression_q,
                    f"Note that there have been {len(prev_attempts)} previous attempts at this problem, indicating that it is a difficult one. You may need to think more abstractly here to solve it. What would a person see if they were looking at this colored grid? How would they most easily understand it?",
                )
        elif py_solved == "pass":
            impression_q = sb.concat(
                impression_q,
                "\n\nYour previous solutions were able to produce the right answer to the test, and the python function worked generally. Don't change anything about your solution unless it improves it without breaking the test cases.",
                f"<PREVIOUS_SUBMISSION>{prev_solution.metadata.get('python_function')}\n\n{prev_solution.metadata.get('py_run_error')}\n\n{prev_solution.metadata.get('py_run_logs')}</PREVIOUS_SUBMISSION>",
            )
    # if related.get("unsolved"):
    #     impression_q = sb.concat(impression_s, "\n\nOne similar unsolved task:", related["unsolved"]["metadata"]["doc"])
    if related.get("solved"):
        impression_q = sb.concat(
            impression_q, "\n\nExample solution from a different problem:", related["solved"]["metadata"]["doc"]
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
    reason = moa(impression_q, max_tokens=1800, num_layers=1)

    if with_solution:
        solution_str = f"<SOLUTION>\n{solutions[challenge.id].result_description()}</SOLUTION>"
        think_with_solution = sb.format(
            "<PAST_THINKING>{impression_q}</PAST_THINKING>\nGiven those past attempts your recent thinking is:<CURRENT_THINKING>\n\n{impression}\n</CURRENT_THINKING>\nNow, here is the correct solution:\n\n{solution_str}\n\nCome up with an approach that incorporates what you learned. The important part here is to identify the key steps that are necessary to solve this problem. Be comprehensive, detailed, but also concise.",
            impression_q=impression_q,
            impression=reason.future.value.text,
            solution_str=solution_str,
        )
        solution_ct = ComputeText(prompt=think_with_solution, model=smart_model, temperature=0.2)
    else:
        think = sb.format(
            "<PAST_THINKING>{impression_q}</PAST_THINKING>\nGiven those past attempts your recent thinking is:<CURRENT_THINKING>\n\n{impression}\n</CURRENT_THINKING>\n\nCome up with an approach that incorporates what you learned. The important part here is to identify the key steps that are necessary to solve this problem. Be comprehensive, detailed, but also concise.",
            impression_q=impression_q,
            impression=reason.future.value.text,
        )
        solution_ct = ComputeText(prompt=think, model=gpt, temperature=0.2)

    res = await substrate.async_run(solution_ct)
    impression = res.get(solution_ct).text
    moa_res = res.get(reason)
    function_candidates = find_all_fns(moa_res.value)
    additional_fn = parse_python_fn_str(impression)
    if additional_fn and additional_fn not in function_candidates:
        function_candidates.append(additional_fn)
    return impression, function_candidates


async def first_attempt(challenge: GridProblem, initial_thoughts: str) -> dict:
    logger.info(f"Attempting {challenge.id}")
    prompt = attempt_challenge(challenge, reasoning=initial_thoughts)
    # ct_try = ComputeText(prompt=prompt, model=smart_model, temperature=0.2, max_tokens=2400)
    # return res.get(ct_try).text
    moa_try = moa(prompt, num_layers=2)
    res = await substrate.async_run(moa_try)
    return res.get(moa_try).value


async def parse_attempt(challenge: GridProblem, first_answer: str) -> SolveAttempt:
    logger.info(f"Parsing {challenge.id}")
    parse_query = sb.concat(
        "From the following message, extract a result as structured JSON:\n\n<MESSAGE>",
        first_answer,
        "</MESSAGE>\n",
        f"In this case there should be {len(challenge.test_cases)} solutions. each solution is a grid, and a grid is a list of lists of ints.",
    )
    parsed = ComputeJSON(
        prompt=parse_query,
        # json_schema=SolveAttempt.simple_json_schema(),
        # json_schema=SolveAttempt.model_json_schema(),
        json_schema=to_strict_json_schema(SolveAttempt),
        # model="Llama3Instruct70B",
        model=gpt,
        _max_retries=2,
        max_tokens=3600,
    )
    res = await substrate.async_run(parsed)
    return SolveAttempt.parse_obj(res.get(parsed).json_object)


def parse_python_fn_str(llm_response: str):
    # find the stuff inside the python code fence:
    parsed_python_fn = None
    if "```python" in llm_response:
        start = llm_response.split("```python")[1]
        if "```" in start:
            parsed_python_fn = start.split("```")[0]
    return parsed_python_fn


def score_output(py_out: RunPythonOut, challenge: GridProblem):
    output_dict = py_out.output
    if not output_dict or not output_dict.get("example_solutions"):
        return -1
    ex_solutions = output_dict["example_solutions"]
    score = 0
    try:
        for sol_idx, grid in enumerate(ex_solutions):
            expected = challenge.examples[sol_idx].output.values
            for row_idx, row in enumerate(grid):
                for col_idx, cell in enumerate(row):
                    if cell == expected[row_idx][col_idx]:
                        score += 1
        if with_solution:
            test_sols = output_dict.get("solutions") or []
            for sol_idx, grid in enumerate(test_sols):
                expected = challenge.test_cases[sol_idx].values
                for row_idx, row in enumerate(grid):
                    for col_idx, cell in enumerate(row):
                        if cell == expected[row_idx][col_idx]:
                            score += 1
    except Exception as e:
        print("Error scoring output", e, py_out, challenge, traceback.format_exc())
        return -1
    return score


async def run_py_fn(
    challenge: GridProblem,
    parsed: SolveAttempt,
    functions: List[str],
    max_tries: int = max_python_tries,
    verbose=False,
) -> Optional[RunPythonOut]:
    async def _run(fn: str, run_label: str) -> RunPythonOut:
        if verbose:
            logger.info(f"Exec Py[{run_label}]: {challenge.id}")
        py_args = {"id": challenge.id, "fn_code": fn, "task_set": task_set, "with_solution": with_solution}
        run_py = RunPython(function=run_eval, kwargs=py_args, pip_install=remote_pip_deps)
        res = await substrate.async_run(run_py)
        out = res.get(run_py)
        if verbose:
            print(f" > PY_OUT[{run_label}]", out)
        return out

    async def _run_all(fns: List[str], run_count: int) -> List[Union[RunPythonOut, None]]:
        if verbose:
            logger.info(f" > BATCH {len(fns)} FNS: {challenge.id}")
        tasks = [_run(fn, f"{run_count}.{idx}") for idx, fn in enumerate(fns)]
        all_res = await asyncio.gather(*tasks, return_exceptions=True)
        ret = []
        for res in all_res:
            if isinstance(res, Exception):
                print("Error running python function:", res)
                traceback.print_exception(type(res), res, res.__traceback__)
                ret.append(None)
            else:
                if res.output is None and res.pkl_output:
                    res.output = cloudpickle.loads(base64.b64decode(res.pkl_output))
                ret.append(res)
        return ret

    def set_results(rpo: RunPythonOut, fn):
        parsed.stdout = rpo.stdout
        parsed.error_message = rpo.stderr
        rp_out = rpo.output
        if rp_out and rp_out.get("solutions"):
            parsed.solutions = rp_out.get("solutions")
        if fn:
            parsed.python_function = fn

    approach_list = "\n".join(parsed.approach)
    curr_best = None
    to_try = functions

    for i in range(max_tries):
        if not to_try:
            break
        try:
            all_opt_results = await _run_all(to_try, i)
            resolved_results = [r for r in all_opt_results if r]
            if not resolved_results:
                continue

            curr_best = resolved_results[0] if not curr_best else curr_best
            for ri, sample_out in enumerate(all_opt_results):
                if not sample_out:
                    continue
                sample_res = sample_out.output
                s_examples = sample_res.get("examples") if sample_res else None
                s_test_cases = sample_res.get("test_cases") if sample_res else None
                print(f"Results [{i}.{ri}]: {challenge.id}", s_examples, s_test_cases)
                has_res = s_examples and s_test_cases
                if has_res and all(s_examples) and (not with_solution or all(s_test_cases)):
                    set_results(sample_out, to_try[ri])
                    return sample_out

                sample_correct = score_output(sample_out, challenge)
                best_count = score_output(curr_best, challenge)
                if sample_correct > best_count:
                    set_results(sample_out, to_try[ri])
                    curr_best = sample_out

            results = curr_best.output

            examples = results.get("examples") if results else None
            test_cases = results.get("test_cases") if results else None

            reflection = f"The general approach was:\n\n{approach_list}\n\nBut the solution did not pass. We need to fix the function and try again."
            if parsed.python_function:
                reflection += f"The function that failed:\n\n```python\n{parsed.python_function}\n```"
            if examples:
                reflection += f"Example input results: {['Pass' if e else 'Fail' for e in examples]}\n"
            if test_cases and with_solution:
                reflection += f"Test case input results: {['Pass' if e else 'Fail' for e in test_cases]}\n"
            reflection += f"Error message: {curr_best.stderr or 'None'}\n"
            reflection += f"Stdout: {curr_best.stdout or 'None'}\n"
            pytest_results = get_py_test(results)

            example_rollup = pytest_results.get("examples")
            test_case_rollup = pytest_results.get("test_cases")
            examples_explanation = ""

            if example_rollup == "fail":
                examples_explanation = "The example inputs all failed to produce the correct output."
            elif example_rollup == "partial":
                examples_explanation = "The example inputs produced a mix of correct and incorrect outputs."
            elif example_rollup == "pass":
                examples_explanation = "The example inputs all produced the correct output."
            test_explanation = ""
            if test_cases:
                test_case_noun = "cases" if len(test_cases) > 1 else "case"
                all_str = " all" if len(test_cases) > 1 else ""
                if test_case_rollup == "fail":
                    test_explanation = f"The test {test_case_noun}{all_str} failed to produce the correct output."
                elif test_case_rollup == "partial":
                    test_explanation = f"The test {test_case_noun} produced a mix of correct and incorrect outputs."
                elif test_case_rollup == "pass":
                    test_explanation = f"The test {test_case_noun}{all_str} produced the correct output."

            reflection += f"{examples_explanation}\n{test_explanation}"

            diagnose = ComputeText(
                prompt="Below is a candidate solution to an ARC challenge problem that tests fundamental reasoning skills.\n"
                + reflection
                + "\n\nDiagnose the issue with the attempt, explaining what went wrong and what needs to be fixed. Your diagnosis should be short, specific, and comprehensive.",
                model=gpt,
                max_tokens=700,
            )

            past_reason = sb.concat(reflection, "\n\nIssue diagnosis:\n\n", diagnose.future.text)
            prompt = attempt_challenge(challenge, reasoning=past_reason)

            # new_attempt = ComputeText(prompt=prompt, model=smart_model, max_tokens=1900)
            new_attempt_moa = await run_moa(prompt, max_tokens=4000, num_layers=3, filename_prefix=challenge.id)
            if verbose:
                logger.info(" > NEW_ATTEMPT\n")
            new_fns = find_all_fns(new_attempt_moa)
            to_try = new_fns

        except Exception as e:
            print(f"Error running python function on attempt {i}", e)
            traceback.print_exc()
    return curr_best


def find_all_fns(moa_response: dict) -> List[str]:
    functions = []
    new_fn = parse_python_fn_str(moa_response["text"])
    if new_fn:
        functions.append(new_fn)
    try:
        layers = moa_response["layers"]
        for layer in reversed(layers):
            for candidate in layer:
                new_fn = parse_python_fn_str(candidate)
                if new_fn and new_fn not in functions:
                    functions.append(new_fn)
    except Exception as e:
        print("Error finding new fn", e, moa_response)
        traceback.print_exc()
    return functions


def get_py_test(results: Optional[dict]):
    if not results:
        return {"examples": "unknown", "test_cases": "unknown"}
    examples = results.get("examples") if results else None
    test_cases = results.get("test_cases") if results else None
    all_examples_pass = all(examples)
    all_test_cases_pass = all(test_cases)
    all_examples_fail = all([not x for x in examples])
    all_test_cases_fail = all([not x for x in test_cases])

    example_status = "pass" if all_examples_pass else "fail" if all_examples_fail else "partial"
    test_status = "pass" if all_test_cases_pass else "fail" if all_test_cases_fail else "partial"
    if not examples:
        example_status = "unknown"
    if not test_cases:
        test_status = "unknown"

    return {"examples": example_status, "test_cases": test_status}


async def log_result(challenge: GridProblem, parsed: SolveAttempt, run_py: Optional[RunPythonOut], extra_meta: dict):
    global attempted, successful
    submission = ComputedResult(
        task_id=challenge.id, outputs=[ColoredGrid(values=s) for s in parsed.solutions if s is not None]
    )
    persisted_comparison = ""
    try:
        _comparison = submission.comparison_report(solutions[challenge.id])
        persisted_comparison = _comparison if with_solution else ""
        logger.info(_comparison)
        did_pass = submission.validate(solutions[challenge.id])
    except Exception as e:
        print(f"Error comparing results: {challenge.id}", e)
        print(
            "=============================\n",
            challenge.id,
            submission.outputs,
            solutions[challenge.id].outputs,
            "=============================",
        )
        traceback.print_exc()
        did_pass = False

    finish_ns = str(time.time_ns())

    if bool(run_py):
        gathered_py_test = get_py_test(run_py.output)
        examples = gathered_py_test.get("examples")
        test_cases = gathered_py_test.get("test_cases")
        if examples == "pass" and (test_cases == "pass" or not with_solution):
            pytest_rollup = "pass"
        elif examples == "fail" and test_cases == "fail":
            pytest_rollup = "fail"
        elif examples == "partial" or test_cases == "partial":
            pytest_rollup = "partial"
        else:
            pytest_rollup = "unknown"
    else:
        pytest_rollup = "unknown"

    extra_meta["py_test"] = pytest_rollup
    extra_meta["py_run_error"] = run_py.stderr if run_py else parsed.error_message
    extra_meta["py_run_logs"] = run_py.stdout if run_py else parsed.stdout
    write_nodes = []
    if did_pass:
        successful += 1
        # NB for now only save to solves if we are "training"
        if with_solution:
            write_nodes.append(
                EmbedText(
                    text=challenge.to_task_description(),
                    collection_name="arc_solves",
                    metadata={**parsed.model_dump(), **extra_meta},
                    embedded_metadata_keys=["approach", "python_function", "py_test"],
                    doc_id=challenge.id,
                    _max_retry=2,
                )
            )
    # todo - diff string in emb
    # diff_string = solution.outputs
    write_nodes.append(
        EmbedText(
            text=sb.concat(
                challenge.to_task_description(),
                f"\n\nComputed:\n{persisted_comparison}" if persisted_comparison and with_solution else "",
            ),
            collection_name="arc_attempts",
            metadata={**parsed.model_dump(), **extra_meta},
            embedded_metadata_keys=[
                "approach",
                "python_function",
                "solution",
                "stdout",
                "error_message",
                "computed_result",
                "py_test",
                "py_run_error",
            ],
            _max_retries=2,
            doc_id=finish_ns,
        )
    )
    await substrate.async_run(*write_nodes)

    logger.info(f"\n\nWrote to {'solves' if did_pass else 'attempts'}")
    logger.info(f"Solve Rate: {successful} of {attempted} ({successful / attempted:.2%})")


async def attempt_old(challenge: GridProblem, run_remote=False, verbose=False):
    global attempted, successful, errored_count
    attempted += 1

    logger.info(f"Starting {challenge.id}")
    initial_thoughts, starting_functions = await get_initial_thoughts(challenge, verbose=verbose)
    if verbose:
        print(" > INITIAL_THOUGHTS\n", initial_thoughts)

    first_answer = await first_attempt(challenge, initial_thoughts)
    if verbose:
        print(" > FIRST_ATTEMPT\n", first_answer)

    to_try = list(set(find_all_fns(first_answer) + starting_functions))
    parsed = await parse_attempt(challenge, first_answer["text"])
    if verbose:
        print(" > PARSED\n", parsed.model_dump())

    if parsed.python_function and parsed.python_function not in to_try:
        to_try.append(parsed.python_function)

    run_py = (
        await run_py_fn(
            challenge,
            functions=to_try,
            parsed=parsed,
            verbose=verbose,
            max_tries=max_python_tries,
        )
        if run_remote
        else None
    )
    extra_meta = {"with_solution": with_solution, "time": int(time.time())}
    await log_result(challenge, parsed, run_py, extra_meta=extra_meta)


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
        logger.info(json.dumps(res.json, indent=2))
        print("\nWrote EMB", res.get(embed).embedding.doc_id)
        prev_event = ResearchEvent.model_validate(res.get(evt).json_object)
        logger.info(prev_event.current_total_knowledge)


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
    logger.info(json.dumps(res.json, indent=2))


def attempt(challenge: GridProblem, previous_solution: Optional[str] = None):
    global attempted, successful, errored_count
    logger.info(f"Starting {challenge.id}")
    attempted += 1
    sln = solutions[challenge.id]
    s = Solver(challenge=challenge, solution=sln)
    succeeded = s.run_solve(max_tries=4, prev_solution=previous_solution)
    if succeeded:
        successful += 1
    logger.info(f"Solve Rate: {successful} of {attempted} ({successful / attempted:.2%})")


async def aprocess_challenge(semaphore, challenge):
    async with semaphore:
        return await attempt(challenge)


async def process_challenge(executor, challenge):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, attempt, challenge)


async def solve_loop(max_concurrent=1, to_process=None, max_challenges=None):
    global errored_count
    to_process = all_challenges if not to_process else to_process
    if max_challenges:
        to_process = to_process[:max_challenges]

    logger.info(f"Processing {len(to_process)} challenges with {max_concurrent} concurrent threads")

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        tasks = [process_challenge(executor, challenge) for challenge in to_process]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            t0 = time.perf_counter()
            try:
                await task
                logger.info(f"Finished {i} of {len(to_process)} [{time.perf_counter() - t0:.2f}s]")
            except Exception as e:
                errored_count += 1
                traceback.print_exc()
                logger.info(f"Error on task {i}: {e}")

    report_results(attempted=attempted, successful=successful, errored=errored_count)


async def asolve_loop(max_concurrent=1, to_process=None, max_challenges=None):
    global errored_count
    semaphore = asyncio.Semaphore(max_concurrent)
    to_process = all_challenges if not to_process else to_process
    if max_challenges:
        to_process = to_process[:max_challenges]
    tasks = [process_challenge(semaphore, challenge) for challenge in to_process]
    logger.info(f"Processing {len(tasks)} challenges with {max_concurrent} concurrent")
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        t0 = time.perf_counter()
        try:
            await task
            logger.info(f"Finished {i} of {len(to_process)} [{time.perf_counter() - t0:.2f}s]")
        except Exception as e:
            errored_count += 1
            traceback.print_exc()
            logger.info(f"Error on task {i}: {e}")

    report_results(attempted=attempted, successful=successful, errored=errored_count)


async def bootstrap_solved():
    train_challenges, train_solutions = load_task_set(task_set_name="training")
    eval_challenges, eval_solutions = load_task_set(task_set_name="evaluation")
    combined_challenges = {**train_challenges, **eval_challenges}
    combined_solutions = {**train_solutions, **eval_solutions}
    verified_so_far = await get_all_verified()
    # verified_by_id = {v.metadata["task_id"]: v for v in verified_so_far}
    # verified_so_far = [verified_by_id["8eb1be9a"]]
    for v in verified_so_far:
        approach = "Approach:\n\n" + "\n".join([" - " + a for a in v.metadata["approach"]])
        py_fn = v.metadata["python_function"]
        c: GridProblem = combined_challenges.get(v.metadata["task_id"])
        if not c:
            logger.info("Challenge not found:", v.metadata["task_id"])
            continue
        previous_solution = approach + "\n\nPython Function:\n" + py_fn
        attempt(c, prev_solution=previous_solution)


async def main():
    # ensure_db()
    # id = "1f876c06"
    # challenge: GridProblem = challenges[id]
    # random_challenge = challenges["e69241bd"]
    # verified_ids = [v.id for v in verified_so_far]
    # print("Skipping previously solved:", len(verified_ids))

    # to_process = [c for c in all_challenges if c.id not in verified_ids]
    # random_challenge = random.choice(all_challenges)
    # await attempt(random_challenge, verbose=True, run_remote=True)

    # await bootstrap_solved()

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

    for i in range(1):
        await solve_loop(max_concurrent=8, to_process=None, max_challenges=8)
    # await solve_loop(max_concurrent=4, max_challenges=8)


if __name__ == "__main__":
    asyncio.run(main())
