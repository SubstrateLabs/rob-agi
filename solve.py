import base64
import json
import os
import time
from typing import List, Literal, Optional

from substrate import Substrate, ComputeText, ComputeJSON, sb, FindOrCreateVectorStore, EmbedText, QueryVectorStore
from arc_util import task_sets, load_task_set
from arc_vec import ResearchEvent, SolveAttempt
from colored_grid import ColoredGrid
from computed_result import ComputedResult
from grid_problem import GridProblem
from solver_functions import (
    get_initial_impression,
    extract_result,
    gather_research,
    explain_research,
    arc_intro,
    attempt_challenge,
)

api_key = os.environ.get("SUBSTRATE_API_KEY")
substrate = Substrate(
    api_key=api_key, timeout=60 * 5, additional_headers={"x-substrate-fp": "1", "x-substrate-debug": "1"}
)

# Stores:
# - Attempts
# - summary
# - try number
# - Research Event Chain
# - id is probably just timestamp
# - has a previous id
# - keeps track of our current understanding of the entire challenge set
# - Latest Knowledge for each problem
#     - Solution (if solved)
#     - Summary of our latest guess
#     - Things we know about the problem
#     - Things we think are important
#     - Things we've tried
#     - Things that don't work
#     - Things that do work
# For each problem, present what we know about it so far
# - LatestKnowledge.summary

col_attempts = FindOrCreateVectorStore(collection_name="arc_attempts", model="jina-v2")
col_research = FindOrCreateVectorStore(collection_name="arc_research_events", model="jina-v2")
col_knowledge = FindOrCreateVectorStore(collection_name="arc_problems", model="jina-v2")

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
# smart_model: ModelType = "Llama3Instruct70B"
smart_model: ModelType = "claude-3-5-sonnet-20240620"
json_model: ModelType = "Llama3Instruct8B"

challenges, solutions = load_task_set(task_set_name=task_sets["training"])


def ensure_db():
    res = substrate.run(col_attempts, col_research, col_knowledge)
    print(json.dumps(res.json, indent=2))


def local_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        data = image_file.read()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


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


def attempt(challenge: GridProblem):
    if len(challenge.test_cases) > 1:
        final_task = f"Respond in JSON with the solutions for the {len(challenge.test_cases)} test cases in this task."
    else:
        final_task = "\n\nRespond in JSON with the solution for the test case in this task."
    reason = ComputeText(prompt=get_initial_impression(challenge), model=smart_model, temperature=0.2)
    make_attempt = ComputeText(
        prompt=attempt_challenge(
            challenge, sb.concat(reason.future.text, "Schema:\n\n", json.dumps(SolveAttempt.model_json_schema()))
        ),
        model=smart_model,
        temperature=0.2,
    )
    parse_attempt = ComputeJSON(
        prompt=sb.format("Parse this to valid JSON:\n\n{json_str}", json_str=make_attempt.future.text),
        json_schema=SolveAttempt.model_json_schema(),
        model=json_model,
        _max_retries=2,
    )
    result = ComputeJSON(
        prompt=sb.concat(extract_result(challenge), sb.jq(parse_attempt.future.json_object, "@json"), final_task),
        json_schema=ComputedResult.json_schema(max_outputs=len(challenge.test_cases)),
        model=json_model,
        _max_retries=2,
    )
    res = substrate.run(result)
    print(json.dumps(res.json, indent=2))
    solution = ComputedResult(outputs=[ColoredGrid(**j) for j in res.get(result).json_object["outputs"]])
    print(solution.comparison_report(solutions[challenge.id]))


# def save_research(event: ResearchEvent):
#     res = substrate.run(event)
#     print(json.dumps(res.json, indent=2))
#     return ResearchEvent.model_validate(res.get(event).json_object)


def research_pass(challenge_list: List[GridProblem], prev_event: Optional[ResearchEvent] = None, passes=1):
    prev_str = json.dumps(prev_event.model_dump()) + explain_research() if prev_event is not None else None
    think = ComputeText(
        prompt=gather_research(challenge_list, previous_research=prev_str), model=smart_model, temperature=0.2
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
    all_challenges = list(challenges.values())
    chunk_size = 5
    time_ns = None
    for i in range(75, len(all_challenges), chunk_size):
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
        print("\nWrote EMB", res.get(embed).embedding.doc_id)
        prev_event = ResearchEvent.model_validate(res.get(evt).json_object)
        print(prev_event.current_total_knowledge)


def distill_research():
    researches = QueryVectorStore(
        collection_name="arc_research_events",
        query_strings=[
            arc_intro(short=True)
            + "We need to find and distill all knowledge learned so far about solving these challenges"
        ],
        top_k=12,
        include_metadata=True,
        include_values=False,
        model="jina-v2",
        # filters={"doc_id": {"$eq": "1722753019873457000"}},
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
            # use jq to extract map to metadatas and stringify:
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
    embed = EmbedText(
        text=f"Research Logs",
        collection_name="arc_research_events",
        metadata=parse_json.future.json_object,
        embedded_metadata_keys=["current_total_knowledge", "new_knowledge"],
        doc_id=str(time_ns),
        _max_retries=2,
    )
    res = substrate.run(embed)
    print(json.dumps(res.json, indent=2))


def main():
    # ensure_db()
    # id = "0520fde7"
    # id = "3bd67248"
    id = "1f876c06"
    challenge: GridProblem = challenges[id]
    attempt(challenge)

    # ids = list(challenges.keys())[0:5]
    # challenge_list = [challenges[id] for id in ids]
    # tail = research_pass(challenge_list)
    # res = substrate.run(tail)
    # print(json.dumps(res.json, indent=2))
    # return ResearchEvent.model_validate(res.get(tail).json_object)

    # visual_parse(challenge)
    # last = ResearchEvent()

    # research_loop(prev_event=last)
    # distill_research()


if __name__ == "__main__":
    main()
