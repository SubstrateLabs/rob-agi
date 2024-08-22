from pathlib import Path

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model


from rob_agi.arc_util import load_task_set
from rob_agi.solver_functions import problem_setup_aider
from rob_agi.test_factory import run_pytest, setup_tests

project_root = Path(__file__).parent.parent
ignore_template = project_root / ".aiderignore"
path = project_root / f"rob_agi/attempts/c_{challenge_id}"
file_entries = {
    "main": path / main_file,
    "test": path / test_file,
    "image": project_root / f"data/task_images/{challenge_id}.png",
}

# challenge_id = "c59eb873"  # easy
challenge_id = "776ffc46"  # hard

task_set = "training"
challenges, solutions = load_task_set(task_set_name=task_set)
main_file = "main.py"
test_file = "test.py"


def get_coder():
    fnames = [file_entries["main"]]
    read_only_fnames = [file_entries["test"]]
    adhoc_ignore = project_root / ".adhoc-aiderignore"
    with open(ignore_template, "r") as tf:
        with open(adhoc_ignore, "w") as f:
            f.write(tf.read())
            f.write("\n")
            f.write(f"!data/task_images/{challenge_id}.png\n")
            f.write(f"!rob_agi/attempts/c_{challenge_id}\n")
    io = InputOutput(chat_history_file=path / ".aider.chat.history.md", llm_history_file=path / ".aider.llm.history.md")
    coder: Coder = Coder.create(
        main_model=Model("claude-3-5-sonnet-20240620"),
        fnames=fnames,
        io=io,
        read_only_fnames=read_only_fnames,
        cache_prompts=True,
        stream=False,
        summarize_from_coder=False,
        edit_format="ask",
        auto_commits=False,
        # max_reflections=3,
        # auto_commits=False, use_git=False
    )
    coder.repo.aider_ignore_file = adhoc_ignore
    # print(coder.repo.aider_ignore_file)
    # print(coder.root)
    # print(coder.get_all_relative_files())
    # print(coder.get_repo_map())
    return coder


max_tries = 1
tries = 0

gp = challenges[challenge_id]
goal = problem_setup_aider(gp)

setup_tests(challenge_id, task_set, path)
current_result = run_pytest(file_entries["test"])
success = current_result["success"]
print(current_result)

while not success and tries < max_tries:
    print(f"Try {tries+1}/{max_tries}")
    if tries == 0:
        prefix = f"{goal}\n\nCurrently the tests are failing. please fix the implementation. the tests never need to be modified."
    else:
        prefix = "The tests are still failing."
    prefix += "\nDiagnose the issue. First think step by step about what is wrong and how to fix it, then come up with the correct solution"
    prompt = f"{prefix}\n\nRESULTS:\n\n{current_result['error']}\n{current_result['output']}"
    prompt += "Document your thinking and approach in the docstring.\n"
    prompt += f"An image of the challenge is provided at {challenge_id}.png"
    coder.run(prompt)
    tries += 1
    current_result = run_pytest(file_entries["test"])
    success = current_result["success"]
    print("SUCCESS: ", success)


def teardown_coder():
    # delete the adhoc ignore file:
    adhoc_ignore.unlink(missing_ok=True)


"""
Process should be:
- think about it, make a plan
- try to implement the plan in code
- run the tests
- if the tests fail
  - diagnose the issue
  - make a plan to fix it
  - implement the plan as diff
  - run the tests
"""
