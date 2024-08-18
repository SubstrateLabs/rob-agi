from pathlib import Path

from aider.coders import Coder
from aider.models import Model


from rob_agi.arc_util import load_task_set
from rob_agi.solver_functions import problem_setup
from rob_agi.test_factory import run_pytest, setup_tests

challenge_id = "c59eb873"

task_set = "training"
challenges, solutions = load_task_set(task_set_name=task_set)
main_file = "main.py"
test_file = "test.py"

path = Path(f"attempts/c_{challenge_id}")
file_entries = {"main": path / main_file, "test": path / test_file}
files = [f for f in file_entries.values()]

setup_tests(challenge_id, task_set, path)

model = Model("claude-3-5-sonnet-20240620")
coder: Coder = Coder.create(main_model=model, fnames=files)  # , auto_commits=False, use_git=False)
current_result = run_pytest(file_entries["test"])
success = current_result["success"]
max_tries = 3
try_count = 0

gp = challenges[challenge_id]
goal = problem_setup(gp)

print(current_result)
while not success and try_count < max_tries:
    print(f"TRY {try_count}")
    failure = current_result["error"]
    prompt = (
        goal + "currently the tests are failing. please fix the implementation. the tests never need to be modified."
    )
    prompt += f"\n\nstderr:\n\n{failure}"
    coder.run(prompt)
    try_count += 1
    current_result = run_pytest(files[1])
    success = current_result["success"]
    print(current_result)
