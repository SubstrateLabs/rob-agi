from pathlib import Path

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model


from rob_agi.arc_util import load_task_set
from rob_agi.solver_functions import problem_setup
from rob_agi.test_factory import run_pytest, setup_tests, get_pytest_error

# challenge_id = "c59eb873" # easy
challenge_id = "776ffc46"  # hard

task_set = "training"
challenges, solutions = load_task_set(task_set_name=task_set)
main_file = "main.py"
test_file = "test.py"

path = Path(f"attempts/c_{challenge_id}")
file_entries = {"main": path / main_file, "test": path / test_file}
files = [f for f in file_entries.values()]

setup_tests(challenge_id, task_set, path)

model = Model("claude-3-5-sonnet-20240620")
io = InputOutput(chat_history_file=path / ".aider.chat.history.md")
coder: Coder = Coder.create(
    main_model=model,
    fnames=files,
    io=io,
    # max_reflections=3,
    # auto_commits=False, use_git=False
)

current_result = run_pytest(file_entries["test"])
success = current_result["success"]
max_tries = 3
tries = 0

gp = challenges[challenge_id]
goal = problem_setup(gp)

print(current_result)

while not success and tries < max_tries:
    prompt = (
        goal
        + "\n\nCurrently the tests are failing. please fix the implementation. the tests never need to be modified."
        + f"\n\n{current_result['error']}\n{current_result['output']}"
    )
    coder.run(prompt)
