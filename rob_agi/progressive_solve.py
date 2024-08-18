from aider.coders import Coder
from aider.models import Model


from rob_agi.arc_util import load_task_set
from rob_agi.test_factory import run_pytest

task_set = "training"
challenges, solutions = load_task_set(task_set_name=task_set)
test_id = "776ffc46"

file_entries = {"main": f"attempts/{test_id}/main.py", "test": f"attempts/{test_id}/test.py"}
files = [f for f in file_entries.values()]


model = Model("claude-3-5-sonnet-20240620")

coder: Coder = Coder.create(main_model=model, fnames=files, auto_commits=False, use_git=False)
current_result = run_pytest(file_entries["test"])

success = current_result["success"]
max_tries = 4
num_tries = 0

print(current_result)
while not success and num_tries < max_tries:
    failure = current_result["error"]
    prompt = "the goal is have a correct function `fib` that returns the nth fibonacci number. currently the tests are failing. please fix the implementation or the tests."
    prompt += f"\n\nstderr:\n\n{failure}"
    coder.run(prompt)
    num_tries += 1
    current_result = run_pytest(files[1])
    success = current_result["success"]
    print(current_result)
