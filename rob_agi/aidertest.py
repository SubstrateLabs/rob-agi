from pathlib import Path

from aider.coders import Coder
from aider.models import Model

import subprocess
import sys

fnames = ["output/fib/impl.py", "output/fib/test.py"]


def run_pytest(test_file):
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", test_file], capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "returncode": result.returncode,
        }
    except Exception as e:
        return {"success": False, "output": "", "error": str(e), "returncode": -1}


model = Model("claude-3-5-sonnet-20240620")

coder: Coder = Coder.create(
    main_model=model,
    fnames=fnames,
    auto_commits=False,
    use_git=False,
    stream=False,
)
coder.io.chat_history_file = Path("output/fib/.aider.chat.history.md")

current_result = run_pytest(fnames[1])
success = current_result["success"]
max_tries = 4
num_tries = 0
print(current_result)
while not success and num_tries < max_tries:
    failure = current_result["error"]
    prompt = "the goal is have a correct function `fib` that returns the nth fibonacci number. currently the tests are failing. please fix the implementation."
    prompt += f"\n\nstderr:\n\n{failure}"
    coder.run(prompt)
    num_tries += 1
    current_result = run_pytest(fnames[1])
    success = current_result["success"]
    print(current_result)

print("DONE=======================================")
# coder.run("create a new function that returns the nth fibonacci number and name it `fib`")
# coder.run("write a test for the fibonacci function")
# coder.run(f"add a new paragraph element with a fun fact about the animal: {animal}")
# coder.run("make this page ready for presentation. style it nicely, do a general pass at the content.")
# # coder.run("/tokens")
