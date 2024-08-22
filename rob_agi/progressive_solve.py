from pathlib import Path
from typing import Optional

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model


from rob_agi.arc_util import load_task_set
from rob_agi.computed_result import ComputedResult
from rob_agi.grid_problem import GridProblem
from rob_agi.solver_functions import problem_setup_aider
from rob_agi.test_factory import run_pytest, setup_tests


project_root = Path(__file__).parent.parent
ignore_template = project_root / ".aiderignore"

main_file = "main.py"
test_file = "test.py"


class Solver:
    def __init__(self, challenge: GridProblem, solution: Optional[ComputedResult]):
        self.challenge_id = challenge.id
        self.challenge = challenge
        self.solution = solution
        self.challenge_root = project_root / f"rob_agi/attempts/c_{challenge.id}"
        self.file_entries = {
            "main": self.challenge_root / main_file,
            "test": self.challenge_root / test_file,
            "image": project_root / f"data/task_images/{challenge.id}.png",
        }
        self.adhoc_ignore = project_root / f".adhoc-aiderignore-{challenge.id}"
        self.goal = problem_setup_aider(challenge)
        self.setup()

    def setup(self):
        self.challenge_root.mkdir(parents=True, exist_ok=True)
        with open(ignore_template, "r") as tf:
            with open(self.adhoc_ignore, "w") as f:
                f.write(tf.read())
                f.write("\n")
                f.write(f"!data/task_images/{self.challenge.id}.png\n")
                f.write(f"!rob_agi/attempts/c_{self.challenge.id}\n")
        setup_tests(self.challenge, self.solution, self.challenge_root)

    def get_coder(self, **kwargs):
        io = InputOutput(
            chat_history_file=self.challenge_root / ".aider.chat.history.md",
            llm_history_file=self.challenge_root / ".aider.llm.history.md",
        )
        coder: Coder = Coder.create(
            main_model=Model("claude-3-5-sonnet-20240620"),
            io=io,
            cache_prompts=True,
            stream=False,
            auto_commits=False,
            **kwargs,
            # max_reflections=3,
            # auto_commits=False, use_git=False
        )
        coder.repo.aider_ignore_file = self.adhoc_ignore
        # print(coder.repo.aider_ignore_file)
        # print(coder.root)
        # print(coder.get_all_relative_files())
        # print(coder.get_repo_map())
        return coder

    def get_ask_coder(self):
        fnames = [self.file_entries["main"], self.file_entries["test"], self.file_entries["image"]]
        return self.get_coder(edit_format="ask", summarize_from_coder=False, fnames=fnames)

    def get_modify_coder(self):
        fnames = [self.file_entries["main"], self.file_entries["image"], project_root / "rob_agi/colored_grid.py"]
        read_only_fnames = [self.file_entries["test"]]
        return self.get_coder(fnames=fnames, read_only_fnames=read_only_fnames)

    def run_tests(self):
        result = run_pytest(self.file_entries["test"])
        print(result)
        return result

    def run_solve(self):
        max_tries = 1
        tries = 0
        current_result = self.run_tests()

        ask_coder = self.get_ask_coder()
        modify_coder = self.get_modify_coder()

        while not current_result["success"] and tries < max_tries:
            print(f"Try {tries+1}/{max_tries}")
            if tries == 0:
                prefix = f"{self.goal}\n\nCurrently the tests are failing. please fix the implementation. the tests never need to be modified."
            else:
                prefix = "The tests are still failing."
            prefix += "\nDiagnose the issue. First think step by step about what is wrong and how to fix it, then come up with the correct solution"
            prompt = f"{prefix}\n\nRESULTS:\n\n{current_result['error']}\n{current_result['output']}"
            prompt += "Document your thinking and approach in the docstring.\n"
            # prompt += f"An image of the challenge is provided at {self.challenge.id}.png"
            prompt += (
                f"colored_grid.py includes a library of functions that may be useful. modify this file if you need."
            )
            res = modify_coder.run(prompt)
            print("res==================================", res)
            tries += 1
            current_result = self.run_tests()
            print("SUCCESS: ", current_result["success"])

    def __del__(self):
        self.teardown()

    def teardown(self):
        # delete the adhoc ignore file:
        self.adhoc_ignore.unlink(missing_ok=True)


if __name__ == "__main__":
    # challenge_id = "c59eb873"  # easy
    challenge_id = "776ffc46"  # hard
    task_set = "training"
    challenges, solutions = load_task_set(task_set_name=task_set)
    challenge = challenges[challenge_id]
    solution = solutions.get(challenge_id)
    solver = Solver(challenge, solution)
    solver.run_solve()

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
