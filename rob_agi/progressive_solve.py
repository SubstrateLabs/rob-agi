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
        self.setup()
        self.goal = problem_setup_aider(challenge)

    def setup(self):
        self.challenge_root.mkdir(parents=True, exist_ok=True)
        with open(ignore_template, "r") as tf:
            with open(self.adhoc_ignore, "w") as f:
                f.write(tf.read())
                f.write("\n")
                # f.write(f"!data/task_images/{self.challenge.id}.png\n")
                f.write(f"!rob_agi/attempts/c_{self.challenge.id}\n")
        setup_tests(self.challenge, self.solution, self.challenge_root)
        try:
            (self.challenge_root / f"image.png").symlink_to(self.file_entries["image"])
        except FileExistsError:
            pass

    def get_coder(self, **kwargs):
        ef = ""
        if "edit_format" in kwargs:
            ef = "-" + kwargs["edit_format"]

        io = InputOutput(
            chat_history_file=self.challenge_root / f".aider.chat.history{ef}.md",
            llm_history_file=self.challenge_root / f".aider.llm.history{ef}.md",
        )
        # io.yes = True
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
        fnames = [self.file_entries["main"], self.file_entries["test"]]
        # fnames = [self.file_entries["main"], self.file_entries["test"]]
        return self.get_coder(edit_format="ask", fnames=fnames)

    def get_modify_coder(self):
        fnames = [self.file_entries["main"]]
        read_only_fnames = [self.file_entries["test"], project_root / "rob_agi/colored_grid.py"]
        return self.get_coder(fnames=fnames, read_only_fnames=read_only_fnames)

    def run_tests(self):
        result = run_pytest(self.file_entries["test"])
        print(result)
        return result

    def get_prefix(self, is_first):
        if is_first:
            prefix = f"{self.goal}\n\nCurrently the tests used to validate the solution are failing."
        else:
            prefix = "The tests used to validate the solution are still failing."
        return prefix

    def get_plan(self, ask_coder, current_result, is_first=False):
        prefix = self.get_prefix(is_first)
        prompt = f"{prefix}\n\n<VALIDATION_OUTPUT>\n{current_result['error']}\n{current_result['output']}</VALIDATION_OUTPUT>\n"
        prompt += "Examine all the information you have, state your understanding of the challenge, and propose a detailed solution to the challenge in words. Any solution must always apply to every case, not just the failing exception here.\n"
        prompt += "Then reflect on your idea. Look very closely and notice if there are any other patterns or discrepancies worth noting. Remember this is about identifying abstract, intuitive ideas about what is happening.\n"
        prompt += f"Explicitly consider how your idea applies to each of the examples and test cases in attempts/{self.challenge_id}/test.py. To check your thinking, illustrate how your idea either works or doesn't for each case.\n"
        prompt += f"If the rule(s) you came up with does not apply to any specific case, call it out and think about a more general idea that does apply in every case. Be meticulous and careful in your reflection. Sometimes you need to zoom out to see how a single idea can apply to all cases.\n"
        ask_coder.run(prompt)

        res = ask_coder.run("Based on that reflection detail a step by step plan for how to solve the challenge\n")
        return res

    def get_edit(self, modify_coder, current_result, plan, is_first=False):
        prefix = self.get_prefix(is_first)
        prompt = f"{prefix}\n\n<VALIDATION_OUTPUT>\n{current_result['error']}\n{current_result['output']}</VALIDATION_OUTPUT>\n"
        prompt += f"\nYour latest thinking is:\n<LATEST_THINKING>\n{plan}\n</LATEST_THINKING>\n"
        prompt += f"Use that latest thinking and solve the challenge by modifying the implementation file. Always ensure that the docstring to solve_{self.challenge_id} includes a correct summary of the solution in words.\n"
        # prompt += f"An image of the challenge is provided at {self.challenge.id}.png"
        # prompt += f"colored_grid.py includes a library of functions that may be useful. modify this file if you need."
        return modify_coder.run(prompt)

    def run_solve(self):
        max_tries = 1
        tries = 0
        current_result = self.run_tests()
        is_failing = not current_result["success"]

        while is_failing and tries < max_tries:
            print(f"-------------------- ATTEMPT {tries+1}/{max_tries} --------------------------\n")
            ask_coder = self.get_ask_coder()
            modify_coder = self.get_modify_coder()
            is_first = True
            plan = self.get_plan(ask_coder, current_result, is_first=is_first)
            self.get_edit(modify_coder, current_result, plan, is_first=is_first)
            # print(modify_coder.aider_edited_files)
            tries += 1
            current_result = self.run_tests()
            is_failing = not current_result["success"]
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
