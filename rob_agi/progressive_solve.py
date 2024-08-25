import time
from pathlib import Path
from typing import Optional
import logging
import yaml

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(threadName)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

from rob_agi.arc_util import load_task_set
from rob_agi.computed_result import ComputedResult
from rob_agi.grid_problem import GridProblem
from rob_agi.solver_functions import problem_setup_aider, get_test_case_descriptions
from rob_agi.test_factory import run_pytest, setup_files, read_meta_file, write_meta_file

project_root = Path(__file__).parent.parent
ignore_template = project_root / ".aiderignore"

main_file = "main.py"
test_file = "test.py"
default_max_tries = 1


class Solver:
    def __init__(self, challenge: GridProblem, solution: Optional[ComputedResult]):
        self.challenge_id = challenge.id
        self.challenge = challenge
        self.solution = solution
        self.challenge_root = project_root / f"rob_agi/attempts/c_{challenge.id}"
        self.file_paths = {
            "main": self.challenge_root / main_file,
            "test": self.challenge_root / test_file,
            "visual_descriptions": self.challenge_root / "visual_descriptions.yaml",
            "image": project_root / f"data/task_images/{challenge.id}.png",
        }
        self.adhoc_ignore = project_root / f".adhoc-aiderignore-{challenge.id}"
        self.setup()
        self.solved, self.latest_plan, self.total_attempts = read_meta_file(self.challenge_root)
        self.goal = problem_setup_aider(challenge)

    def setup(self):
        self.challenge_root.mkdir(parents=True, exist_ok=True)
        with open(ignore_template, "r") as tf:
            with open(self.adhoc_ignore, "w") as f:
                f.write(tf.read())
                f.write("\n")
                f.write(f"rob_agi/attempts/c_{self.challenge.id}/image.png\n")
                f.write(f"!rob_agi/attempts/c_{self.challenge.id}\n")
                f.write(f"!.adhoc-aiderignore-{self.challenge.id}\n")
        setup_files(self.challenge, self.solution, self.challenge_root)
        try:
            (self.challenge_root / f"image.png").symlink_to(self.file_paths["image"])
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
        io.yes = False
        auto_commits = kwargs.pop("auto_commits", True)
        coder: Coder = Coder.create(
            main_model=Model("claude-3-5-sonnet-20240620"),
            io=io,
            cache_prompts=True,
            stream=False,
            auto_commits=auto_commits,
            **kwargs,
        )
        coder.repo.aider_ignore_file = self.adhoc_ignore
        # logger.info(coder.repo.aider_ignore_file)
        # logger.info(coder.root)
        # logger.info(coder.get_all_relative_files())
        # logger.info(coder.get_repo_map())
        return coder

    def get_ask_coder(self, fnames=None):
        if fnames is None:
            fnames = [self.file_paths["main"], self.file_paths["test"]]
        return self.get_coder(edit_format="ask", fnames=fnames)

    def get_modify_coder(self):
        fnames = [self.file_paths["main"], self.file_paths["visual_descriptions"]]
        read_only_fnames = [self.file_paths["test"], project_root / "rob_agi/colored_grid.py"]
        return self.get_coder(fnames=fnames, read_only_fnames=read_only_fnames, auto_commits=True)

    def run_tests(self):
        result = run_pytest(self.file_paths["test"])
        logger.info(result)
        return result

    def get_prefix(self, is_first):
        if is_first:
            prefix = f"{self.goal}\n\nCurrently the tests used to validate the solution are failing. This means that your previous solution is incorrect."
        else:
            prefix = "The tests used to validate the solution are still failing. This means that your previous solution is incorrect."
        return prefix

    def get_plan(self, current_result, is_first=False):
        desc = self.get_visual_descriptions()
        ask_coder = self.get_ask_coder()
        prefix = self.get_prefix(is_first)
        prompt = f"{prefix}\n\n<VALIDATION_OUTPUT>\n{current_result['error']}\n{current_result['output']}</VALIDATION_OUTPUT>\n"
        prompt += f"\n<VISUAL_DESCRIPTIONS>{desc}</VISUAL_DESCRIPTIONS>\n"
        prompt += "Examine all the information you have, state your understanding of the challenge, and propose a detailed solution to the challenge in words. Any solution must always apply to every case, not just the failing exception here.\n"
        prompt += "Then reflect on your idea. Look very closely and notice if there are any other patterns or discrepancies worth noting. Remember this is about identifying abstract, intuitive ideas about what is happening.\n"
        prompt += f"Explicitly consider how your idea applies to each of the examples and test cases in attempts/{self.challenge_id}/test.py. To check your thinking, illustrate how your idea either works or doesn't for each case.\n"
        prompt += f"If the rule(s) you came up with does not apply to any specific case, call it out and think about a more general idea that does apply in every case. Be meticulous and careful in your reflection. Sometimes you need to zoom out to see how a single idea can apply to all cases.\n"
        ask_coder.run(prompt)

        res = ask_coder.run("Based on that reflection detail a step by step plan for how to solve the challenge\n")
        return res

    def get_edit(self, current_result, plan, is_first=True, update_visual_descriptions=True) -> str:
        modify_coder = self.get_modify_coder()
        prefix = self.get_prefix(is_first)
        prompt = f"{prefix}\n\n<VALIDATION_OUTPUT>\n{current_result['error']}\n{current_result['output']}</VALIDATION_OUTPUT>\n"
        prompt += f"\nYour latest thinking is:\n<LATEST_THINKING>\n{plan}\n</LATEST_THINKING>\n"
        prompt += f"Use that latest thinking and solve the challenge by modifying the implementation file. Always ensure that the docstring to solve_{self.challenge_id} includes a correct summary of the solution in words.\n"
        prompt += "Make sure your code changes are in the SEARCH/REPLACE format."
        # prompt += f"An image of the challenge is provided at {self.challenge.id}.png"
        # prompt += f"colored_grid.py includes a library of functions that may be useful. modify this file if you need."
        logger.info(f"\n~~~~~~~~~EDITED~~~~~~~~~~~\n{modify_coder.aider_edited_files}")
        modifications = modify_coder.run(prompt)
        if update_visual_descriptions:
            update_prompt = "If the visual_descriptions.yaml can be improved (more detail, better abstractions, cutting irrelevant info), include those changes too"
            modify_coder.run(update_prompt)
        return modifications

    def get_visual_descriptions(self, overwrite: bool = True) -> str:
        target_file = self.file_paths["visual_descriptions"]
        if not overwrite and target_file.exists():
            return self.parse_descriptions()
        fnames = [self.file_paths["test"]]
        descriptions_coder = self.get_ask_coder(fnames=fnames)
        prompt = get_test_case_descriptions()
        yaml_content = descriptions_coder.run(prompt)
        target_file.write_text(yaml_content)
        return self.parse_descriptions()

    def parse_descriptions(self) -> str:
        target = self.file_paths["visual_descriptions"]
        if not target.exists():
            return ""

        parsed = Solver.parse_yaml_file(target)
        result = ""
        if parsed:
            for key, value in parsed.items():
                result += f"{key}_input:\n{value['input']}\n"
                result += f"{key}_output:\n{value['output']}\n"
        return result

    @staticmethod
    def parse_yaml_file(file_path):
        with open(file_path, "r") as file:
            try:
                data = yaml.safe_load(file)
                return data
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file: {e}")
                return None

    def run_solve(self, max_tries=default_max_tries, prev_solution=None) -> bool:
        t0 = time.perf_counter()
        local_tries = 0
        current_result = self.run_tests()
        is_failing = not current_result["success"]

        while is_failing and local_tries < max_tries:
            logger.info(f"-------------------- ATTEMPT {local_tries+1}/{max_tries} --------------------------\n")
            if prev_solution:
                plan = prev_solution
            else:
                plan = self.get_plan(current_result)

            self.get_edit(current_result, plan)
            local_tries += 1
            self.total_attempts += 1
            current_result = self.run_tests()
            is_failing = not current_result["success"]
            write_meta_file(
                self.challenge_root, solved=not is_failing, latest_plan=plan, total_attempts=self.total_attempts
            )
            logger.info(f"SUCCESS: {current_result['success']}")

        logger.info(f"Total time: {time.perf_counter() - t0:.2f}s")
        return current_result["success"]

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
    c = challenges[challenge_id]
    sln = solutions.get(challenge_id)
    solver = Solver(c, sln)
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
