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
formatter = logging.Formatter("\n%(asctime)s [%(threadName)s]\n%(message)s", "%H:%M")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

from rob_agi.arc_util import load_task_set
from rob_agi.computed_result import ComputedResult
from rob_agi.grid_problem import GridProblem
from rob_agi.solver_functions import problem_setup_aider, get_test_case_descriptions
from rob_agi.test_factory import run_pytest, setup_files, read_meta_file, write_meta_file, TestOutput, run_experiment

project_root = Path(__file__).parent.parent
ignore_template = project_root / ".aiderignore"

main_file = "main.py"
test_file = "test.py"
default_max_tries = 1


class Solver:
    def __init__(self, challenge: GridProblem, solution: Optional[ComputedResult] = None):
        self.challenge_id = challenge.id
        self.challenge = challenge
        self.solution = solution
        self.challenge_root = project_root / f"rob_agi/attempts/c_{challenge.id}"
        self.file_paths = {
            "main": self.challenge_root / main_file,
            "test": self.challenge_root / test_file,
            "visual_descriptions": self.challenge_root / "visual_descriptions.yaml",
            "notebook": self.challenge_root / "notebook.txt",
            "experiment": self.challenge_root / "experiment.py",
            "image": project_root / f"data/task_images/{challenge.id}.png",
            "distilled": project_root / f"rob_agi/distilled_solves.txt",
            "colored_grid": project_root / "rob_agi/colored_grid.py",
        }
        self.adhoc_ignore = project_root / f".adhoc-aiderignore-{challenge.id}"
        self.setup()
        self.solved, self.latest_plan, self.total_attempts = read_meta_file(self.challenge_root)
        self.goal = problem_setup_aider(challenge)

    def setup(self, symlink_image=False):
        self.challenge_root.mkdir(parents=True, exist_ok=True)
        with open(ignore_template, "r") as tf:
            with open(self.adhoc_ignore, "w") as f:
                f.write(tf.read())
                f.write("\n")
                if symlink_image:
                    f.write(f"rob_agi/attempts/c_{self.challenge.id}/image.png\n")
                f.write(f"!rob_agi/attempts/c_{self.challenge.id}\n")
                f.write(f"!.adhoc-aiderignore-{self.challenge.id}\n")
        setup_files(self.challenge, self.solution, self.challenge_root)
        if symlink_image:
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
        auto_commits = kwargs.pop("auto_commits", False)
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
            fnames = [
                self.file_paths["main"],
                self.file_paths["test"],
                self.file_paths["distilled"],
                self.file_paths["notebook"],
                self.file_paths["experiment"],
            ]
        return self.get_coder(edit_format="ask", fnames=fnames)

    def get_reflect_coder(self, from_coder=None, fnames=None):
        if fnames is None:
            fnames = [
                self.file_paths["visual_descriptions"],
                self.file_paths["notebook"],
                self.file_paths["experiment"],
            ]
        read_only_fnames = [
            self.file_paths["main"],
            self.file_paths["test"],
            self.file_paths["distilled"],
        ]
        return self.get_coder(
            from_coder=from_coder, fnames=fnames, read_only_fnames=read_only_fnames, auto_commits=True
        )

    def get_modify_coder(self, fnames=None):
        if fnames is None:
            fnames = [self.file_paths["main"], self.file_paths["visual_descriptions"]]
        read_only_fnames = [
            self.file_paths["notebook"],
            self.file_paths["test"],
            self.file_paths["colored_grid"],
            self.file_paths["distilled"],
        ]
        return self.get_coder(fnames=fnames, read_only_fnames=read_only_fnames, auto_commits=True)

    def run_tests(self) -> TestOutput:
        result = run_pytest(self.file_paths["test"])
        logger.info(result)
        return result

    def get_prefix(self, is_first: bool) -> str:
        if is_first:
            prefix = f"{self.goal}\n\nCurrently the tests used to validate the solution are failing. This means that your previous solution is incorrect."
        else:
            prefix = "The tests used to validate the solution are still failing. This means that your previous solution is incorrect."
        return prefix

    def get_plan(self, current_result: TestOutput, is_first: bool = False) -> str:
        desc = self.get_visual_descriptions()
        ask_coder = self.get_ask_coder()
        prefix = self.get_prefix(is_first)

        prompt = (
            f"{prefix}\n\n<VALIDATION_OUTPUT>\n{current_result.error}\n{current_result.output}</VALIDATION_OUTPUT>\n"
        )
        prompt += f"\n<VISUAL_DESCRIPTIONS>\n{desc}\n</VISUAL_DESCRIPTIONS>\n"
        exp = run_experiment(self.file_paths["experiment"])
        logger.info(f"======================================\n\n{exp}\n\n")
        if exp:
            prompt += f"\n<EXPERIMENT_OUTPUT>\nSTDOUT:{exp.output}\nSTDERR:{exp.error}\n</EXPERIMENT_OUTPUT>\n"
        prompt += f"This is attempt number {self.total_attempts + 1} to solve the challenge.\n"
        if self.total_attempts > 7:
            prompt += "This means that this is either especially difficult or you've likely been going down the wrong path. Feel free to relinquish a lot of what you think you know about this problem and try to zoom out and see it with fresh eyes.\n"
        if self.total_attempts > 2:
            prompt += "Before we try to come up with a new solution let's take in all of this information, notice what may be important, and ask ourselves some relevant questions to help us introspect and explore the problem more completely.\n"
            prompt += "Look at all the cases you've been presented with. Come up with 3-5 questions that you think are important to ask yourself. These questions should help you understand the challenge better and guide you to a better, more general solution that solves the challenge. Maybe they are about discrepancies, new things you've noticed that might be important, your own assumptions, specific failure cases, conspicuous patterns, things you are unsure about, etc. Do not answer the questions yet.\n"
            ask_coder.run(prompt)
            ask_coder.run(
                "Now, think carefully and answer the questions you came up with. But don't propose a solution quite yet.\n"
            )
            solution = ""
        else:
            solution = prompt
        # ask_coder.run(prompt)
        solution += "Next, examine all the information you have, state your understanding of the challenge, and propose a detailed solution to the challenge in words. Any solution must always apply to every case, so always try to generalize over all cases rather than only replicating individual cases.\n"
        solution += "Then reflect on your idea. Look very closely and notice if there are any other patterns or discrepancies worth noting. Remember this is about identifying abstract, intuitive ideas about what is happening. This takes humility, be honest, give it good effort and avoid over confidence; don't ever apologize, think hard and creatively.\n"
        solution += f"Explicitly consider how your idea applies to each of the examples and test cases in attempts/{self.challenge_id}/test.py. To check your thinking, illustrate how your idea either works or doesn't for each case.\n"
        solution += f"If the rule(s) you came up with does not apply to any specific case, call it out and think about a more general idea that does apply in every case. Be meticulous and careful in your reflection. Sometimes you need to zoom out to see how a single idea can apply to all cases.\n"
        ask_coder.run(solution)

        res = ask_coder.run("Based on that reflection detail a step by step plan for how to solve the challenge\n")
        return res

    def get_edit(self, current_result: TestOutput, plan, is_first=True, update_visual_desc=False) -> str:
        modify_coder = self.get_modify_coder()
        prefix = self.get_prefix(is_first)
        prompt = f"{prefix}\n\n<TEST_OUTPUT>\n{current_result.error}\n{current_result.output}</TEST_OUTPUT>\n"
        prompt += f"\nYour most recent plan is:\n<RECENT_PLANNING>\n{plan}\n</RECENT_PLANNING>\n"
        prompt += f"Use that latest planning and solve the challenge by modifying the implementation file. Always ensure that the docstring to solve_{self.challenge_id} includes a correct summary of the solution in words.\n"
        prompt += "Make sure your code changes are in the SEARCH/REPLACE format."
        modifications = modify_coder.run(prompt)
        logger.info(f"\n~~~~~~~~~CODE_EDIT~~~~~~~~~~~\n{modify_coder.aider_edited_files}")
        res = self.run_tests()
        reflect_coder = self.get_reflect_coder(modify_coder)
        if not res.success:
            reflect = f"The output of the tests after your changes is:\n\n<STDERR>\n{res.error}</STDERR>\n\n<STDOUT>{res.output}</STDOUT>\n"
            reflect += "Based on the output of the tests, reflect on what you have learned. `notebook.txt` is where you keep the latest notes for solving the challenge. This file should always contain accurate and up-to-date information about the challenge, and over time it will help future versions of you solve it. Revise it or append to it according to what you've learned. It should be well maintained and never be more than 3 pages long, ideally shorter. It's often useful to explicitly keep track of things that you have tried that do not work, since it prevents your thinking from going in circles. What are you absolutely certain about? What are you guessing about? What remains unknown? Make sure you use the SEARCH/REPLACE format\n"
            reflect_coder.run(reflect)
            experiment = "Also if you want to run an experiment in python to help you find out more pointed information, you can write code in `experiment.py`. This file will be run immediately and the stdout and stderr will be available in the next pass at solving. Make sure you use the SEARCH/REPLACE format"
            reflect_coder.run(experiment)
            if update_visual_desc:
                update_prompt = "If the `visual_descriptions.yaml` can be improved (more detail, more accurate, better intuitive abstractions, cutting irrelevant info, clarity, etc), include those changes too. This file is purely for descriptions of the grid images. It should not have any information about the code or the solution. These descriptions should help someone trying to solve this problem though, so it should include language that is relevant for solving the problem. Make sure you use the SEARCH/REPLACE format.\n"
                modify_coder.run(update_prompt)
            logger.info(f"\n~~~~~~~~~REFLECT_EDIT~~~~~~~~~~~\n{modify_coder.aider_edited_files}")
        return modifications

    def get_visual_descriptions(self, overwrite: bool = False) -> str:
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
                result += f"{key}_input:\n{value.get('input')}\n"
                result += f"{key}_output:\n{value.get('output')}\n"
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
        current_result = self.run_tests()
        if current_result.success:
            logger.info(f"Tests are passing. No need to run the solver. ({time.perf_counter() - t0:.2f}s)")
            write_meta_file(
                self.challenge_root,
                solved=True,
                latest_plan=self.latest_plan,
                total_attempts=self.total_attempts,
            )
            return True

        is_failing = not current_result.success
        local_tries = 0
        while is_failing and local_tries < max_tries:
            logger.info(f"-------------------- ATTEMPT {local_tries+1}/{max_tries} --------------------------\n")
            if prev_solution:
                plan = prev_solution
            else:
                plan = self.get_plan(current_result)

            update_desc = self.total_attempts > 0 and self.total_attempts % 4 == 0
            self.get_edit(current_result, plan, update_visual_desc=update_desc)
            local_tries += 1
            self.total_attempts += 1
            current_result = self.run_tests()
            is_failing = not current_result.success
            write_meta_file(
                self.challenge_root,
                solved=not is_failing,
                latest_plan=plan,
                total_attempts=self.total_attempts,
            )
        logger.info(f"Total time: {time.perf_counter() - t0:.2f}s")
        return current_result.success

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
    print(c)
    sln = solutions.get(challenge_id)
    solver = Solver(c, sln)
    print("\n\n" + c.test_cases[0].human_print() + "\n\n")
    # solver.run_solve(max_tries=4)

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
