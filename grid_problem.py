from typing import Dict, Any, List

from pydantic import BaseModel, model_validator
from colored_grid import ColoredGrid


class Solution(BaseModel):
    input: ColoredGrid
    output: ColoredGrid


class GridProblem(BaseModel):
    """
    GridProblem represents a spatial reasoning problem where the goal is to find a function that
    transforms one 2D colored grid into another.

    A problem consists of a few examples of the transform in the form of input output pairs.
    A problem also has one or more test cases that need to be solved. About 96% of problems have only one test case.

    The goal is to find the function that fits the transform, then apply it to the test case to compute a solution.
    The function should validate correctly on all the example cases.
    """

    id: str
    examples: list[Solution]
    test_cases: list[ColoredGrid]

    def to_task_description(self) -> str:
        task_string = f"Task ID: {self.id}\n"
        for i, example in enumerate(self.examples):
            task_string += f"Example {i + 1}:\n"
            task_string += f"Input:\n{str(example.input)}\n"
            task_string += f"Output:\n{str(example.output)}\n\n"

        for i, test_case in enumerate(self.test_cases):
            task_string += f"Test Case {i + 1}:\n{str(test_case)}\n\n"

        return task_string

    def example_string(self):
        task_string = ""
        for i, example in enumerate(self.examples):
            task_string += f"Example {i + 1}:\n"
            task_string += f"Input:\n{str(example.input)}\n"
            task_string += f"Output:\n{str(example.output)}\n\n"
        return task_string

    @classmethod
    def parse(cls, id: str, train: List[Dict[str, Any]], test: List[Dict[str, Any]]) -> "GridProblem":
        examples_list = []
        for ex in train:
            examples_list.append(
                Solution(input=ColoredGrid(values=ex["input"]), output=ColoredGrid(values=ex["output"]))
            )
        test_cases_list = []
        for test_case in test:
            test_cases_list.append(ColoredGrid(values=test_case["input"]))
        return cls(id=id, examples=examples_list, test_cases=test_cases_list)
