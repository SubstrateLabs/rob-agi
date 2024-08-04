from typing import List, Optional
from pydantic import BaseModel, Field

from colored_grid import ColoredGrid


class ComputedResult(BaseModel):
    task_id: str = Field(..., description="ID of the task being solved")
    outputs: List[ColoredGrid]

    def validate(self, expected: "ComputedResult") -> bool:
        if len(self.outputs) != len(expected.outputs):
            return False
        for i, output in enumerate(self.outputs):
            if output != expected.outputs[i]:
                return False
        return True

    def comparison_report(self, expected: "ComputedResult") -> str:
        result = ""
        for i, output in enumerate(self.outputs):
            result += f"Output {i + 1}:\n"
            result += str(output)
            result += f"\nExpected:\n"
            result += str(expected.outputs[i])
            result += f"\nMatch: {output == expected.outputs[i]}\n"
        return result

    @classmethod
    def json_schema(cls, max_outputs: Optional[int] = None) -> dict:
        schema = cls.model_json_schema()
        if max_outputs is None:
            return schema
        schema["properties"]["outputs"]["maxItems"] = max_outputs
        schema["properties"]["outputs"]["minItems"] = max_outputs
        return schema

    def result_description(self) -> str:
        result = f"Task ID: {self.task_id}\n"
        for i, output in enumerate(self.outputs):
            result += f"Output {i + 1}:\n{str(output)}\n"
        return result

    # def to_task_description(self) -> str:
    #     task_string = f"Task ID: {self.id}\n"
    #     for i, example in enumerate(self.examples):
    #         task_string += f"Example {i + 1}:\n"
    #         task_string += f"Input:\n{str(example.input)}\n"
    #         task_string += f"Output:\n{str(example.output)}\n\n"
    #
    #     for i, test_case in enumerate(self.test_cases):
    #         task_string += f"Test Case {i + 1}:\n{str(test_case)}\n\n"
    #
    #     return task_string
