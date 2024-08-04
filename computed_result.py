from typing import List, Optional
from pydantic import BaseModel

from colored_grid import ColoredGrid


class ComputedResult(BaseModel):
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
