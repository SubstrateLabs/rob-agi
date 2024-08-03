from typing import List
from pydantic import BaseModel

from colored_grid import ColoredGrid


class ComputedResult(BaseModel):
    outputs: List[ColoredGrid]

    def validate(self, expected: List[ColoredGrid]) -> bool:
        if len(self.outputs) != len(expected):
            return False
        for output, expected_output in zip(self.outputs, expected):
            if output != expected_output:
                return False
        return True
