from typing import Dict
from pydantic import BaseModel

from rob_agi.computed_result import ComputedResult
from rob_agi.grid_problem import GridProblem


class TaskSet(BaseModel):
    challenges: Dict[str, GridProblem]
    solutions: Dict[str, ComputedResult]
