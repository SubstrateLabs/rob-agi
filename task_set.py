from typing import Dict
from pydantic import BaseModel

from computed_result import ComputedResult
from grid_problem import GridProblem


class TaskSet(BaseModel):
    challenges: Dict[str, GridProblem]
    solutions: Dict[str, ComputedResult]
