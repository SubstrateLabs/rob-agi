from typing import List, Optional, Any, Dict, get_type_hints

from pydantic import BaseModel, Field


class Node(BaseModel):
    id: int
    label: str


class Edge(BaseModel):
    source_id: int
    target_id: int
    data: Dict[str, Any]


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)


def simplify_type(type_hint):
    type_str = str(type_hint)
    if type_str.startswith("<class '") and type_str.endswith("'>"):
        return type_str[8:-2]  # Remove <class '...'> wrapper
    return type_str.replace("typing.", "")


class SuccessfulSolve(BaseModel):
    task_id: str = Field(..., description="ID of the task being solved")
    concepts_used: List[str] = Field(..., description="List of important concepts used to solve the problem")
    approach: List[str] = Field(
        ..., description="Detailed list of steps and thinking done to solve the problem, ordered"
    )
    solution: List[List[int]] = Field(..., description="List of lists representing the solution grid")
    python_function: str = Field(
        ...,
        description="Valid python function that solves the problem both for the example cases and the test case. This function should be named `solve_<challenge_id>` and take in a single argument, `input` of type ColoredGrid and return a ColoredGrid. The <challenge_id> should be replaced with the task_id of the challenge: def solve_<challenge_id>(input: ColoredGrid) -> ColoredGrid: ...",
    )

    @classmethod
    def field_summary(cls) -> str:
        summary = [f"Model: {cls.__name__}\n"]
        for name, field in cls.model_fields.items():
            field_type = simplify_type(get_type_hints(cls)[name])
            description = field.description or "No description"
            description = " ".join(description.split())
            summary.append(f"{name} ({field_type}): {description}")
        return "\n".join(summary)


class SolveAttempt(SuccessfulSolve):
    stdout: Optional[str] = Field(None, description="Standard output of the python function")
    error_message: Optional[str] = Field(None, description="Error message if the attempt was not successful")

    @classmethod
    def simple_json_schema(cls):
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "concepts_used": {"type": "array", "items": {"type": "string"}},
                "approach": {"type": "array", "items": {"type": "string"}},
                "solution": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "python_function": {"type": "string"},
                "stdout": {"type": "string"},
                "error_message": {"type": "string"},
            },
            "required": ["task_id", "concepts_used", "approach", "solution", "python_function"],
        }


class ResearchEvent(BaseModel):
    """
    Research phase happens at the beginning and periodically later. Its job is to gather information about the problem
    space in general. In this phase we can ask questions, gather ideas, think about how to better understand and approach.
    """

    current_total_knowledge: str
    """Organized, detailed summary of all the knowledge about the dataset that that we know so far. This is a living document that grows and refactors as we learn more and see more challenges. It is a place to make sure we don't forget anything important and distill the most important and useful information"""
    ordered_concept_list: list[str]
    """Running list of ideas, concepts, and knowledge that is necessary to solve all the challenges in the dataset. This list is ordered by importance and relevance. It changes over time as we see and solve more"""
    new_knowledge: str
    """New knowledge gained from the current research event. This will be incorporated into the current total knowledge, so should be a summary of the most important new things learned"""
