from typing import List, Optional, Any, Dict

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


class SolveAttempt(BaseModel):
    task_id: str = Field(..., description="ID of the task being solved")
    concepts_used: List[str] = Field(..., description="List of important concepts used to solve the problem")
    approach: List[str] = Field(
        ..., description="Detailed list of steps and thinking done to solve the problem, ordered"
    )
    python_function: str = Field(
        ...,
        description="""Python function that solves the problem both for the example cases and the test case. 
        This function should be named `solve` and take in a single argument, `input` of type ColoredGrid and return a ColoredGrid
        Example:
        def solve(input: ColoredGrid) -> ColoredGrid:
            pass""",
    )
    solution: List[List[int]] = Field(..., description="List of lists representing the solution grid")
    stdout: Optional[str] = Field(None, description="Standard output of the python function")
    error_message: Optional[str] = Field(None, description="Error message if the attempt was not successful")


class SuccessfulSolve(BaseModel):
    task_id: str = Field(..., description="ID of the task being solved")
    concepts_used: List[str] = Field(..., description="List of important concepts used to solve the problem")
    approach: List[str] = Field(
        ..., description="Detailed list of steps and thinking done to solve the problem, ordered"
    )
    python_function: str = Field(
        ...,
        description="""Python function that solves the problem both for the example cases and the test case. 
        This function should be named `solve` and take in a single argument, `input` of type ColoredGrid and return a ColoredGrid
        Example:
        def solve(input: ColoredGrid) -> ColoredGrid:
            pass""",
    )


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
