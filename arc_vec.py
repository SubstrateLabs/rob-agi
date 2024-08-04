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
    main_ideas: str
    """Short, dense, comprehensive description of the main idea behind the solution"""
    approach: str
    """Detailed approach and thinking taken to solve the problem"""
    python_function: str
    """
    Python function that the user wrote to solve the problem
    This function should be named `solve` and take in a single argument, `input` of type ColoredGrid and return a ColoredGrid
    Example:
    def solve(input: ColoredGrid) -> ColoredGrid:
       pass
    """
    was_successful: bool
    """Whether the attempt was successful or not"""
    error_message: Optional[str] = None


class ResearchEvent(BaseModel):
    """
    Research phase happens at the beginning and periodically later. Its job is to gather information about the problem
    space in general. In this phase we can ask questions, make hypotheses, and gather data.
    """

    open_questions: list[str]
    """Unanswered questions that arise when observing new information"""
    observations: list[str]
    """Carefully considered observations about this new information that helps in understanding the problem space broadly"""
    data: list[str]
    """Specific data points to remember that may be generally useful in understanding the domain"""
    ontology: KnowledgeGraph
    """A graph of concepts and relationships that are relevant and important to keep in mind when thinking about the problem space"""
