from typing import List, Optional

from substrate import sb

from colored_grid import ColoredGrid
from grid_problem import GridProblem


def get_initial_impression(challenge: GridProblem) -> str:
    return f"""Below is a challenge in a spatial reasoning test. You are tasked to solve it.
    The challenge involves transforming one colored 2D grid into another.
    You will be given a few examples of the transform in the form of input output pairs.
    You will also be given a test case that you need to solve.
    The goal is to find the function that fits all the examples, then apply it to the test case to compute a solution. 
    
    In many cases the transform is something that a human can recognize easily by pattern matching or by 
    reasoning over spatial patterns and applying "core" knowledge concepts. Some are tricky but all have a solution
    that is relatively simple to describe in words.
    
    The grids are represented as a list of lists where each inner list represents a row and each element in the row
    represents a color.
    Number values correspond to colored squares: {ColoredGrid.color_mapping_str}
    
    All grids are rectangular and can be between 1x1 and 30x30 in size.
    
    Here is your current challenge:

    {challenge.to_task_description()}
    
    """


def gather_research(challenges: List[GridProblem], previous_research: Optional[str] = None) -> str:
    challenge_list = "\n- ".join([c.to_task_description() for c in challenges])

    base = f"""Below is a collection of {len(challenges)} spatial reasoning tests.
    Each challenge involves transforming one colored 2D grid into another.
    Each challenge contains a few examples of the transform demonstrated by input output pairs.
    Each challenge also contains a test case that is unsolved.
    The goal is to be able to identify the correct transformation and be able to implement it as a python function.

    In many cases the transform is something that a human can recognize easily by pattern matching or by 
    reasoning over spatial patterns and applying "core" knowledge concepts. Some are tricky but all have a solution
    that is relatively simple to describe in words.

    The grids are represented as a list of lists where each inner list represents a row and each element in the row
    represents a color.
    Number values correspond to colored squares: {ColoredGrid.color_mapping_str}

    All grids are rectangular and can be between 1x1 and 30x30 in size.

    Your goal right now is to gather information about the problem space in general. You are in the research phase.
    Take note of things that you think are important, keep questions in your mind if something is unclear,
    and try to distill what you are taking in. 
    
    <CHALLENGES>
    - {challenge_list}
    </CHALLENGES>
    """
    if not previous_research:
        return base

    base += f"""
        These are the notes from your previous research. Update them with new information.
        Based on the new information, you may want to ask new questions, remove old questions, update your ontology
        or otherwise refactor or refine your understanding of the problem space.
        Do not forget important old information, and new information should be integrated with the old.
        """
    return sb.concat(base, previous_research)


def extract_result(challenge: GridProblem) -> str:
    return f"""You have been tasked to solve a spatial reasoning test.
    You are given a few examples of a transform that you need to find.
    The goal is to find the function that fits the transform, then apply it to the test case.

    You were given this challenge:
    {challenge.to_task_description()}

    Your evaluation was:
    """
