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
