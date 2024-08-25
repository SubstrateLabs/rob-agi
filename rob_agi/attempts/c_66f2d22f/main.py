from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_66f2d22f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by analyzing 2x4 sections and marking columns of interest.
    
    The function works as follows:
    1. Creates a new 7x4 output grid.
    2. Analyzes each 2x4 section of the input grid (corresponding to 2 columns).
    3. If a section contains any non-black squares, marks the corresponding
       column in the output grid as gray (5). Otherwise, marks it as black (0).
    
    This process effectively detects and highlights the presence of colored regions
    in the input grid, simplifying the complex color patterns into a binary representation.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(4)])
    
    def analyze_section(section: List[List[int]]) -> bool:
        """
        Check if the 2x4 section contains any non-black squares.
        Return True if it does, False otherwise.
        """
        return any(cell != 0 for row in section for cell in row)
    
    for i in range(7):
        section = [
            input_grid.values[j][2*i:2*i+2] for j in range(4)
        ]
        if analyze_section(section):
            for j in range(4):
                output_grid.values[j][i] = 5
    
    return output_grid
