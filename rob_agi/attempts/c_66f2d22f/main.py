from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_66f2d22f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by analyzing 4x2 sections and marking columns of interest.
    
    The function works as follows:
    1. Creates a new 7x4 output grid, initially filled with black (0).
    2. Analyzes each 4x2 section of the input grid (corresponding to 1 column in the output).
    3. If either column in the section contains 3 or more non-black squares,
       marks the corresponding column in the output grid as gray (5).
    4. Otherwise, leaves the corresponding column in the output grid as black (0).
    
    This process effectively detects and highlights the presence of significant
    colored regions in the input grid, simplifying the complex color patterns
    into a binary representation.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(4)])
    
    for i in range(7):
        col1_count = sum(1 for row in range(4) if input_grid.values[row][2*i] != 0)
        col2_count = sum(1 for row in range(4) if input_grid.values[row][2*i+1] != 0)
        
        if col1_count >= 3 or col2_count >= 3:
            for row in range(4):
                output_grid.values[row][i] = 5
    
    return output_grid
