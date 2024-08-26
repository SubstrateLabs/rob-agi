from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_67b4a34d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the challenge by extracting specific cells from the 16x16 input grid to form a 4x4 output grid.
    
    The solution selects cells from the input grid in the following pattern:
    - Top-left 2x2 quadrant: cells (4,4), (4,5), (5,4), (5,5)
    - Top-right 2x2 quadrant: cells (4,10), (4,11), (5,10), (5,11)
    - Bottom-left 2x2 quadrant: cells (10,4), (10,5), (11,4), (11,5)
    - Bottom-right 2x2 quadrant: cells (10,10), (10,11), (11,10), (11,11)
    
    Args:
    input_grid (ColoredGrid): A 16x16 input grid
    
    Returns:
    ColoredGrid: A 4x4 grid extracted from specific positions of the input grid
    """
    rows, cols = input_grid.get_dimensions()
    if rows != 16 or cols != 16:
        raise ValueError("Input grid must be 16x16")
    
    output_values = [
        [input_grid.values[4][4], input_grid.values[4][10], input_grid.values[4][5], input_grid.values[4][11]],
        [input_grid.values[10][4], input_grid.values[10][10], input_grid.values[10][5], input_grid.values[10][11]],
        [input_grid.values[5][4], input_grid.values[5][10], input_grid.values[5][5], input_grid.values[5][11]],
        [input_grid.values[11][4], input_grid.values[11][10], input_grid.values[11][5], input_grid.values[11][11]]
    ]
    
    return ColoredGrid(values=output_values)
