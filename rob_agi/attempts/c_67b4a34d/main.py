from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_67b4a34d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the challenge by extracting the corner cells from the central 8x8 region of the 16x16 input grid to form a 4x4 output grid.
    
    The solution selects cells from the input grid in the following pattern:
    - Top-left 2x2 quadrant: cells (4,4), (4,5), (5,4), (5,5)
    - Top-right 2x2 quadrant: cells (4,11), (4,12), (5,11), (5,12)
    - Bottom-left 2x2 quadrant: cells (11,4), (11,5), (12,4), (12,5)
    - Bottom-right 2x2 quadrant: cells (11,11), (11,12), (12,11), (12,12)
    
    Args:
    input_grid (ColoredGrid): A 16x16 input grid
    
    Returns:
    ColoredGrid: A 4x4 grid extracted from specific positions of the input grid
    """
    rows, cols = input_grid.get_dimensions()
    if rows != 16 or cols != 16:
        raise ValueError("Input grid must be 16x16")
    
    output_values = [
        [input_grid.values[4][4], input_grid.values[4][11], input_grid.values[4][5], input_grid.values[4][12]],
        [input_grid.values[11][4], input_grid.values[11][11], input_grid.values[11][5], input_grid.values[11][12]],
        [input_grid.values[5][4], input_grid.values[5][11], input_grid.values[5][5], input_grid.values[5][12]],
        [input_grid.values[12][4], input_grid.values[12][11], input_grid.values[12][5], input_grid.values[12][12]]
    ]
    
    return ColoredGrid(values=output_values)
