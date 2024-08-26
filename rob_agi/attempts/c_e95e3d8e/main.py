from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_e95e3d8e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying the repeating pattern
    and filling in black areas with the correct pattern elements.
    
    1. Identifies the complete repeating pattern in the input grid
    2. Extracts the pattern and validates it
    3. Fills in black (0) cells with the corresponding pattern element
    4. Returns a new grid with the complete pattern, maintaining original non-black cells
    """
    pattern = identify_complete_pattern(input_grid)
    return fill_pattern(input_grid, pattern)

def identify_complete_pattern(input_grid: ColoredGrid) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    for height in range(1, rows + 1):
        for width in range(1, cols + 1):
            pattern = input_grid.extract_subgrid(0, 0, height, width)
            if is_valid_pattern(input_grid, pattern):
                return pattern
    return input_grid  # Fallback to full grid if no pattern found

def is_valid_pattern(grid: ColoredGrid, pattern: ColoredGrid) -> bool:
    rows, cols = grid.get_dimensions()
    pattern_rows, pattern_cols = pattern.get_dimensions()
    for i in range(rows):
        for j in range(cols):
            if grid.values[i][j] != 0 and grid.values[i][j] != pattern.values[i % pattern_rows][j % pattern_cols]:
                return False
    return True

def fill_pattern(input_grid: ColoredGrid, pattern: ColoredGrid) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    pattern_rows, pattern_cols = pattern.get_dimensions()
    new_values = [
        [
            input_grid.values[i][j] if input_grid.values[i][j] != 0 
            else pattern.values[i % pattern_rows][j % pattern_cols]
            for j in range(cols)
        ]
        for i in range(rows)
    ]
    return ColoredGrid(values=new_values)
