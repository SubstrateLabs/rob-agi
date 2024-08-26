from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_e95e3d8e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying the repeating pattern
    and filling in black areas with the correct pattern elements.
    
    1. Identifies the smallest repeating pattern unit in the input grid
    2. Fills in black (0) cells with the corresponding pattern element
    3. Returns a new grid with the complete pattern, maintaining original non-black cells
    """
    pattern_unit = identify_pattern_unit(input_grid)
    return fill_pattern(input_grid, pattern_unit)

def identify_pattern_unit(input_grid: ColoredGrid) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    for height in range(1, rows + 1):
        for width in range(1, cols + 1):
            if is_valid_pattern(input_grid, height, width):
                return input_grid.extract_subgrid(0, 0, height, width)
    return input_grid  # Fallback to full grid if no pattern found

def is_valid_pattern(grid: ColoredGrid, height: int, width: int) -> bool:
    rows, cols = grid.get_dimensions()
    pattern = grid.extract_subgrid(0, 0, height, width)
    for i in range(rows):
        for j in range(cols):
            if grid.values[i][j] != 0 and grid.values[i][j] != pattern.values[i % height][j % width]:
                return False
    return True

def fill_pattern(input_grid: ColoredGrid, pattern: ColoredGrid) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    new_values = [
        [
            input_grid.values[i][j] if input_grid.values[i][j] != 0 
            else pattern.values[i % pattern.num_rows][j % pattern.num_cols]
            for j in range(cols)
        ]
        for i in range(rows)
    ]
    return ColoredGrid(values=new_values)
