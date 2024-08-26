from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_e95e3d8e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying the pattern unit,
    creating a full pattern grid, and revealing the underlying pattern in black areas.
    
    1. Identifies the smallest repeating pattern unit in the input grid
    2. Creates a full pattern grid by tiling the pattern unit
    3. Generates the output grid by revealing the underlying pattern in black areas
    4. Returns a new grid with the complete pattern, maintaining original non-black cells
    """
    pattern_unit = identify_pattern_unit(input_grid)
    full_pattern = create_full_pattern(pattern_unit, input_grid.num_rows, input_grid.num_cols)
    return reveal_pattern(input_grid, full_pattern)

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
    for i in range(0, rows, height):
        for j in range(0, cols, width):
            if not all(
                grid.values[i + r][j + c] in (0, pattern.values[r % height][c % width])
                for r in range(min(height, rows - i))
                for c in range(min(width, cols - j))
            ):
                return False
    return True

def create_full_pattern(pattern_unit: ColoredGrid, rows: int, cols: int) -> ColoredGrid:
    return pattern_unit.tile_grid(max(rows // pattern_unit.num_rows + 1, cols // pattern_unit.num_cols + 1))

def reveal_pattern(input_grid: ColoredGrid, full_pattern: ColoredGrid) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    new_values = [
        [
            input_grid.values[i][j] if input_grid.values[i][j] != 0 else full_pattern.values[i][j]
            for j in range(cols)
        ]
        for i in range(rows)
    ]
    return ColoredGrid(values=new_values)
