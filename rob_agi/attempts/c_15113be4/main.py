from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_15113be4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by introducing or modifying a secondary color (sky blue, magenta, or green)
    in a balanced pattern, primarily in the upper half of the grid. The function follows these steps:
    1. Identifies the secondary color to use (8: sky blue, 6: magenta, or 3: green).
    2. Creates or modifies an L-shape pattern in the top-left corner using the secondary color.
    3. Extends the pattern across the upper half of the grid, creating additional L-shapes.
    4. Balances the design by adding smaller patterns in the lower half.
    5. Preserves the yellow grid structure and existing patterns in the lower half.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the applied pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Identify the secondary color
    secondary_color = identify_secondary_color(output_grid)

    # Create or modify L-shape in top-left corner
    create_top_left_l_shape(output_grid, secondary_color)

    # Extend pattern across upper half
    extend_pattern(output_grid, secondary_color)

    # Balance design in lower half
    balance_design(output_grid, secondary_color)

    return output_grid

def identify_secondary_color(grid: ColoredGrid) -> int:
    colors = grid.get_unique_colors()
    if 3 in colors:
        return 3  # green
    elif 6 in colors:
        return 6  # magenta
    elif 8 in colors:
        return 8  # sky blue
    else:
        return 3  # default to green if no secondary color is present

def create_top_left_l_shape(grid: ColoredGrid, color: int):
    for r in range(3):
        for c in range(3):
            if grid.get_cell(r, c) != 4:  # Don't modify yellow cells
                if r == 0 or c == 0:
                    grid.set_cell(r, c, color)

def extend_pattern(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(0, rows // 2, 4):
        for c in range(0, cols, 4):
            if r == 0 and c == 0:
                continue  # Skip top-left corner
            create_l_shape(grid, r, c, color)

def create_l_shape(grid: ColoredGrid, r: int, c: int, color: int):
    directions = [(0, 0), (0, 1), (1, 0)]
    if all(is_valid_cell(grid, r + dr, c + dc) for dr, dc in directions):
        for dr, dc in directions:
            if grid.get_cell(r + dr, c + dc) != 4:
                grid.set_cell(r + dr, c + dc, color)

def is_valid_cell(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    return 0 <= r < rows and 0 <= c < cols

def balance_design(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows // 2, rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1 and is_valid_cell(grid, r, c + 1) and grid.get_cell(r, c + 1) == 1:
                grid.set_cell(r, c, color)
                break  # Only one change per row to maintain balance
