from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_85fa5666(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending colored squares diagonally.
    
    The function identifies non-red colored squares and extends them diagonally:
    - Green (3) and Sky Blue (8): extend from top-right to bottom-left
    - Magenta (6) and Orange (7): extend from top-left to bottom-right
    Red (2) 2x2 blocks remain unchanged. Extensions stop before intersecting
    with other colors, paths, red blocks, or grid boundaries.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    colored_squares = []

    # Identify colored squares and red blocks
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) != 0 and not is_red_block(output_grid, r, c):
                colored_squares.append((r, c, output_grid.get_cell(r, c)))

    # Extend colors
    for r, c, color in colored_squares:
        extend_color(output_grid, r, c, color)

    return output_grid

def is_red_block(grid: ColoredGrid, row: int, col: int) -> bool:
    """Check if a given square is part of a red 2x2 block."""
    if grid.get_cell(row, col) != 2:
        return False
    for dr, dc in [(0, 1), (1, 0), (1, 1)]:
        if not is_valid_square(grid, row + dr, col + dc) or grid.get_cell(row + dr, col + dc) != 2:
            return False
    return True

def is_valid_square(grid: ColoredGrid, row: int, col: int) -> bool:
    """Check if a given coordinate is within the grid boundaries."""
    rows, cols = grid.get_dimensions()
    return 0 <= row < rows and 0 <= col < cols

def extend_color(grid: ColoredGrid, start_row: int, start_col: int, color: int):
    """Extend a color diagonally from its starting point."""
    direction = (-1, 1) if color in [3, 8] else (-1, -1)
    forward_path = get_path(grid, start_row, start_col, direction)
    backward_path = get_path(grid, start_row, start_col, (-direction[0], -direction[1]))
    
    # Apply paths
    for r, c in forward_path + backward_path:
        grid.set_cell(r, c, color)

def get_path(grid: ColoredGrid, start_row: int, start_col: int, direction: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get the path of a color extension in a given direction."""
    path = []
    r, c = start_row + direction[0], start_col + direction[1]
    while is_valid_square(grid, r, c):
        if grid.get_cell(r, c) != 0 or is_red_block(grid, r, c):
            break
        path.append((r, c))
        r, c = r + direction[0], c + direction[1]
    return path
