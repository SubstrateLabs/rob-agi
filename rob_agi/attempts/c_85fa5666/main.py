from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_85fa5666(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending colored squares diagonally.
    
    The function processes colors in priority order: Sky Blue (8), Green (3), Orange (7), Magenta (6).
    Each color extends diagonally:
    - Sky Blue (8) and Green (3): extend from top-right to bottom-left
    - Orange (7) and Magenta (6): extend from top-left to bottom-right
    Red (2) 2x2 blocks remain unchanged and block extensions.
    Higher priority colors overwrite lower priority ones.
    Extensions stop at grid boundaries or when encountering higher or equal priority colors.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    colored_squares = []
    red_blocks = set()

    # Identify colored squares and red blocks
    for r in range(rows):
        for c in range(cols):
            color = output_grid.get_cell(r, c)
            if color != 0:
                if color == 2 and is_red_block(output_grid, r, c):
                    red_blocks.update([(r+dr, c+dc) for dr in range(2) for dc in range(2)])
                elif (r, c) not in red_blocks:
                    colored_squares.append((r, c, color))

    # Sort colored squares by priority
    colored_squares.sort(key=lambda x: {8: 0, 3: 1, 7: 2, 6: 3}.get(x[2], 4))

    # Extend colors
    for r, c, color in colored_squares:
        extend_color(output_grid, r, c, color, red_blocks)

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

def extend_color(grid: ColoredGrid, start_row: int, start_col: int, color: int, red_blocks: Set[Tuple[int, int]]):
    """Extend a color diagonally from its starting point."""
    direction = (-1, 1) if color in [3, 8] else (-1, -1)
    x1, y1 = start_row, start_col
    x2, y2 = start_row, start_col
    dx1, dy1 = direction
    dx2, dy2 = -dx1, -dy1
    color_priority = {8: 0, 3: 1, 7: 2, 6: 3}

    while True:
        extended = False
        if is_valid_square(grid, x1, y1) and (x1, y1) not in red_blocks:
            cell_color = grid.get_cell(x1, y1)
            if cell_color == 0 or color_priority.get(cell_color, 4) > color_priority[color]:
                grid.set_cell(x1, y1, color)
                x1, y1 = x1 + dx1, y1 + dy1
                extended = True
        if is_valid_square(grid, x2, y2) and (x2, y2) not in red_blocks:
            cell_color = grid.get_cell(x2, y2)
            if cell_color == 0 or color_priority.get(cell_color, 4) > color_priority[color]:
                grid.set_cell(x2, y2, color)
                x2, y2 = x2 + dx2, y2 + dy2
                extended = True
        if not extended:
            break
