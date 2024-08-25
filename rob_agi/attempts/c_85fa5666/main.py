from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict

def solve_85fa5666(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending colored squares diagonally.
    
    The function processes colors in priority order: Sky Blue (8), Green (3), Orange (7), Magenta (6).
    Each color extends diagonally:
    - Sky Blue (8) and Green (3): extend from top-right to bottom-left
    - Orange (7) and Magenta (6): extend from top-left to bottom-right
    Red (2) 2x2 blocks remain unchanged and block extensions.
    Higher priority colors overwrite lower priority ones.
    Extensions continue through intersections and stop at grid boundaries or red blocks.
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
    colored_squares.sort(key=lambda x: get_color_priority(x[2]))

    # Extend colors
    for r, c, color in colored_squares:
        extend_color(output_grid, r, c, color, red_blocks)

    return output_grid

def is_red_block(grid: ColoredGrid, row: int, col: int) -> bool:
    """Check if a given square is part of a red 2x2 block."""
    if grid.get_cell(row, col) != 2:
        return False
    for dr, dc in [(0, 1), (1, 0), (1, 1)]:
        if not is_valid_cell(row + dr, col + dc, grid) or grid.get_cell(row + dr, col + dc) != 2:
            return False
    return True

def is_valid_cell(row: int, col: int, grid: ColoredGrid) -> bool:
    """Check if a given coordinate is within the grid boundaries."""
    rows, cols = grid.get_dimensions()
    return 0 <= row < rows and 0 <= col < cols

def get_color_priority(color: int) -> int:
    """Return the priority of a given color."""
    return {8: 0, 3: 1, 7: 2, 6: 3}.get(color, 4)

def extend_color(grid: ColoredGrid, start_row: int, start_col: int, color: int, red_blocks: Set[Tuple[int, int]]):
    """Extend a color diagonally from its starting point."""
    directions = [(-1, 1), (1, -1)] if color in [8, 3] else [(-1, -1), (1, 1)]
    
    for dx, dy in directions:
        x, y = start_row, start_col
        while True:
            x, y = x + dx, y + dy
            if not is_valid_cell(x, y, grid) or (x, y) in red_blocks:
                break
            cell_color = grid.get_cell(x, y)
            if cell_color == 0 or get_color_priority(color) < get_color_priority(cell_color):
                grid.set_cell(x, y, color)
            else:
                break
