from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def solve_85fa5666(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending colored squares diagonally with bouncing behavior.
    
    The function processes colors in priority order: Sky Blue (8), Green (3), Orange (7), Magenta (6).
    Each color extends diagonally:
    - Sky Blue (8) and Green (3): start from top-right to bottom-left
    - Orange (7) and Magenta (6): start from top-left to bottom-right
    Colors bounce off grid boundaries, red blocks, and higher/equal priority colors.
    Red (2) 2x2 blocks remain unchanged and block extensions.
    Higher priority colors overwrite lower priority ones.
    Extensions continue until forming a loop or being blocked in all directions.
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
        extend_color_with_bounce(output_grid, r, c, color, red_blocks)

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

def extend_color_with_bounce(grid: ColoredGrid, start_row: int, start_col: int, color: int, red_blocks: Set[Tuple[int, int]]):
    """Extend a color diagonally from its starting point with bouncing behavior."""
    initial_direction = (-1, 1) if color in [8, 3] else (-1, -1)
    visited = set()
    queue = deque([(start_row, start_col, initial_direction)])

    while queue:
        r, c, (dx, dy) = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        nr, nc = r + dx, c + dy
        if not is_valid_cell(nr, nc, grid) or (nr, nc) in red_blocks:
            # Bounce
            dx, dy = -dy, -dx
            nr, nc = r + dx, c + dy

        if is_valid_cell(nr, nc, grid) and (nr, nc) not in red_blocks:
            cell_color = grid.get_cell(nr, nc)
            if cell_color == 0 or get_color_priority(color) < get_color_priority(cell_color):
                grid.set_cell(nr, nc, color)
                queue.append((nr, nc, (dx, dy)))
            else:
                # Try bouncing in the other direction
                dx, dy = -dx, -dy
                nr, nc = r + dx, c + dy
                if is_valid_cell(nr, nc, grid) and (nr, nc) not in red_blocks:
                    cell_color = grid.get_cell(nr, nc)
                    if cell_color == 0 or get_color_priority(color) < get_color_priority(cell_color):
                        grid.set_cell(nr, nc, color)
                        queue.append((nr, nc, (dx, dy)))
