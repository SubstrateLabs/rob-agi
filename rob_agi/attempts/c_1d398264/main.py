from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def is_valid_position(row: int, col: int, grid: List[List[int]]) -> bool:
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])

def expand(start_row: int, start_col: int, direction: Tuple[int, int], color: int, grid: List[List[int]]) -> None:
    current_row, current_col = start_row, start_col
    while is_valid_position(current_row, current_col, grid):
        if grid[current_row][current_col] != 0 and (current_row, current_col) != (start_row, start_col):
            break
        grid[current_row][current_col] = color
        current_row += direction[0]
        current_col += direction[1]

def solve_1d398264(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding non-black cells according to color-specific rules.
    Each color has a unique expansion pattern:
    - Blue (1): Expands diagonally up-right (limited)
    - Red (2): Expands horizontally to fill the row, then vertically from the ends
    - Green (3): Expands diagonally up-left and down-right (limited)
    - Yellow (4): Expands horizontally to fill the entire row
    - Gray (5): Does not expand
    - Magenta (6): Expands diagonally up-left (limited or full if on edge)
    - Orange (7): Expands diagonally down-right (limited or full if on edge)
    - Sky Blue (8): Expands vertically down to fill the column, then horizontally at the bottom
    The expansion continues until it hits an edge or a non-black cell.
    Original non-black cells are preserved, and expansions are applied in a specific order.
    """
    grid = input_grid.deep_copy().values
    rows, cols = len(grid), len(grid[0])

    def expand_color(r, c, color):
        if color == 1:  # Blue
            expand_diagonal_limited(r, c, -1, 1, color)
        elif color == 2:  # Red
            expand_horizontal_full(r, color)
            expand_vertical_from_ends(r, color)
        elif color == 3:  # Green
            expand_diagonal_limited(r, c, -1, -1, color)
            expand_diagonal_limited(r, c, 1, 1, color)
        elif color == 4:  # Yellow
            expand_horizontal_full(r, color)
        elif color == 6:  # Magenta
            if r == 0 or c == 0:
                expand_diagonal_full(r, c, -1, -1, color)
            else:
                expand_diagonal_limited(r, c, -1, -1, color)
        elif color == 7:  # Orange
            if r == rows - 1 or c == cols - 1:
                expand_diagonal_full(r, c, 1, 1, color)
            else:
                expand_diagonal_limited(r, c, 1, 1, color)
        elif color == 8:  # Sky Blue
            expand_vertical_down(r, c, color)
            expand_horizontal_full(rows - 1, color)

    def expand_diagonal_limited(r, c, dr, dc, color):
        limit = min(rows, cols)
        for i in range(limit):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                grid[nr][nc] = color
            else:
                break

    def expand_diagonal_full(r, c, dr, dc, color):
        while 0 <= r < rows and 0 <= c < cols:
            if grid[r][c] == 0:
                grid[r][c] = color
            r += dr
            c += dc

    def expand_horizontal_full(r, color):
        for c in range(cols):
            if grid[r][c] == 0:
                grid[r][c] = color

    def expand_vertical_from_ends(r, color):
        left_col, right_col = 0, cols - 1
        while left_col < cols and grid[r][left_col] != color:
            left_col += 1
        while right_col >= 0 and grid[r][right_col] != color:
            right_col -= 1
        for c in [left_col, right_col]:
            for nr in range(rows):
                if grid[nr][c] == 0:
                    grid[nr][c] = color

    def expand_vertical_down(r, c, color):
        for nr in range(r, rows):
            if grid[nr][c] == 0:
                grid[nr][c] = color

    non_black_cells = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    expansion_order = [1, 3, 6, 7, 2, 4, 8]

    for color in expansion_order:
        for r, c, cell_color in non_black_cells:
            if cell_color == color:
                expand_color(r, c, color)

    return ColoredGrid(values=grid)
