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
    - Blue (1): Expands diagonally up-right (unlimited)
    - Red (2): Expands horizontally to fill the row, then vertically from the ends
    - Green (3): Expands diagonally up-left and down-right (unlimited)
    - Yellow (4): Expands horizontally to fill the entire row
    - Gray (5): Does not expand
    - Magenta (6): Expands diagonally up-left (unlimited)
    - Orange (7): Expands diagonally down-right (unlimited)
    - Sky Blue (8): Expands vertically down to fill the column, then horizontally at the bottom
    The expansion continues until it hits an edge or a non-black cell.
    Original non-black cells are preserved, and expansions are applied in order of appearance.
    """
    grid = input_grid.deep_copy().values
    rows, cols = len(grid), len(grid[0])

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def expand_color(r, c, color):
        if color == 1:  # Blue
            expand_diagonal(r, c, -1, 1, color)
        elif color == 2:  # Red
            expand_horizontal(r, color)
            expand_vertical_from_ends(r, color)
        elif color == 3:  # Green
            expand_diagonal(r, c, -1, -1, color)
            expand_diagonal(r, c, 1, 1, color)
        elif color == 4:  # Yellow
            expand_horizontal(r, color)
        elif color == 6:  # Magenta
            expand_diagonal(r, c, -1, -1, color)
        elif color == 7:  # Orange
            expand_diagonal(r, c, 1, 1, color)
        elif color == 8:  # Sky Blue
            expand_vertical(r, c, color)
            expand_horizontal_bottom(color)

    def expand_diagonal(r, c, dr, dc, color):
        nr, nc = r + dr, c + dc
        while is_valid(nr, nc) and grid[nr][nc] == 0:
            grid[nr][nc] = color
            nr, nc = nr + dr, nc + dc

    def expand_horizontal(r, color):
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

    def expand_vertical(r, c, color):
        for nr in range(r + 1, rows):
            if grid[nr][c] == 0:
                grid[nr][c] = color

    def expand_horizontal_bottom(color):
        for c in range(cols):
            if grid[rows-1][c] == color:
                for nc in range(cols):
                    if grid[rows-1][nc] == 0:
                        grid[rows-1][nc] = color

    original_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 0]

    for r, c in original_cells:
        expand_color(r, c, grid[r][c])

    return ColoredGrid(values=grid)
