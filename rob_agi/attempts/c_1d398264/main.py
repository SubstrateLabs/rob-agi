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
    - Blue (1): Expands diagonally up-right
    - Red (2): Expands horizontally, then vertically at the ends
    - Green (3): Expands diagonally down-right and up-left
    - Yellow (4): Expands horizontally in both directions
    - Gray (5): Does not expand
    - Magenta (6): Expands diagonally up-left
    - Orange (7): Expands diagonally down-right
    - Sky Blue (8): Expands vertically down, then horizontally at the bottom
    The expansion continues until it hits an edge or a non-black cell.
    Original non-black cells are preserved, and expansions are applied in a specific order.
    """
    grid = input_grid.deep_copy().values
    rows, cols = len(grid), len(grid[0])

    def expand_color(r, c, color):
        if color == 1:  # Blue
            expand_diagonal(r, c, -1, 1, color)
        elif color == 2:  # Red
            expand_horizontal(r, c, color)
            expand_vertical(r, c, color)
        elif color == 3:  # Green
            expand_diagonal(r, c, 1, 1, color)
            expand_diagonal(r, c, -1, -1, color)
        elif color == 4:  # Yellow
            expand_horizontal(r, c, color)
        elif color == 6:  # Magenta
            expand_diagonal(r, c, -1, -1, color)
        elif color == 7:  # Orange
            expand_diagonal(r, c, 1, 1, color)
        elif color == 8:  # Sky Blue
            expand_vertical(r, c, color)
            expand_horizontal(r, rows-1, color)

    def expand_diagonal(r, c, dr, dc, color):
        while 0 <= r < rows and 0 <= c < cols and (grid[r][c] == 0 or grid[r][c] == color):
            grid[r][c] = color
            r += dr
            c += dc

    def expand_horizontal(r, c, color):
        for dc in [-1, 1]:
            nc = c
            while 0 <= nc < cols and (grid[r][nc] == 0 or grid[r][nc] == color):
                grid[r][nc] = color
                nc += dc

    def expand_vertical(r, c, color):
        for dr in [-1, 1]:
            nr = r
            while 0 <= nr < rows and (grid[nr][c] == 0 or grid[nr][c] == color):
                grid[nr][c] = color
                nr += dr

    non_black_cells = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    expansion_order = [1, 3, 2, 4, 6, 7, 8]

    for color in expansion_order:
        for r, c, cell_color in non_black_cells:
            if cell_color == color:
                expand_color(r, c, color)

    return ColoredGrid(values=grid)
