from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_1d398264(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding non-black cells according to color-specific rules:
    - Blue (1): Expands diagonally up-right
    - Red (2): Expands horizontally to fill the row, then vertically from the ends
    - Green (3): Expands diagonally up-left and down-right
    - Yellow (4): Expands horizontally to fill the entire row
    - Gray (5): Does not expand
    - Magenta (6): Expands diagonally up-left
    - Orange (7): Expands diagonally down-right
    - Sky Blue (8): Expands vertically down, then horizontally at the bottom
    Expansion continues until it hits a boundary or a non-black cell.
    Original non-black cells are preserved, and expansions are applied in order of appearance.
    """
    grid = input_grid.deep_copy().values
    rows, cols = len(grid), len(grid[0])
    original_cells = set((r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 0)

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def expand_diagonal(r: int, c: int, dr: int, dc: int, color: int) -> None:
        while is_valid(r + dr, c + dc) and grid[r + dr][c + dc] == 0:
            r, c = r + dr, c + dc
            grid[r][c] = color

    def expand_horizontal(r: int, color: int) -> None:
        for c in range(cols):
            if (r, c) not in original_cells:
                grid[r][c] = color

    def expand_vertical(r: int, c: int, color: int) -> None:
        for nr in range(r + 1, rows):
            if (nr, c) not in original_cells:
                grid[nr][c] = color
            else:
                break

    def expand_color(r: int, c: int, color: int) -> None:
        if color == 1:  # Blue
            expand_diagonal(r, c, -1, 1, color)
        elif color == 2:  # Red
            expand_horizontal(r, color)
            left = min(i for i in range(cols) if grid[r][i] == color)
            right = max(i for i in range(cols) if grid[r][i] == color)
            for col in [left, right]:
                for row in range(rows):
                    if (row, col) not in original_cells:
                        grid[row][col] = color
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
            if r == rows - 1 or any(grid[r+1][i] == color for i in range(cols)):
                for i in range(cols):
                    if (rows-1, i) not in original_cells:
                        grid[rows-1][i] = color

    for r, c in sorted(original_cells):
        expand_color(r, c, grid[r][c])

    return ColoredGrid(values=grid)
