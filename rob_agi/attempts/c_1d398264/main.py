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
    cells_to_expand = sorted([(r, c, grid[r][c]) for r, c in original_cells])

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

    def expand_vertical(r: int, c: int, dr: int, color: int) -> None:
        while is_valid(r + dr, c) and grid[r + dr][c] == 0:
            r += dr
            grid[r][c] = color

    def expand_red(r: int, c: int, color: int) -> None:
        expand_horizontal(r, color)
        left = min(i for i in range(cols) if grid[r][i] == color)
        right = max(i for i in range(cols) if grid[r][i] == color)
        for col in [left, right]:
            expand_vertical(r - 1, col, -1, color)
            expand_vertical(r + 1, col, 1, color)

    def expand_sky_blue(r: int, c: int, color: int) -> None:
        expand_vertical(r, c, 1, color)
        bottom_row = max(i for i in range(rows) if grid[i][c] == color)
        for col in range(cols):
            if (bottom_row, col) not in original_cells:
                grid[bottom_row][col] = color

    def expand_color(r: int, c: int, color: int) -> None:
        if color == 1:  # Blue
            expand_diagonal(r, c, -1, 1, color)
        elif color == 2:  # Red
            expand_red(r, c, color)
        elif color == 3:  # Green
            expand_diagonal(r, c, -1, -1, color)
            expand_diagonal(r, c, 1, 1, color)
        elif color == 4:  # Yellow
            expand_horizontal(r, color)
        elif color == 5:  # Gray
            pass  # Does not expand
        elif color == 6:  # Magenta
            expand_diagonal(r, c, -1, -1, color)
        elif color == 7:  # Orange
            expand_diagonal(r, c, 1, 1, color)
        elif color == 8:  # Sky Blue
            expand_sky_blue(r, c, color)

    for r, c, color in cells_to_expand:
        expand_color(r, c, color)

    return ColoredGrid(values=grid)
