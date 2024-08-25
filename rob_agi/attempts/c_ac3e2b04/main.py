from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac3e2b04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) structures that complement
    existing red (2) and green (3) patterns. The function creates a minimal,
    symmetrical blue structure connecting green squares and balancing red lines.

    1. Identifies green squares (3x3 areas with a red center)
    2. Analyzes existing red line patterns
    3. Creates a minimal blue structure to connect green squares or balance red lines
    4. Extends blue lines to grid edges or nearest red lines
    5. Ensures symmetry across both vertical and horizontal axes
    6. Handles intersections between blue and red lines
    7. Removes any isolated blue cells
    8. Preserves all original red and green cells

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with added blue structures
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    green_squares = find_green_squares(output_grid)
    red_lines = find_red_lines(output_grid)

    create_blue_structure(output_grid, green_squares, red_lines)
    extend_blue_lines(output_grid)
    ensure_symmetry(output_grid)
    handle_intersections(output_grid)
    remove_isolated_blue_cells(output_grid)

    return output_grid

def find_green_squares(grid: ColoredGrid) -> List[Tuple[int, int]]:
    green_squares = []
    rows, cols = grid.get_dimensions()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if (grid.get_cell(r, c) == 2 and
                all(grid.get_cell(r+dr, c+dc) == 3 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)])):
                green_squares.append((r, c))
    return green_squares

def find_red_lines(grid: ColoredGrid) -> Tuple[List[int], List[int]]:
    rows, cols = grid.get_dimensions()
    vertical_lines = [c for c in range(cols) if all(grid.get_cell(r, c) == 2 for r in range(rows))]
    horizontal_lines = [r for r in range(rows) if all(grid.get_cell(r, c) == 2 for c in range(cols))]
    return vertical_lines, horizontal_lines

def create_blue_structure(grid: ColoredGrid, green_squares: List[Tuple[int, int]], red_lines: Tuple[List[int], List[int]]):
    if green_squares:
        connect_green_squares(grid, green_squares)
    else:
        balance_red_lines(grid, red_lines)

def connect_green_squares(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    min_r = min(r for r, _ in green_squares)
    max_r = max(r for r, _ in green_squares)
    min_c = min(c for _, c in green_squares)
    max_c = max(c for _, c in green_squares)

    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid.get_cell(r, c) == 0:
                grid.set_cell(r, c, 1)

def balance_red_lines(grid: ColoredGrid, red_lines: Tuple[List[int], List[int]]):
    vertical_lines, horizontal_lines = red_lines
    rows, cols = grid.get_dimensions()

    if len(vertical_lines) > len(horizontal_lines):
        mid_row = rows // 2
        for c in range(cols):
            if grid.get_cell(mid_row, c) == 0:
                grid.set_cell(mid_row, c, 1)
    else:
        mid_col = cols // 2
        for r in range(rows):
            if grid.get_cell(r, mid_col) == 0:
                grid.set_cell(r, mid_col, 1)

def extend_blue_lines(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                # Extend horizontally
                for dc in [-1, 1]:
                    nc = c + dc
                    while 0 <= nc < cols and grid.get_cell(r, nc) == 0:
                        grid.set_cell(r, nc, 1)
                        nc += dc
                # Extend vertically
                for dr in [-1, 1]:
                    nr = r + dr
                    while 0 <= nr < rows and grid.get_cell(nr, c) == 0:
                        grid.set_cell(nr, c, 1)
                        nr += dr

def ensure_symmetry(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                grid.set_cell(rows - 1 - r, c, 1)
                grid.set_cell(r, cols - 1 - c, 1)
                grid.set_cell(rows - 1 - r, cols - 1 - c, 1)

def handle_intersections(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:
                if any(grid.get_cell(r + dr, c + dc) == 1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]):
                    for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
                            grid.set_cell(nr, nc, 1)

def remove_isolated_blue_cells(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]
                                if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 1)
                if neighbors == 0:
                    grid.set_cell(r, c, 0)
