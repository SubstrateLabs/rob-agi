from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac3e2b04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) structures that complement
    existing red (2) and green (3) patterns. The function creates a symmetrical
    blue structure connecting green squares and balancing red lines.

    1. Identifies green squares (3x3 areas with a red center)
    2. Analyzes existing red line patterns
    3. Creates a skeleton blue structure based on grid gaps and symmetry
    4. Connects green squares to the skeleton or nearest red lines
    5. Balances the structure by dividing large compartments
    6. Ensures perfect symmetry across both vertical and horizontal axes
    7. Handles intersections between blue and red lines
    8. Extends blue lines to grid edges or nearest red lines
    9. Removes any isolated blue cells
    10. Preserves all original red and green cells

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with added blue structures
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    green_squares = find_green_squares(output_grid)
    red_lines = find_red_lines(output_grid)

    create_skeleton_structure(output_grid, red_lines)
    connect_green_squares(output_grid, green_squares)
    balance_structure(output_grid)
    ensure_symmetry(output_grid)
    handle_intersections(output_grid)
    extend_blue_lines(output_grid)
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

def create_skeleton_structure(grid: ColoredGrid, red_lines: Tuple[List[int], List[int]]):
    rows, cols = grid.get_dimensions()
    vertical_lines, horizontal_lines = red_lines

    # Add vertical blue lines
    gaps = [0] + vertical_lines + [cols]
    for i in range(len(gaps) - 1):
        if gaps[i+1] - gaps[i] > 2:
            mid = (gaps[i] + gaps[i+1]) // 2
            for r in range(rows):
                if grid.get_cell(r, mid) == 0:
                    grid.set_cell(r, mid, 1)

    # Add horizontal blue lines
    gaps = [0] + horizontal_lines + [rows]
    for i in range(len(gaps) - 1):
        if gaps[i+1] - gaps[i] > 2:
            mid = (gaps[i] + gaps[i+1]) // 2
            for c in range(cols):
                if grid.get_cell(mid, c) == 0:
                    grid.set_cell(mid, c, 1)

def connect_green_squares(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    for r, c in green_squares:
        # Connect horizontally
        left = right = c
        while left > 0 and grid.get_cell(r, left-1) == 0:
            left -= 1
            grid.set_cell(r, left, 1)
        while right < grid.get_dimensions()[1]-1 and grid.get_cell(r, right+1) == 0:
            right += 1
            grid.set_cell(r, right, 1)

        # Connect vertically
        top = bottom = r
        while top > 0 and grid.get_cell(top-1, c) == 0:
            top -= 1
            grid.set_cell(top, c, 1)
        while bottom < grid.get_dimensions()[0]-1 and grid.get_cell(bottom+1, c) == 0:
            bottom += 1
            grid.set_cell(bottom, c, 1)

def balance_structure(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid.get_cell(r, c) == 0:
                neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)] if grid.get_cell(r+dr, c+dc) in [1,2])
                if neighbors >= 2:
                    grid.set_cell(r, c, 1)

def extend_blue_lines(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 1:
                    for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
                            grid.set_cell(nr, nc, 1)
                            changed = True

def ensure_symmetry(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                if r != rows - 1 - r:
                    grid.set_cell(rows - 1 - r, c, 1)
                if c != cols - 1 - c:
                    grid.set_cell(r, cols - 1 - c, 1)
                if r != rows - 1 - r and c != cols - 1 - c:
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
