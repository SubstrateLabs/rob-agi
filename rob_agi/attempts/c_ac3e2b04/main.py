from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac3e2b04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) structures that complement
    existing red (2) and green (3) patterns. The function creates a symmetrical
    blue structure connecting green squares and balancing red lines.

    1. Identifies green squares (3x3 areas with a red center)
    2. Creates horizontal blue lines connecting green squares
    3. Creates vertical blue lines from green squares to grid edges
    4. Ensures symmetry across both vertical and horizontal axes
    5. Handles intersections between blue and red lines
    6. Extends blue lines to grid edges where appropriate
    7. Preserves all original red and green cells

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with added blue structures
    """
    output_grid = input_grid.deep_copy()
    green_squares = find_green_squares(output_grid)
    
    create_horizontal_structure(output_grid, green_squares)
    create_vertical_structure(output_grid, green_squares)
    ensure_symmetry(output_grid)
    handle_intersections(output_grid)
    extend_to_edges(output_grid)
    
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

def create_horizontal_structure(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    for r, c in green_squares:
        left = right = c
        while left > 0 and grid.get_cell(r, left-1) == 0:
            left -= 1
        while right < cols-1 and grid.get_cell(r, right+1) == 0:
            right += 1
        for col in range(left, right+1):
            if grid.get_cell(r, col) == 0:
                grid.set_cell(r, col, 1)

def create_vertical_structure(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    for r, c in green_squares:
        for row in range(rows):
            if grid.get_cell(row, c) == 0:
                grid.set_cell(row, c, 1)

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
                neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
                blue_neighbors = [n for n in neighbors if 0 <= n[0] < rows and 0 <= n[1] < cols and grid.get_cell(n[0], n[1]) == 1]
                if len(blue_neighbors) >= 2:
                    for nr, nc in neighbors:
                        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
                            grid.set_cell(nr, nc, 1)

def extend_to_edges(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        if 1 in [grid.get_cell(r, c) for c in range(cols)]:
            left = right = None
            for c in range(cols):
                if grid.get_cell(r, c) == 1:
                    if left is None:
                        left = c
                    right = c
            for c in range(left):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 1)
            for c in range(right+1, cols):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 1)
    
    for c in range(cols):
        if 1 in [grid.get_cell(r, c) for r in range(rows)]:
            top = bottom = None
            for r in range(rows):
                if grid.get_cell(r, c) == 1:
                    if top is None:
                        top = r
                    bottom = r
            for r in range(top):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 1)
            for r in range(bottom+1, rows):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 1)
