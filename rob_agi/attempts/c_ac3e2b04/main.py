from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac3e2b04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) structures that complement
    existing red (2) and green (3) patterns. The function creates a symmetrical
    blue structure connecting green squares and balancing red lines.

    1. Identifies green squares (3x3 areas with a red center)
    2. Analyzes existing red line patterns
    3. Creates a skeleton blue structure based on grid center and symmetry
    4. Connects green squares to the skeleton
    5. Extends blue lines vertically from the center of green squares
    6. Ensures symmetry across both vertical and horizontal axes
    7. Handles intersections between blue and red lines
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

    create_skeleton_structure(output_grid, green_squares)
    connect_green_squares(output_grid, green_squares)
    extend_vertical_lines(output_grid, green_squares)
    ensure_symmetry(output_grid)
    handle_intersections(output_grid, red_lines)

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

def create_skeleton_structure(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    center_row, center_col = rows // 2, cols // 2
    
    # Create vertical line in the center
    for r in range(rows):
        if grid.get_cell(r, center_col) == 0:
            grid.set_cell(r, center_col, 1)
    
    # Create horizontal line from the center of green squares
    for r, c in green_squares:
        if grid.get_cell(r, center_col) == 0:
            grid.set_cell(r, center_col, 1)
        for col in range(min(c, center_col), max(c, center_col) + 1):
            if grid.get_cell(r, col) == 0:
                grid.set_cell(r, col, 1)

def connect_green_squares(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    center_col = cols // 2
    for r, c in green_squares:
        # Connect horizontally to the center
        for col in range(min(c, center_col), max(c, center_col) + 1):
            if grid.get_cell(r, col) == 0:
                grid.set_cell(r, col, 1)

def extend_vertical_lines(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
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

def handle_intersections(grid: ColoredGrid, red_lines: Tuple[List[int], List[int]]):
    rows, cols = grid.get_dimensions()
    vertical_lines, horizontal_lines = red_lines
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:
                if c in vertical_lines:
                    if r > 0 and grid.get_cell(r-1, c) == 1:
                        grid.set_cell(r+1, c, 1)
                    if r < rows-1 and grid.get_cell(r+1, c) == 1:
                        grid.set_cell(r-1, c, 1)
                if r in horizontal_lines:
                    if c > 0 and grid.get_cell(r, c-1) == 1:
                        grid.set_cell(r, c+1, 1)
                    if c < cols-1 and grid.get_cell(r, c+1) == 1:
                        grid.set_cell(r, c-1, 1)
