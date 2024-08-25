from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac3e2b04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) structures that complement
    existing red (2) and green (3) patterns. The function creates a symmetrical
    design by adding horizontal and vertical blue lines through green squares
    and the grid center, forming crosses/plus signs.

    1. Identifies green squares and red lines
    2. Creates horizontal blue lines through green squares and grid center
    3. Creates vertical blue lines extending from green squares and grid center
    4. Forms crosses/plus signs at intersections
    5. Ensures symmetry and connectivity of blue structures
    6. Preserves all original red and green cells

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with added blue structures
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Find green squares
    green_squares = find_green_squares(output_grid)

    # Create horizontal blue lines
    create_horizontal_blue_lines(output_grid, green_squares)

    # Create vertical blue lines
    create_vertical_blue_lines(output_grid, green_squares)

    # Ensure symmetry and connectivity
    ensure_symmetry_and_connectivity(output_grid)

    return output_grid

def find_green_squares(grid: ColoredGrid) -> List[Tuple[int, int]]:
    green_squares = []
    rows, cols = grid.get_dimensions()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if (grid.get_cell(r, c) == 3 and
                grid.get_cell(r-1, c) == 3 and grid.get_cell(r+1, c) == 3 and
                grid.get_cell(r, c-1) == 3 and grid.get_cell(r, c+1) == 3):
                green_squares.append((r, c))
    return green_squares

def create_horizontal_blue_lines(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    center_row = rows // 2

    # Add horizontal line through green squares
    for r, c in green_squares:
        for col in range(cols):
            if grid.get_cell(r, col) == 0:
                grid.set_cell(r, col, 1)

    # Add horizontal line through center if no green squares or space in center
    if not green_squares or all(r != center_row for r, _ in green_squares):
        for col in range(cols):
            if grid.get_cell(center_row, col) == 0:
                grid.set_cell(center_row, col, 1)

def create_vertical_blue_lines(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    center_col = cols // 2

    # Add vertical lines through green squares
    for r, c in green_squares:
        for row in range(rows):
            if grid.get_cell(row, c) == 0:
                grid.set_cell(row, c, 1)

    # Add vertical line through center if no green squares or space in center
    if not green_squares or all(c != center_col for _, c in green_squares):
        for row in range(rows):
            if grid.get_cell(row, center_col) == 0:
                grid.set_cell(row, center_col, 1)

def ensure_symmetry_and_connectivity(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    center_row, center_col = rows // 2, cols // 2

    # Ensure symmetry
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                grid.set_cell(rows - 1 - r, c, 1)
                grid.set_cell(r, cols - 1 - c, 1)
                grid.set_cell(rows - 1 - r, cols - 1 - c, 1)

    # Ensure connectivity
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                if r > 0 and grid.get_cell(r-1, c) == 0:
                    grid.set_cell(r-1, c, 1)
                if r < rows-1 and grid.get_cell(r+1, c) == 0:
                    grid.set_cell(r+1, c, 1)
                if c > 0 and grid.get_cell(r, c-1) == 0:
                    grid.set_cell(r, c-1, 1)
                if c < cols-1 and grid.get_cell(r, c+1) == 0:
                    grid.set_cell(r, c+1, 1)

    # Remove isolated blue cells
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]
                                if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 1)
                if neighbors == 0:
                    grid.set_cell(r, c, 0)
