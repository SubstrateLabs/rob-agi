from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac3e2b04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) structures that complement
    existing red (2) and green (3) patterns. The function creates a symmetrical
    design by adding horizontal and vertical blue lines through green squares.

    1. Identifies green squares (3x3 areas with a red center)
    2. Creates horizontal blue lines through green squares
    3. Creates vertical blue lines extending from green squares
    4. Handles special cases for single green squares or aligned green squares
    5. Ensures symmetry of the blue structure
    6. Connects isolated blue segments
    7. Removes any isolated blue cells
    8. Preserves all original red and green cells

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with added blue structures
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Find green squares
    green_squares = find_green_squares(output_grid)

    # Create horizontal and vertical blue lines
    create_blue_lines(output_grid, green_squares)

    # Handle special cases
    handle_special_cases(output_grid, green_squares)

    # Ensure symmetry
    ensure_symmetry(output_grid)

    # Connect isolated segments and clean up
    connect_and_cleanup(output_grid)

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

def create_blue_lines(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    for r, c in green_squares:
        # Horizontal line
        for col in range(cols):
            if grid.get_cell(r, col) == 0:
                grid.set_cell(r, col, 1)
        # Vertical line
        for row in range(rows):
            if grid.get_cell(row, c) == 0:
                grid.set_cell(row, c, 1)

def handle_special_cases(grid: ColoredGrid, green_squares: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    if len(green_squares) == 1:
        r, c = green_squares[0]
        for row in range(rows):
            if grid.get_cell(row, cols // 2) == 0:
                grid.set_cell(row, cols // 2, 1)
    elif all(r == green_squares[0][0] for r, _ in green_squares):
        for row in range(rows):
            if grid.get_cell(row, cols // 2) == 0:
                grid.set_cell(row, cols // 2, 1)

def ensure_symmetry(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                if grid.get_cell(rows - 1 - r, c) == 0:
                    grid.set_cell(rows - 1 - r, c, 1)
                if grid.get_cell(r, cols - 1 - c) == 0:
                    grid.set_cell(r, cols - 1 - c, 1)

def connect_and_cleanup(grid: ColoredGrid):
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
                            break
                    if changed:
                        break
            if changed:
                break

    # Remove isolated blue cells
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]
                                if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 1)
                if neighbors == 0:
                    grid.set_cell(r, c, 0)
