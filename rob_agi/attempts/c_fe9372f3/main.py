from rob_agi.colored_grid import ColoredGrid

def solve_fe9372f3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following pattern:
    1. Locate the red (2) cross in the input grid.
    2. Create a 5x5 central pattern around the red cross with blue (1) diagonals and sky blue (8) diamond.
    3. Extend the pattern vertically for the full height of the grid.
    4. Create a horizontal pattern with sky blue (8) line and yellow (4) squares every 4 cells.
    5. Ensure the pattern is centered on the red cross and extends to the grid edges.
    6. Preserve the original red cross.
    7. Fill remaining cells with black (0).
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Find the center of the red cross
    center_row, center_col = find_red_cross_center(input_grid)

    # Create the 5x5 central pattern
    create_central_pattern(output_grid, center_row, center_col)

    # Extend the pattern vertically
    extend_vertical_pattern(output_grid, center_row, center_col)

    # Extend the pattern horizontally and add repeating pattern
    extend_horizontal_pattern(output_grid, center_row, center_col)

    # Copy the red cross from the input grid
    copy_red_cross(input_grid, output_grid)

    return output_grid

def find_red_cross_center(grid: ColoredGrid) -> tuple[int, int]:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:
                return r, c
    raise ValueError("Red cross not found in the input grid")

def create_central_pattern(grid: ColoredGrid, center_row: int, center_col: int):
    for r in range(center_row - 2, center_row + 3):
        for c in range(center_col - 2, center_col + 3):
            if r == center_row and c == center_col:
                continue  # Skip the center cell (part of the red cross)
            if abs(r - center_row) + abs(c - center_col) == 2:
                grid.set_cell(r, c, 8)  # Sky blue diamond
            elif abs(r - center_row) == abs(c - center_col) == 2:
                grid.set_cell(r, c, 1)  # Blue diagonals

def extend_vertical_pattern(grid: ColoredGrid, center_row: int, center_col: int):
    rows, _ = grid.get_dimensions()
    for r in range(rows):
        if r == center_row:
            continue  # Skip the center row (handled by horizontal pattern)
        dist = abs(r - center_row)
        if dist % 4 == 0:
            grid.set_cell(r, center_col, 8)  # Sky blue vertical line
        if dist % 2 == 1:
            grid.set_cell(r, center_col - 1, 1)  # Left blue diagonal
            grid.set_cell(r, center_col + 1, 1)  # Right blue diagonal

def extend_horizontal_pattern(grid: ColoredGrid, center_row: int, center_col: int):
    _, cols = grid.get_dimensions()
    for c in range(cols):
        if c == center_col:
            continue  # Skip the center cell (part of the red cross)
        if (c - center_col) % 4 == 0:
            grid.set_cell(center_row, c, 4)  # Yellow squares
        else:
            grid.set_cell(center_row, c, 8)  # Sky blue line

def copy_red_cross(input_grid: ColoredGrid, output_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 2:
                output_grid.set_cell(r, c, 2)
