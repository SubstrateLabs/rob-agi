from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_95a58926(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying vertical and horizontal lines,
    and marking intersections with a specific color pattern.

    1. Identifies the secondary color (non-black, non-gray).
    2. Identifies the intersection color (usually gray, but can be different).
    3. Locates vertical and horizontal line segments.
    4. Creates a new grid with the transformed pattern.
    5. Fills horizontal lines with the secondary color.
    6. Fills vertical lines with the intersection color.
    7. Marks intersections of horizontal and vertical lines with the intersection color.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid according to the identified pattern.
    """
    rows, cols = input_grid.get_dimensions()
    secondary_color = find_secondary_color(input_grid)
    intersection_color = find_intersection_color(input_grid, secondary_color)
    vertical_lines = find_vertical_lines(input_grid)
    horizontal_rows = find_horizontal_rows(input_grid)

    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Fill horizontal lines
    for row in horizontal_rows:
        new_grid.values[row] = [secondary_color] * cols

    # Fill vertical lines and mark intersections
    for col in vertical_lines:
        for row in range(rows):
            if row in horizontal_rows:
                new_grid.values[row][col] = intersection_color
            else:
                new_grid.values[row][col] = secondary_color

    return new_grid

def find_secondary_color(grid: ColoredGrid) -> int:
    for row in grid.values:
        for cell in row:
            if cell not in [0, 5]:  # not black or gray
                return cell
    return 0  # default to black if no secondary color found

def find_intersection_color(grid: ColoredGrid, secondary_color: int) -> int:
    for row in grid.values:
        for cell in row:
            if cell not in [0, secondary_color]:
                return cell
    return 5  # default to gray if no intersection color found

def find_vertical_lines(grid: ColoredGrid) -> Set[int]:
    rows, cols = grid.get_dimensions()
    vertical_lines = set()
    for col in range(cols):
        if all(grid.values[row][col] != 0 for row in range(rows)):
            vertical_lines.add(col)
    return vertical_lines

def find_horizontal_rows(grid: ColoredGrid) -> Set[int]:
    return {row for row, row_values in enumerate(grid.values) if any(cell != 0 for cell in row_values)}
