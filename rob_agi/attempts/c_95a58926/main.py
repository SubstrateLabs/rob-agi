from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_95a58926(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and preserving vertical gray lines,
    modifying horizontal gray lines to include a secondary color at intersections,
    and removing any stray secondary color cells that aren't at intersections.

    1. Analyzes the input grid to find the secondary color and positions of gray lines.
    2. Creates a new grid with vertical gray lines copied from the input.
    3. Processes horizontal gray lines, marking intersections with the secondary color.
    4. Removes any secondary color cells that aren't at valid intersections.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid according to the identified pattern.
    """
    rows, cols = input_grid.get_dimensions()
    secondary_color = find_secondary_color(input_grid)
    vertical_lines = find_vertical_lines(input_grid)
    horizontal_lines = find_horizontal_lines(input_grid, secondary_color)

    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Copy vertical gray lines
    for col in vertical_lines:
        for row in range(rows):
            new_grid.values[row][col] = input_grid.values[row][col]

    # Process horizontal gray lines
    for row in horizontal_lines:
        for col in range(cols):
            if col in vertical_lines:
                new_grid.values[row][col] = secondary_color
            else:
                new_grid.values[row][col] = 5  # gray

    return new_grid

def find_secondary_color(grid: ColoredGrid) -> int:
    for row in grid.values:
        for cell in row:
            if cell not in [0, 5]:  # not black or gray
                return cell
    return 0  # default to black if no secondary color found

def find_vertical_lines(grid: ColoredGrid) -> List[int]:
    rows, cols = grid.get_dimensions()
    return [col for col in range(cols) if all(grid.values[row][col] == 5 for row in range(rows))]

def find_horizontal_lines(grid: ColoredGrid, secondary_color: int) -> List[int]:
    rows, cols = grid.get_dimensions()
    return [row for row in range(rows) if all(grid.values[row][col] in [5, secondary_color] for col in range(cols))]
