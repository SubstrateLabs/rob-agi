from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import Counter

def solve_95a58926(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying vertical and horizontal lines,
    and marking intersections with a specific color pattern.

    1. Analyzes the input grid to identify the line color and dot color.
    2. Locates vertical and horizontal line segments.
    3. Determines the intersection color as the minimum of line and dot colors.
    4. Creates a new grid with the transformed pattern:
       - Fills the background with black (0).
       - Draws horizontal and vertical lines with the line color.
       - Marks intersections with the intersection color.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid according to the identified pattern.
    """
    rows, cols = input_grid.get_dimensions()
    color_counts = count_colors(input_grid)
    line_color, dot_color = identify_colors(color_counts)
    intersection_color = min(line_color, dot_color)
    
    vertical_lines = find_vertical_lines(input_grid, line_color)
    horizontal_rows = find_horizontal_rows(input_grid, line_color)

    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Fill horizontal lines
    for row in horizontal_rows:
        new_grid.values[row] = [line_color] * cols

    # Fill vertical lines and mark intersections
    for col in vertical_lines:
        for row in range(rows):
            if row in horizontal_rows:
                new_grid.values[row][col] = intersection_color
            else:
                new_grid.values[row][col] = line_color

    return new_grid

def count_colors(grid: ColoredGrid) -> Dict[int, int]:
    return Counter(cell for row in grid.values for cell in row if cell != 0)

def identify_colors(color_counts: Dict[int, int]) -> Tuple[int, int]:
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_colors[0][0], sorted_colors[1][0]

def find_vertical_lines(grid: ColoredGrid, line_color: int) -> Set[int]:
    rows, cols = grid.get_dimensions()
    return {col for col in range(cols) if any(grid.values[row][col] == line_color for row in range(rows))}

def find_horizontal_rows(grid: ColoredGrid, line_color: int) -> Set[int]:
    return {row for row, row_values in enumerate(grid.values) if any(cell == line_color for cell in row_values)}
