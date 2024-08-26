from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_95a58926(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying vertical gray lines and horizontal gray rows,
    extending horizontal gray lines to full width, and marking intersections with a secondary color.

    1. Identifies the secondary color (non-black, non-gray).
    2. Locates vertical gray line segments.
    3. Identifies rows containing any gray cells.
    4. Creates a new grid with full-width horizontal gray lines.
    5. Adds vertical gray line segments.
    6. Marks intersections of horizontal and vertical gray lines with the secondary color.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid according to the identified pattern.
    """
    rows, cols = input_grid.get_dimensions()
    secondary_color = find_secondary_color(input_grid)
    vertical_segments = find_vertical_segments(input_grid)
    horizontal_rows = find_horizontal_rows(input_grid)

    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Fill horizontal gray lines
    for row in horizontal_rows:
        new_grid.values[row] = [5] * cols

    # Fill vertical gray segments
    for col, start, end in vertical_segments:
        for row in range(start, end + 1):
            new_grid.values[row][col] = 5

    # Mark intersections
    for row in horizontal_rows:
        for col, start, end in vertical_segments:
            if start <= row <= end:
                new_grid.values[row][col] = secondary_color

    return new_grid

def find_secondary_color(grid: ColoredGrid) -> int:
    for row in grid.values:
        for cell in row:
            if cell not in [0, 5]:  # not black or gray
                return cell
    return 0  # default to black if no secondary color found

def find_vertical_segments(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    rows, cols = grid.get_dimensions()
    segments = []
    for col in range(cols):
        start = None
        for row in range(rows):
            if grid.values[row][col] == 5:
                if start is None:
                    start = row
            elif start is not None:
                segments.append((col, start, row - 1))
                start = None
        if start is not None:
            segments.append((col, start, rows - 1))
    return segments

def find_horizontal_rows(grid: ColoredGrid) -> List[int]:
    return [row for row, row_values in enumerate(grid.values) if 5 in row_values]
