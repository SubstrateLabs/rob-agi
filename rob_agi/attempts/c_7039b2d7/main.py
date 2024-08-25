from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_7039b2d7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by extracting a single cell from the grid pattern.

    1. Identifies the background color (most frequent color).
    2. Detects the grid pattern by finding horizontal and vertical lines.
    3. Calculates the dimensions of a single cell.
    4. Locates a valid cell to extract.
    5. Extracts and returns the cell as a new ColoredGrid.
    """
    rows, cols = input_grid.get_dimensions()
    background_color = Counter([cell for row in input_grid.values for cell in row]).most_common(1)[0][0]

    # Detect vertical and horizontal lines
    vertical_lines = [j for j in range(cols) if any(input_grid.values[i][j] != background_color for i in range(rows))]
    horizontal_lines = [i for i in range(rows) if any(input_grid.values[i][j] != background_color for j in range(cols))]

    # If no lines are detected, return a 1x1 grid with the background color
    if not vertical_lines and not horizontal_lines:
        return ColoredGrid(values=[[background_color]])

    # Calculate cell dimensions
    cell_width = min(b - a for a, b in zip(vertical_lines, vertical_lines[1:])) - 1
    cell_height = min(b - a for a, b in zip(horizontal_lines, horizontal_lines[1:])) - 1

    # Locate a valid cell to extract
    for i in range(rows - cell_height):
        for j in range(cols - cell_width):
            if all(input_grid.values[r][c] == background_color 
                   for r in range(i, i + cell_height + 1) 
                   for c in range(j, j + cell_width + 1)):
                return ColoredGrid(values=[row[j:j+cell_width] for row in input_grid.values[i:i+cell_height]])

    # If no valid cell is found, return a 1x1 grid with the background color
    return ColoredGrid(values=[[background_color]])
