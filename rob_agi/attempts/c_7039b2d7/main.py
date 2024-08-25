from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_7039b2d7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by finding the largest rectangle within a single cell
    of the grid pattern.

    1. Identifies the background color (most frequent color).
    2. Detects the grid pattern by finding horizontal and vertical lines.
    3. Identifies cells within the grid pattern.
    4. Finds the largest rectangle of background color within a single cell.
    5. Extracts and returns this largest rectangle as a new ColoredGrid.
    """
    rows, cols = input_grid.get_dimensions()
    background_color = Counter([cell for row in input_grid.values for cell in row]).most_common(1)[0][0]

    # Detect vertical and horizontal lines
    vertical_lines = [j for j in range(cols) if any(input_grid.values[i][j] != background_color for i in range(rows))]
    horizontal_lines = [i for i in range(rows) if any(input_grid.values[i][j] != background_color for j in range(cols))]

    # Find cells
    cells = []
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            top = horizontal_lines[i] + 1
            left = vertical_lines[j] + 1
            bottom = horizontal_lines[i + 1]
            right = vertical_lines[j + 1]
            cells.append((top, left, bottom, right))

    # Find largest rectangle within a single cell
    max_area = 0
    best_rect = None
    for cell in cells:
        top, left, bottom, right = cell
        area = (bottom - top) * (right - left)
        if area > max_area and all(input_grid.values[i][j] == background_color 
                                   for i in range(top, bottom) 
                                   for j in range(left, right)):
            max_area = area
            best_rect = cell

    # Extract and return largest rectangle
    if best_rect:
        top, left, bottom, right = best_rect
        return ColoredGrid(values=[row[left:right] for row in input_grid.values[top:bottom]])
    else:
        # If no valid rectangle found, return a 1x1 grid with the background color
        return ColoredGrid(values=[[background_color]])
