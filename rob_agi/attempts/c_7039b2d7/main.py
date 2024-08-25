from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_7039b2d7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by finding the largest rectangle of
    background color between horizontal and vertical lines of a different color.

    1. Identifies the background color (most frequent color).
    2. Finds all horizontal and vertical lines of non-background color.
    3. Searches for the largest rectangle of background color between these lines.
    4. Extracts and returns this largest rectangle as a new ColoredGrid.
    """
    # Step 1: Identify background color
    background_color = Counter([cell for row in input_grid.values for cell in row]).most_common(1)[0][0]

    # Step 2: Find horizontal and vertical lines
    horizontal_lines = [-1] + [i for i, row in enumerate(input_grid.values) if any(cell != background_color for cell in row)] + [len(input_grid.values)]
    vertical_lines = [-1] + [j for j in range(len(input_grid.values[0])) if any(row[j] != background_color for row in input_grid.values)] + [len(input_grid.values[0])]

    # Step 3: Find largest rectangle
    max_area = 0
    best_rect = (0, 0, 0, 0)  # (top, left, bottom, right)

    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            top, bottom = horizontal_lines[i] + 1, horizontal_lines[i+1]
            left, right = vertical_lines[j] + 1, vertical_lines[j+1]
            
            if is_valid_rectangle(input_grid, top, left, bottom-1, right-1, background_color):
                area = (bottom - top) * (right - left)
                if area > max_area:
                    max_area = area
                    best_rect = (top, left, bottom-1, right-1)

    # Step 4: Extract and return largest rectangle
    top, left, bottom, right = best_rect
    return ColoredGrid(values=[row[left:right+1] for row in input_grid.values[top:bottom+1]])

def is_valid_rectangle(grid: ColoredGrid, top: int, left: int, bottom: int, right: int, color: int) -> bool:
    return all(grid.values[r][c] == color for r in range(top, bottom+1) for c in range(left, right+1))
