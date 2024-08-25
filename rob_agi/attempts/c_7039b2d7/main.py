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
    horizontal_lines = [i for i, row in enumerate(input_grid.values) if all(cell != background_color for cell in row)]
    vertical_lines = [j for j in range(len(input_grid.values[0])) if all(row[j] != background_color for row in input_grid.values)]

    # Step 3: Find largest rectangle
    max_area = 0
    best_rect = (0, 0, 0, 0)  # (top, left, bottom, right)

    for top in [-1] + horizontal_lines:
        for bottom in horizontal_lines + [len(input_grid.values)]:
            if bottom <= top + 1:
                continue
            for left in [-1] + vertical_lines:
                for right in vertical_lines + [len(input_grid.values[0])]:
                    if right <= left + 1:
                        continue
                    if is_valid_rectangle(input_grid, top+1, left+1, bottom-1, right-1, background_color):
                        area = (bottom - top - 1) * (right - left - 1)
                        if area > max_area:
                            max_area = area
                            best_rect = (top+1, left+1, bottom-1, right-1)

    # Step 4: Extract and return largest rectangle
    top, left, bottom, right = best_rect
    return ColoredGrid(values=[row[left:right+1] for row in input_grid.values[top:bottom+1]])

def is_valid_rectangle(grid: ColoredGrid, top: int, left: int, bottom: int, right: int, color: int) -> bool:
    return all(grid.values[r][c] == color for r in range(top, bottom+1) for c in range(left, right+1))
