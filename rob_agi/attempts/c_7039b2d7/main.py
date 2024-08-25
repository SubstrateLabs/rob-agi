from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_7039b2d7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by finding the largest rectangle of
    background color in the input grid.

    1. Identifies the background color (most frequent color).
    2. Creates a binary matrix where background color is 1 and others are 0.
    3. Uses dynamic programming to find the largest rectangle of 1s.
    4. Extracts and returns this largest rectangle as a new ColoredGrid.
    """
    # Step 1: Identify background color
    background_color = Counter([cell for row in input_grid.values for cell in row]).most_common(1)[0][0]

    # Step 2: Create binary matrix
    binary_matrix = [[1 if cell == background_color else 0 for cell in row] for row in input_grid.values]

    # Step 3: Find largest rectangle
    max_area, best_rect = find_largest_rectangle(binary_matrix)

    # Step 4: Extract and return largest rectangle
    top, left, bottom, right = best_rect
    return ColoredGrid(values=[row[left:right] for row in input_grid.values[top:bottom]])

def find_largest_rectangle(matrix: List[List[int]]) -> Tuple[int, Tuple[int, int, int, int]]:
    rows, cols = len(matrix), len(matrix[0])
    heights = [[0] * cols for _ in range(rows)]
    lefts = [[0] * cols for _ in range(rows)]
    max_area = 0
    best_rect = (0, 0, 0, 0)  # (top, left, bottom, right)

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                heights[i][j] = heights[i-1][j] + 1 if i > 0 else 1
                lefts[i][j] = lefts[i][j-1] + 1 if j > 0 else 1
                
                width = lefts[i][j]
                for height in range(heights[i][j], 0, -1):
                    area = height * width
                    if area > max_area:
                        max_area = area
                        best_rect = (i-height+1, j-width+1, i+1, j+1)
                    width = min(width, lefts[i-height+1][j])
            else:
                heights[i][j] = lefts[i][j] = 0

    return max_area, best_rect
