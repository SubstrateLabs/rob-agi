from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict

def solve_845d6e51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies the horizontal gray (5) line dividing the grid.
    2. Collects unique colors (excluding 0, 3, and 5) from cells up to and including the gray line.
    3. Creates a descending sequence of these collected colors.
    4. Replaces green (3) cells with colors from the sequence, cycling through it.
    5. Processes the entire grid up to and including the gray line.
    6. Returns the transformed grid.
    """
    # Step 1: Find the gray line
    gray_line_row = find_gray_line(input_grid)

    # Step 2: Collect replacement colors
    replacement_sequence = collect_replacement_colors(input_grid, gray_line_row)

    # Step 3 & 4: Process the grid
    new_grid = input_grid.deep_copy()
    color_index = 0
    for row in range(gray_line_row + 1):  # Include the gray line
        for col in range(len(new_grid.values[row])):
            if new_grid.values[row][col] == 3:  # Green
                new_grid.values[row][col] = replacement_sequence[color_index]
                color_index = (color_index + 1) % len(replacement_sequence)

    # Step 5: Return the modified grid
    return new_grid

def find_gray_line(grid: ColoredGrid) -> int:
    for i, row in enumerate(grid.values):
        if all(cell == 5 for cell in row):
            return i
    return -1  # If no gray line is found

def collect_replacement_colors(grid: ColoredGrid, gray_line_row: int) -> List[int]:
    colors = set()
    for row in grid.values[:gray_line_row + 1]:
        for cell in row:
            if cell not in {0, 3, 5}:
                colors.add(cell)
    return sorted(list(colors), reverse=True)
