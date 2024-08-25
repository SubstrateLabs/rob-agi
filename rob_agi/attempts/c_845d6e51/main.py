from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict

def solve_845d6e51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies the horizontal gray (5) line dividing the grid.
    2. Collects unique colors (excluding 0, 3, and 5) from cells up to and including the gray line.
    3. Creates a descending sequence of these collected colors.
    4. Replaces all green (3) cells in the entire grid with colors from the sequence, cycling through it.
    5. Returns the transformed grid.
    """
    # Step 1: Find the gray line
    gray_line_row = next(i for i, row in enumerate(input_grid.values) if all(cell == 5 for cell in row))

    # Step 2: Collect replacement colors
    replacement_colors = sorted(set(cell for row in input_grid.values[:gray_line_row+1] for cell in row 
                                if cell not in {0, 3, 5}), reverse=True)

    # Step 3: Create a new grid
    new_grid = input_grid.deep_copy()

    # Step 4: Replace green cells
    color_index = 0
    for row in range(len(new_grid.values)):
        for col in range(len(new_grid.values[row])):
            if new_grid.values[row][col] == 3:  # If the cell is green
                new_grid.values[row][col] = replacement_colors[color_index]
                color_index = (color_index + 1) % len(replacement_colors)

    # Step 5: Return the modified grid
    return new_grid
