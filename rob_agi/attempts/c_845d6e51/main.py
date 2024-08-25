from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict

def solve_845d6e51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies the horizontal gray (5) line dividing the grid.
    2. Leaves all colors above and including the gray line unchanged.
    3. Below the gray line, replaces green (3) regions with colors present above the gray line.
    4. Uses a descending sequence of colors above the gray line for replacements.
    5. Cycles through the replacement colors for different green regions.

    The transformation process:
    - Finds the gray line and collects colors above it.
    - Creates a descending sequence of replacement colors.
    - Processes the grid, replacing green regions below the gray line.
    - Returns the transformed grid.
    """
    # Step 1: Find the gray line and collect colors above it
    gray_line_row = find_gray_line(input_grid)
    colors_above = collect_colors_above(input_grid, gray_line_row)

    # Step 2: Create the replacement color sequence
    replacement_sequence = sorted(list(colors_above), reverse=True)

    # Step 3: Process the grid
    new_grid = input_grid.deep_copy()
    replace_green_regions(new_grid, gray_line_row, replacement_sequence)

    return new_grid

def find_gray_line(grid: ColoredGrid) -> int:
    for i, row in enumerate(grid.values):
        if all(cell == 5 for cell in row):
            return i
    return -1  # If no gray line is found

def collect_colors_above(grid: ColoredGrid, gray_line_row: int) -> Set[int]:
    colors = set()
    for row in grid.values[:gray_line_row]:
        for cell in row:
            if cell not in {0, 3, 5}:
                colors.add(cell)
    return colors

def replace_green_regions(grid: ColoredGrid, gray_line_row: int, replacement_sequence: List[int]):
    processed_cells = set()
    sequence_index = 0

    for r in range(gray_line_row + 1, len(grid.values)):
        for c in range(len(grid.values[r])):
            if grid.values[r][c] == 3 and (r, c) not in processed_cells:
                replacement_color = replacement_sequence[sequence_index % len(replacement_sequence)]
                flood_fill(grid, r, c, 3, replacement_color, processed_cells)
                sequence_index += 1

def flood_fill(grid: ColoredGrid, r: int, c: int, target_color: int, replacement_color: int, processed_cells: Set[Tuple[int, int]]):
    if (r < 0 or r >= len(grid.values) or
        c < 0 or c >= len(grid.values[0]) or
        grid.values[r][c] != target_color or
        (r, c) in processed_cells):
        return

    grid.values[r][c] = replacement_color
    processed_cells.add((r, c))

    flood_fill(grid, r+1, c, target_color, replacement_color, processed_cells)
    flood_fill(grid, r-1, c, target_color, replacement_color, processed_cells)
    flood_fill(grid, r, c+1, target_color, replacement_color, processed_cells)
    flood_fill(grid, r, c-1, target_color, replacement_color, processed_cells)
