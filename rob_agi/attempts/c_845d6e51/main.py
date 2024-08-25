from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict

def solve_845d6e51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies the horizontal gray (5) line dividing the grid.
    2. Leaves all colors above and including the gray line unchanged.
    3. Below the gray line, replaces green (3) regions with other colors present in the grid.
    4. Uses a color promotion sequence to determine replacement colors.
    5. Ensures different green regions are replaced with different colors when possible.

    The transformation process:
    - Analyzes the input grid to find unique colors.
    - Creates a mapping for green replacement based on present colors.
    - Processes the grid, replacing green regions below the gray line.
    - Returns the transformed grid.
    """
    # Step 1: Analyze the input grid
    gray_line_row = find_gray_line(input_grid)
    present_colors = set(color for row in input_grid.values for color in row) - {0, 3, 5}

    # Step 2: Define color promotion sequence
    color_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

    # Step 3: Create mapping for green replacement
    replacement_map = create_replacement_map(present_colors, color_sequence)

    # Step 4: Process the grid
    new_grid = input_grid.deep_copy()
    replace_green_regions(new_grid, gray_line_row, replacement_map)

    return new_grid

def find_gray_line(grid: ColoredGrid) -> int:
    for i, row in enumerate(grid.values):
        if all(cell == 5 for cell in row):
            return i
    return -1  # If no gray line is found

def create_replacement_map(present_colors: Set[int], color_sequence: List[int]) -> List[int]:
    start_index = color_sequence.index(3) + 1
    replacement_colors = []
    for i in range(len(color_sequence)):
        color = color_sequence[(start_index + i) % len(color_sequence)]
        if color in present_colors:
            replacement_colors.append(color)
        if len(replacement_colors) >= 2:
            break
    return replacement_colors

def replace_green_regions(grid: ColoredGrid, gray_line_row: int, replacement_map: List[int]):
    visited = set()
    replacement_index = 0

    for r in range(gray_line_row + 1, len(grid.values)):
        for c in range(len(grid.values[r])):
            if grid.values[r][c] == 3 and (r, c) not in visited:
                replacement_color = replacement_map[replacement_index % len(replacement_map)]
                flood_fill(grid, r, c, 3, replacement_color, visited)
                replacement_index += 1

def flood_fill(grid: ColoredGrid, r: int, c: int, target_color: int, replacement_color: int, visited: Set[Tuple[int, int]]):
    if (r < 0 or r >= len(grid.values) or
        c < 0 or c >= len(grid.values[0]) or
        grid.values[r][c] != target_color or
        (r, c) in visited):
        return

    grid.values[r][c] = replacement_color
    visited.add((r, c))

    flood_fill(grid, r+1, c, target_color, replacement_color, visited)
    flood_fill(grid, r-1, c, target_color, replacement_color, visited)
    flood_fill(grid, r, c+1, target_color, replacement_color, visited)
    flood_fill(grid, r, c-1, target_color, replacement_color, visited)
