from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_9b365c51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by projecting vertical color lines from the left side
    onto sky blue (8) regions on the right side of the grid.

    1. Analyzes the input grid to create an ordered list of unique colors.
    2. Identifies sky blue (8) regions on the right side of the grid.
    3. Creates a deep copy of the input grid and clears the left side.
    4. Fills each sky blue region with colors from the sequence, wrapping around if necessary.
    5. Returns the transformed grid.

    Each sky blue region is filled with a single color, and the color sequence
    continues from one region to the next, wrapping around when it reaches the end.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    # Step 1: Analyze the input grid
    color_sequence = analyze_input_grid(input_grid)

    # Step 2: Identify sky blue regions
    sky_blue_regions = identify_sky_blue_regions(input_grid)

    # Step 3: Create a deep copy of the input grid and clear the left side
    output_grid = input_grid.deep_copy()
    for row in range(output_grid.num_rows):
        for col in range(7):
            output_grid.values[row][col] = 0

    # Step 4: Fill sky blue regions
    color_index = 0
    for region in sky_blue_regions:
        top, left, height, width = region
        fill_color = color_sequence[color_index % len(color_sequence)]
        fill_region(output_grid, top, left, height, width, fill_color)
        color_index += 1

    # Step 5: Return the transformed grid
    return output_grid

def analyze_input_grid(grid: ColoredGrid) -> List[int]:
    color_sequence = []
    for col in range(7):
        for row in range(grid.num_rows):
            color = grid.values[row][col]
            if color != 0 and color not in color_sequence:
                color_sequence.append(color)
    return color_sequence

def identify_sky_blue_regions(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    regions = []
    visited = set()
    for row in range(grid.num_rows):
        for col in range(7, grid.num_cols):
            if grid.values[row][col] == 8 and (row, col) not in visited:
                top, left, height, width = flood_fill(grid, row, col, visited)
                regions.append((top, left, height, width))
    return sorted(regions, key=lambda r: (r[0], r[1]))  # Sort by top row, then left column

def flood_fill(grid: ColoredGrid, row: int, col: int, visited: set) -> Tuple[int, int, int, int]:
    stack = [(row, col)]
    top, left, bottom, right = row, col, row, col
    while stack:
        r, c = stack.pop()
        if (r, c) not in visited and 0 <= r < grid.num_rows and 7 <= c < grid.num_cols and grid.values[r][c] == 8:
            visited.add((r, c))
            top, left = min(top, r), min(left, c)
            bottom, right = max(bottom, r), max(right, c)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((r + dr, c + dc))
    return top, left, bottom - top + 1, right - left + 1

def fill_region(grid: ColoredGrid, top: int, left: int, height: int, width: int, color: int):
    for r in range(top, top + height):
        for c in range(left, left + width):
            if grid.values[r][c] == 8:
                grid.values[r][c] = color
