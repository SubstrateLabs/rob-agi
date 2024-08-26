from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b365c51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by projecting vertical color lines from the left side
    onto sky blue (8) regions on the right side of the grid.

    1. Analyzes the input grid to identify unique colors and their vertical positions.
    2. Identifies sky blue (8) regions on the right side of the grid.
    3. Creates a deep copy of the input grid.
    4. Clears the left side of the grid (first 7 columns).
    5. Fills each sky blue region with a color based on its vertical position.
    6. Returns the transformed grid.

    The color sequence is determined by the vertical order of colors on the left side.
    Each sky blue region is filled with a single color based on its vertical position.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    # Step 1: Analyze the input grid
    color_sequence, color_positions = analyze_input_grid(input_grid)

    # Step 2: Identify sky blue regions
    sky_blue_regions = identify_sky_blue_regions(input_grid)

    # Step 3: Create a deep copy of the input grid
    output_grid = input_grid.deep_copy()

    # Step 4: Clear the left side of the grid
    for row in range(output_grid.num_rows):
        for col in range(7):
            output_grid.values[row][col] = 0

    # Step 5: Fill sky blue regions
    for region in sky_blue_regions:
        top, left, height, width = region
        color_index = find_color_index(top, color_positions)
        fill_color = color_sequence[color_index % len(color_sequence)]
        fill_region(output_grid, top, left, height, width, fill_color)

    # Step 6: Return the transformed grid
    return output_grid

def analyze_input_grid(grid: ColoredGrid) -> Tuple[List[int], List[int]]:
    colors = []
    positions = []
    for col in range(7):
        for row in range(grid.num_rows):
            color = grid.values[row][col]
            if color != 0 and color not in colors:
                colors.append(color)
                positions.append(row)
    return colors, positions

def identify_sky_blue_regions(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    regions = []
    visited = set()
    for row in range(grid.num_rows):
        for col in range(7, grid.num_cols):
            if grid.values[row][col] == 8 and (row, col) not in visited:
                top, left, height, width = flood_fill(grid, row, col, visited)
                regions.append((top, left, height, width))
    return sorted(regions, key=lambda r: r[0])  # Sort by top row

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

def find_color_index(top: int, color_positions: List[int]) -> int:
    return next((i for i, pos in enumerate(color_positions) if pos > top), 0)

def fill_region(grid: ColoredGrid, top: int, left: int, height: int, width: int, color: int):
    for r in range(top, top + height):
        for c in range(left, left + width):
            if grid.values[r][c] == 8:
                grid.values[r][c] = color
