from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_3194b014(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 3194b014 challenge by finding the most significant color in the input grid
    and creating a 3x3 grid filled with that color.

    The solution follows these steps:
    1. Analyze the input grid to count occurrences of each color and find connected regions.
    2. Calculate a significance score for each color based on total count, largest connected region, and centrality.
    3. Identify the most significant color.
    4. Create and return a 3x3 grid filled with the most significant color.

    Args:
    input_grid (ColoredGrid): The input grid to be processed.

    Returns:
    ColoredGrid: A 3x3 grid filled with the most significant color.
    """
    color_analysis = analyze_colors(input_grid)
    most_significant_color = max(color_analysis, key=lambda c: color_analysis[c]['score'])
    return ColoredGrid(values=[[most_significant_color for _ in range(3)] for _ in range(3)])

def analyze_colors(grid: ColoredGrid) -> Dict[int, Dict]:
    rows, cols = grid.get_dimensions()
    color_data = defaultdict(lambda: {'count': 0, 'largest_region': 0, 'centrality': 0})
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited:
                color = grid.get_cell(r, c)
                region_size = dfs(grid, r, c, color, visited)
                color_data[color]['count'] += region_size
                color_data[color]['largest_region'] = max(color_data[color]['largest_region'], region_size)
                color_data[color]['centrality'] += region_size * calculate_centrality(r, c, rows, cols)

    for color in color_data:
        color_data[color]['score'] = (
            color_data[color]['count'] * 0.5 +
            color_data[color]['largest_region'] * 0.3 +
            color_data[color]['centrality'] * 0.2
        )

    return color_data

def dfs(grid: ColoredGrid, r: int, c: int, color: int, visited: set) -> int:
    if (r, c) in visited or not (0 <= r < grid.num_rows and 0 <= c < grid.num_cols) or grid.get_cell(r, c) != color:
        return 0
    visited.add((r, c))
    size = 1
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        size += dfs(grid, r + dr, c + dc, color, visited)
    return size

def calculate_centrality(r: int, c: int, rows: int, cols: int) -> float:
    center_r, center_c = rows / 2, cols / 2
    distance = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
    max_distance = ((rows / 2) ** 2 + (cols / 2) ** 2) ** 0.5
    return 1 - (distance / max_distance)
