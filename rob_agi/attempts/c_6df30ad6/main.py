from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_6df30ad6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding the largest gray region and replacing it with the most frequent non-gray color.
    
    1. Finds the largest connected region of gray (5) in the input grid.
    2. Counts the frequency of all non-gray, non-black colors.
    3. Determines the most frequent color (or highest value if tied).
    4. Creates a new grid with the largest gray region filled with the new color.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    largest_gray_region = find_largest_gray_region(input_grid)
    color_frequencies = count_color_frequencies(input_grid)
    new_color = determine_new_color(color_frequencies)
    
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    for x, y in largest_gray_region:
        output_grid.values[x][y] = new_color
    
    return output_grid

def find_largest_gray_region(grid: ColoredGrid) -> List[Tuple[int, int]]:
    def dfs(x: int, y: int) -> List[Tuple[int, int]]:
        if not (0 <= x < grid.num_rows and 0 <= y < grid.num_cols) or grid.values[x][y] != 5 or (x, y) in visited:
            return []
        visited.add((x, y))
        region = [(x, y)]
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(dfs(x + dx, y + dy))
        return region

    visited = set()
    largest_region = []
    for i in range(grid.num_rows):
        for j in range(grid.num_cols):
            if grid.values[i][j] == 5 and (i, j) not in visited:
                region = dfs(i, j)
                if len(region) > len(largest_region):
                    largest_region = region
    return largest_region

def count_color_frequencies(grid: ColoredGrid) -> Dict[int, int]:
    frequencies = {}
    for row in grid.values:
        for cell in row:
            if cell not in [0, 5]:
                frequencies[cell] = frequencies.get(cell, 0) + 1
    return frequencies

def determine_new_color(color_frequencies: Dict[int, int]) -> int:
    if not color_frequencies:
        return 1  # Default to blue if no other colors are present
    max_frequency = max(color_frequencies.values())
    max_colors = [color for color, freq in color_frequencies.items() if freq == max_frequency]
    return max(max_colors)
