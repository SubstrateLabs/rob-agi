from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_6df30ad6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding the largest gray region and replacing it with the highest-valued non-gray, non-black color.
    
    1. Finds the largest connected region of gray (5) in the input grid.
    2. Identifies the highest-valued color that is not gray (5) or black (0).
    3. Creates a new grid with the largest gray region filled with the highest-valued color.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    largest_gray_region = find_largest_gray_region(input_grid)
    highest_color = find_highest_color(input_grid)
    
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    for x, y in largest_gray_region:
        output_grid.values[x][y] = highest_color
    
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

def find_highest_color(grid: ColoredGrid) -> int:
    highest_color = 0
    for row in grid.values:
        for cell in row:
            if cell not in [0, 5] and cell > highest_color:
                highest_color = cell
    return highest_color if highest_color > 0 else 1  # Default to blue (1) if no other colors are present
