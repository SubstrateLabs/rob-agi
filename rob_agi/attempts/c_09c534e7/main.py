from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_09c534e7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying contiguous regions of the same color
    and updating their values based on the region size.
    
    The transformation follows these rules:
    1. Find all contiguous regions of the same color.
    2. For each region, calculate a new value: N = (V + size) % 10, where V is the original value
       and size is the number of squares in the region.
    3. Update all squares in the region with the new value.
    4. Single squares with a value greater than 1 that are not adjacent to same-colored squares
       are left unchanged.
    5. Ensure no value in the output is less than its corresponding value in the input.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    regions = find_regions(input_grid)
    for region in regions:
        if len(region) > 1 or input_grid.values[region[0][0]][region[0][1]] == 1:
            process_region(output_grid, region)
    return output_grid

def find_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    regions = []
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                region = []
                dfs(grid, r, c, grid.values[r][c], region, visited)
                regions.append(region)
    return regions

def dfs(grid: ColoredGrid, r: int, c: int, color: int, region: List[Tuple[int, int]], visited: set):
    if (r, c) in visited or r < 0 or r >= grid.num_rows or c < 0 or c >= grid.num_cols or grid.values[r][c] != color:
        return
    visited.add((r, c))
    region.append((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        dfs(grid, r + dr, c + dc, color, region, visited)

def process_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    color = grid.values[region[0][0]][region[0][1]]
    new_value = max(color, (color + len(region)) % 10)
    for r, c in region:
        grid.values[r][c] = new_value
