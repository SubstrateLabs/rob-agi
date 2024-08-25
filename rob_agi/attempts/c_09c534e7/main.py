from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_09c534e7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying contiguous regions of the same color
    and updating their values based on the region size and color progression.
    
    The transformation follows these rules:
    1. Find all contiguous regions of the same color.
    2. For each region, determine a new color based on the original color and region size.
    3. Update the interior of the region with the new color.
    4. Set the border of the region to either the original color or the next color in the progression.
    5. Expand isolated higher-value cells to small regions.
    6. Ensure no value in the output is less than its corresponding value in the input.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    regions = find_regions(input_grid)
    for region in regions:
        process_region(output_grid, region)
    expand_isolated_cells(output_grid)
    ensure_no_decrease(input_grid, output_grid)
    return output_grid

def find_regions(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    regions = []
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                region = set()
                dfs(grid, r, c, grid.values[r][c], region, visited)
                regions.append(region)
    return regions

def dfs(grid: ColoredGrid, r: int, c: int, color: int, region: Set[Tuple[int, int]], visited: Set[Tuple[int, int]]):
    if (r, c) in visited or r < 0 or r >= grid.num_rows or c < 0 or c >= grid.num_cols or grid.values[r][c] != color:
        return
    visited.add((r, c))
    region.add((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        dfs(grid, r + dr, c + dc, color, region, visited)

def process_region(grid: ColoredGrid, region: Set[Tuple[int, int]]):
    color = grid.values[list(region)[0][0]][list(region)[0][1]]
    new_color = determine_new_color(color, len(region))
    border = find_border(grid, region)
    for r, c in region - border:
        grid.values[r][c] = new_color
    for r, c in border:
        grid.values[r][c] = min(color + 1, 9)

def determine_new_color(color: int, size: int) -> int:
    if size <= 3:
        return min(color + 1, 9)
    elif size <= 8:
        return min(color + 2, 9)
    else:
        return min(color + 3, 9)

def find_border(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    border = set()
    for r, c in region:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region and 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                border.add((r, c))
                break
    return border

def expand_isolated_cells(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] > 1 and is_isolated(grid, r, c):
                expand_cell(grid, r, c)

def is_isolated(grid: ColoredGrid, r: int, c: int) -> bool:
    color = grid.values[r][c]
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.values[nr][nc] == color:
            return False
    return True

def expand_cell(grid: ColoredGrid, r: int, c: int):
    color = grid.values[r][c]
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                if dr == 0 and dc == 0:
                    grid.values[nr][nc] = color
                else:
                    grid.values[nr][nc] = max(grid.values[nr][nc], color - 1)

def ensure_no_decrease(input_grid: ColoredGrid, output_grid: ColoredGrid):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            output_grid.values[r][c] = max(output_grid.values[r][c], input_grid.values[r][c])
