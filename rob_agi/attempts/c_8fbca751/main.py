from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_8fbca751(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by outlining blue shapes with red.
    
    This function identifies all connected blue (8) regions in the input grid,
    then outlines each region with red (2) cells. The outline includes diagonally
    adjacent cells but does not extend beyond the grid boundaries or overwrite
    existing non-black cells.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid with blue shapes outlined in red.
    """
    grid = input_grid.deep_copy()
    blue_regions = find_blue_regions(grid)
    
    for region in blue_regions:
        adjacent_cells = get_adjacent_cells(region, grid.get_dimensions())
        for row, col in adjacent_cells:
            if grid.values[row][col] == 0:  # Only change black cells to red
                grid.values[row][col] = 2
    
    return grid

def find_blue_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    visited = set()
    regions = []
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8 and (r, c) not in visited:
                region = []
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == 8:
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                stack.append((nr, nc))
                regions.append(region)
    return regions

def get_adjacent_cells(region: List[Tuple[int, int]], dimensions: Tuple[int, int]) -> Set[Tuple[int, int]]:
    rows, cols = dimensions
    adjacent = set()
    for r, c in region:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in region:
                    adjacent.add((nr, nc))
    return adjacent
