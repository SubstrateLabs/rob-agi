from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_bd14c3bf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by changing blue shapes to red if they are vulnerable
    or isolated. A blue region is considered vulnerable if any of its cells have fewer than two blue
    neighbors in their 8-cell neighborhood. A blue region is considered isolated if the average number
    of blue cells in a 5x5 area around each cell of the region is below a certain threshold.
    Original red shapes and black cells remain unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with vulnerable or isolated blue shapes changed to red.
    """
    output_grid = input_grid.deep_copy()
    blue_regions = find_blue_regions(output_grid)
    regions_to_change = []

    for region in blue_regions:
        if is_vulnerable(output_grid, region) or is_isolated(output_grid, region):
            regions_to_change.append(region)

    for region in regions_to_change:
        for row, col in region:
            output_grid.set_cell(row, col, 2)

    return output_grid

def find_blue_regions(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    blue_regions = []

    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1 and (r, c) not in visited:
                region = get_connected_region(grid, (r, c))
                blue_regions.append(region)
                visited.update(region)

    return blue_regions

def is_vulnerable(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> bool:
    for r, c in region:
        if count_blue_neighbors(grid, r, c) < 2:
            return True
    return False

def is_isolated(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> bool:
    total_blue_count = sum(count_extended_blue_neighborhood(grid, r, c) for r, c in region)
    average_blue_count = total_blue_count / len(region)
    return average_blue_count < 8  # Threshold can be adjusted

def count_blue_neighbors(grid: ColoredGrid, row: int, col: int) -> int:
    rows, cols = grid.get_dimensions()
    return sum(
        1 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
        if (dr != 0 or dc != 0) and 0 <= row + dr < rows and 0 <= col + dc < cols
        and grid.get_cell(row + dr, col + dc) == 1
    )

def count_extended_blue_neighborhood(grid: ColoredGrid, row: int, col: int) -> int:
    rows, cols = grid.get_dimensions()
    count = 0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if 0 <= row + dr < rows and 0 <= col + dc < cols:
                if grid.get_cell(row + dr, col + dc) == 1:
                    count += 1
    return count

def get_connected_region(grid: ColoredGrid, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    connected_region = set()
    stack = [start]

    while stack:
        r, c = stack.pop()
        if (r, c) not in connected_region and grid.get_cell(r, c) == 1:
            connected_region.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    stack.append((nr, nc))

    return connected_region
