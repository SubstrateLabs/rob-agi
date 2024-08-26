from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_bd14c3bf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by changing blue shapes with vulnerable sections to red,
    while preserving blue shapes that are robust. A blue cell is considered vulnerable if it has fewer
    than two blue neighbors in its 8-cell neighborhood (including diagonals). If any cell in a connected
    blue region is vulnerable, the entire region is changed to red. Original red shapes and black cells
    remain unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with vulnerable blue shapes changed to red.
    """
    output_grid = input_grid.deep_copy()
    vulnerable_cells = find_vulnerable_blue_regions(output_grid)
    regions_to_change = set()

    for cell in vulnerable_cells:
        if cell not in regions_to_change:
            connected_region = get_connected_region(output_grid, cell)
            regions_to_change.update(connected_region)

    for row in range(output_grid.get_dimensions()[0]):
        for col in range(output_grid.get_dimensions()[1]):
            if (row, col) in regions_to_change and output_grid.get_cell(row, col) == 1:
                output_grid.set_cell(row, col, 2)

    return output_grid

def find_vulnerable_blue_regions(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    vulnerable_cells = set()

    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                blue_neighbors = sum(
                    1 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                    if (dr != 0 or dc != 0) and 0 <= r + dr < rows and 0 <= c + dc < cols
                    and grid.get_cell(r + dr, c + dc) == 1
                )
                if blue_neighbors < 2:
                    vulnerable_cells.add((r, c))

    return vulnerable_cells

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
