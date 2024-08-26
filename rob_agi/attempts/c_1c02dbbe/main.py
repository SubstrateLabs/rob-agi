from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_1c02dbbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions based on seed points.
    
    The function identifies seed points (non-gray, non-black cells), sorts them
    in reading order (top-to-bottom, left-to-right), then determines and fills
    zones of influence for each seed point. The transformation respects the
    precedence of earlier seed points and preserves the original structure
    including black cells and borders.
    
    Steps:
    1. Find and sort seed points
    2. Determine zones of influence for each seed point
    3. Fill zones with respective colors
    4. Preserve original black cells and borders
    5. Maintain remaining gray areas not claimed by any zone
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with expanded color regions
    """
    output_grid = input_grid.deep_copy()
    seed_points = find_and_sort_seed_points(output_grid)
    zones = determine_zones(output_grid, seed_points)
    fill_zones(output_grid, zones)
    preserve_original_structure(output_grid, input_grid)
    return output_grid

def find_and_sort_seed_points(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    seed_points = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color not in [0, 5]:  # Not black or gray
                seed_points.append((color, r, c))
    return sorted(seed_points, key=lambda x: (x[1], x[2]))  # Sort by row, then column

def determine_zones(grid: ColoredGrid, seed_points: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int, int, int]]:
    zones = []
    rows, cols = grid.get_dimensions()
    for color, row, col in seed_points:
        right = next((c for c in range(col+1, cols) if grid.get_cell(row, c) not in [0, 5]), cols-1)
        bottom = next((r for r in range(row+1, rows) if grid.get_cell(r, col) not in [0, 5]), rows-1)
        zones.append((color, row, col, bottom, right))
    return zones

def fill_zones(grid: ColoredGrid, zones: List[Tuple[int, int, int, int, int]]):
    for color, top, left, bottom, right in zones:
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if grid.get_cell(r, c) in [0, 5]:  # Only fill black or gray cells
                    grid.set_cell(r, c, color)

def preserve_original_structure(output_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = output_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 0:  # If originally black
                output_grid.set_cell(r, c, 0)  # Keep it black
    # Ensure border is black
    for r in range(rows):
        output_grid.set_cell(r, 0, 0)
        output_grid.set_cell(r, cols-1, 0)
    for c in range(cols):
        output_grid.set_cell(0, c, 0)
        output_grid.set_cell(rows-1, c, 0)
