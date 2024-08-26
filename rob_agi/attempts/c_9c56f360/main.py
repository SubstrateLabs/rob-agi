from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9c56f360(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving green (3) regions as far left as possible
    while maintaining contact with at least one sky blue (8) square and avoiding
    overlap with other green regions.

    1. Identifies all green regions in the grid.
    2. For each green region, finds the leftmost valid position.
    3. Moves the green region to the new position if different from the original.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with green regions moved.
    """
    grid = input_grid.deep_copy()
    green_regions = grid.find_connected_regions(3)
    
    for region in green_regions:
        current_position = min(region)  # Leftmost, topmost point of the region
        new_position = find_leftmost_valid_position(grid, region, current_position[0])
        
        if new_position != current_position:
            move_region(grid, region, new_position)
    
    return grid

def find_leftmost_valid_position(grid: ColoredGrid, region: List[Tuple[int, int]], start_row: int) -> Tuple[int, int]:
    rows, cols = grid.get_dimensions()
    region_height = max(r for r, _ in region) - min(r for r, _ in region) + 1
    region_width = max(c for _, c in region) - min(c for _, c in region) + 1
    
    for r in range(max(0, start_row - region_height), min(rows - region_height + 1, start_row + region_height)):
        for c in range(cols - region_width + 1):
            if is_valid_position(grid, region, r, c):
                return (r, c)
    
    return min(region)  # Return original position if no valid new position found

def is_valid_position(grid: ColoredGrid, region: List[Tuple[int, int]], row: int, col: int) -> bool:
    rows, cols = grid.get_dimensions()
    region_height = max(r for r, _ in region) - min(r for r, _ in region) + 1
    region_width = max(c for _, c in region) - min(c for _, c in region) + 1
    
    # Check if the region fits within the grid
    if row + region_height > rows or col + region_width > cols:
        return False
    
    # Check if the region overlaps with other green regions or is adjacent to sky blue
    adjacent_to_sky_blue = False
    for r in range(row, row + region_height):
        for c in range(col, col + region_width):
            if grid.get_cell(r, c) == 3:  # Green
                return False
            if not adjacent_to_sky_blue:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 8:
                        adjacent_to_sky_blue = True
                        break
    
    return adjacent_to_sky_blue

def move_region(grid: ColoredGrid, region: List[Tuple[int, int]], new_position: Tuple[int, int]):
    # Clear the original region
    for r, c in region:
        grid.set_cell(r, c, 0)
    
    # Set the new region
    offset_r, offset_c = new_position[0] - min(r for r, _ in region), new_position[1] - min(c for _, c in region)
    for r, c in region:
        grid.set_cell(r + offset_r, c + offset_c, 3)
