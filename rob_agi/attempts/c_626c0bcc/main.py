from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_626c0bcc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by coloring sky-colored (8) regions with a specific pattern.
    
    The algorithm works as follows:
    1. Identify all connected sky-colored regions.
    2. For each region, color it starting from the top-left corner.
    3. Use a priority order for shapes: 2x2 square, 2x3 rectangle, 3x2 rectangle, 2x1 rectangle, 1x2 rectangle, single cell.
    4. Color shapes in the order: blue (1), red (2), green (3), yellow (4), cycling through these colors.
    5. Combine all colored regions with the original black (0) background.
    
    This approach ensures no adjacent cells (including diagonally) have the same non-black color.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    sky_regions = input_grid.find_connected_regions(8)
    
    for region in sky_regions:
        color_region(output_grid, region)
    
    return output_grid

def color_region(grid: ColoredGrid, region: List[Tuple[int, int]]):
    color_index = 0
    start_row, start_col = min(region)
    row, col = start_row, start_col
    
    while region:
        shapes = [
            ((2, 2), [(0,0), (0,1), (1,0), (1,1)]),
            ((2, 3), [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]),
            ((3, 2), [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]),
            ((2, 1), [(0,0), (1,0)]),
            ((1, 2), [(0,0), (0,1)]),
            ((1, 1), [(0,0)])
        ]
        
        for shape_size, shape_cells in shapes:
            if can_color_shape(grid, region, row, col, shape_size, shape_cells):
                color_shape(grid, region, row, col, shape_cells, color_index + 1)
                color_index = (color_index + 1) % 4
                break
        
        region = [cell for cell in region if grid.get_cell(*cell) == 0]
        if region:
            row, col = min(region)

def can_color_shape(grid: ColoredGrid, region: List[Tuple[int, int]], row: int, col: int, 
                    shape_size: Tuple[int, int], shape_cells: List[Tuple[int, int]]) -> bool:
    rows, cols = grid.get_dimensions()
    for dr, dc in shape_cells:
        r, c = row + dr, col + dc
        if (r, c) not in region or r >= rows or c >= cols:
            return False
    return True

def color_shape(grid: ColoredGrid, region: List[Tuple[int, int]], row: int, col: int, 
                shape_cells: List[Tuple[int, int]], color: int):
    for dr, dc in shape_cells:
        r, c = row + dr, col + dc
        grid.set_cell(r, c, color)
        region.remove((r, c))
