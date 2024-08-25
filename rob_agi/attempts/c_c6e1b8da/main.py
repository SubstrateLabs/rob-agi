from rob_agi.colored_grid import ColoredGrid
from collections import defaultdict
from typing import List, Tuple, Dict
import math

def solve_c6e1b8da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adjusting the position and shape of colored regions.
    
    The transformation follows these steps:
    1. Identify and analyze colored regions in the input grid.
    2. Calculate the center of mass for each region.
    3. Align regions to a flexible grid structure based on their center of mass.
    4. Adjust region positions to avoid overlaps and fit within grid boundaries.
    5. Expand or compress regions to better fit the grid structure while maintaining their shape.
    6. Ensure a border of empty space around the edges of the grid.
    7. Optimize space usage by shifting regions to fill large empty areas.
    8. Make final adjustments to ensure proper alignment and no overlaps.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the rules.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    regions = identify_regions(input_grid)
    processed_regions = process_regions(regions, rows, cols)
    
    for color, cells in processed_regions.items():
        for r, c in cells:
            output_grid.set_cell(r, c, color)
    
    output_grid = ensure_border(output_grid)
    output_grid = optimize_space(output_grid, processed_regions)
    
    return output_grid

def identify_regions(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int]]]:
    regions = defaultdict(list)
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            color = grid.get_cell(r, c)
            if color != 0:
                regions[color].append((r, c))
    return regions

def process_regions(regions: Dict[int, List[Tuple[int, int]]], rows: int, cols: int) -> Dict[int, List[Tuple[int, int]]]:
    processed = {}
    grid_size = 3  # Base size of the grid cells for alignment
    
    for color, cells in regions.items():
        center = calculate_center_of_mass(cells)
        new_cells = align_to_grid(cells, center, grid_size, rows, cols)
        new_cells = adjust_region_shape(new_cells, grid_size)
        processed[color] = new_cells
    
    return processed

def calculate_center_of_mass(cells: List[Tuple[int, int]]) -> Tuple[float, float]:
    return sum(r for r, _ in cells) / len(cells), sum(c for _, c in cells) / len(cells)

def align_to_grid(cells: List[Tuple[int, int]], center: Tuple[float, float], grid_size: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    center_r, center_c = center
    aligned_r = round(center_r / grid_size) * grid_size
    aligned_c = round(center_c / grid_size) * grid_size
    
    dr = aligned_r - center_r
    dc = aligned_c - center_c
    
    new_cells = []
    for r, c in cells:
        new_r = min(max(round(r + dr), 0), rows - 1)
        new_c = min(max(round(c + dc), 0), cols - 1)
        new_cells.append((new_r, new_c))
    
    return new_cells

def adjust_region_shape(cells: List[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
    min_r, max_r = min(r for r, _ in cells), max(r for r, _ in cells)
    min_c, max_c = min(c for _, c in cells), max(c for _, c in cells)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    if height > grid_size or width > grid_size:
        scale = min(grid_size / height, grid_size / width)
        center_r, center_c = calculate_center_of_mass(cells)
        
        new_cells = []
        for r, c in cells:
            new_r = round(center_r + (r - center_r) * scale)
            new_c = round(center_c + (c - center_c) * scale)
            new_cells.append((new_r, new_c))
        
        return list(set(new_cells))  # Remove duplicates
    
    return cells

def ensure_border(grid: ColoredGrid) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        grid.set_cell(r, 0, 0)
        grid.set_cell(r, cols-1, 0)
    for c in range(cols):
        grid.set_cell(0, c, 0)
        grid.set_cell(rows-1, c, 0)
    return grid

def optimize_space(grid: ColoredGrid, regions: Dict[int, List[Tuple[int, int]]]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    empty_spaces = find_empty_spaces(grid)
    
    for space in empty_spaces:
        for color, cells in regions.items():
            if can_move_region(grid, cells, space):
                new_cells = move_region(cells, space)
                for r, c in cells:
                    grid.set_cell(r, c, 0)
                for r, c in new_cells:
                    grid.set_cell(r, c, color)
                regions[color] = new_cells
                break
    
    return grid

def find_empty_spaces(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    rows, cols = grid.get_dimensions()
    empty_spaces = []
    
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid.get_cell(r, c) == 0:
                width = 1
                height = 1
                while c + width < cols - 1 and all(grid.get_cell(r, c+i) == 0 for i in range(width+1)):
                    width += 1
                while r + height < rows - 1 and all(grid.get_cell(r+i, c) == 0 for i in range(height+1)):
                    height += 1
                if width > 1 and height > 1:
                    empty_spaces.append((r, c, height, width))
    
    return sorted(empty_spaces, key=lambda x: x[2]*x[3], reverse=True)

def can_move_region(grid: ColoredGrid, cells: List[Tuple[int, int]], space: Tuple[int, int, int, int]) -> bool:
    space_r, space_c, space_h, space_w = space
    region_h = max(r for r, _ in cells) - min(r for r, _ in cells) + 1
    region_w = max(c for _, c in cells) - min(c for _, c in cells) + 1
    
    return region_h <= space_h and region_w <= space_w

def move_region(cells: List[Tuple[int, int]], space: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    space_r, space_c, _, _ = space
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    
    dr = space_r - min_r
    dc = space_c - min_c
    
    return [(r+dr, c+dc) for r, c in cells]
