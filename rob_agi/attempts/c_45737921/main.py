from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_45737921(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 45737921 challenge by rotating colors within each connected region of the grid.
    
    The solution works as follows:
    1. Create a deep copy of the input grid.
    2. Find all non-black connected regions in the grid.
    3. For each region, rotate its colors based on their numerical order.
    4. Return the modified grid.
    """
    output_grid = input_grid.deep_copy()
    regions = output_grid.find_connected_regions(color=None)  # Find all non-black regions
    for region in regions:
        rotate_region_colors(output_grid, region)
    return output_grid

def rotate_region_colors(grid: ColoredGrid, region: List[Tuple[int, int]]):
    unique_colors = get_unique_colors(grid, region)
    color_map = create_rotation_mapping(unique_colors)
    apply_rotation(grid, region, color_map)

def get_unique_colors(grid: ColoredGrid, region: List[Tuple[int, int]]) -> List[int]:
    colors = set(grid.get_cell(r, c) for r, c in region)
    colors.discard(0)  # Remove black if present
    return sorted(colors)

def create_rotation_mapping(colors: List[int]) -> Dict[int, int]:
    return {colors[i]: colors[(i + 1) % len(colors)] for i in range(len(colors))}

def apply_rotation(grid: ColoredGrid, region: List[Tuple[int, int]], color_map: Dict[int, int]):
    for r, c in region:
        cell_value = grid.get_cell(r, c)
        if cell_value in color_map:
            grid.set_cell(r, c, color_map[cell_value])
