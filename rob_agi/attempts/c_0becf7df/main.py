from rob_agi.colored_grid import ColoredGrid
from itertools import combinations
from typing import List, Tuple, Dict

def solve_0becf7df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by swapping two pairs of colors
    while maintaining the connectedness of regions. The solution involves:
    1. Identifying all color regions in the input grid
    2. Finding the best pair of color pairs to swap
    3. Swapping the selected color pairs
    4. Ensuring the connectedness of the swapped regions
    5. Returning the transformed grid
    """
    color_regions = find_color_regions(input_grid)
    colors = list(color_regions.keys())
    best_swap = find_best_color_swap(input_grid, colors)
    
    new_grid = input_grid.deep_copy()
    for color1, color2 in best_swap:
        swap_regions(new_grid, color1, color2)
        ensure_connectedness(new_grid, color1)
        ensure_connectedness(new_grid, color2)
    
    return new_grid

def find_color_regions(grid: ColoredGrid) -> Dict[int, List[List[Tuple[int, int]]]]:
    color_regions = {}
    for color in range(10):  # 0 to 9
        regions = grid.find_connected_regions(color)
        if regions:
            color_regions[color] = regions
    return color_regions

def calculate_difference(grid1: ColoredGrid, grid2: ColoredGrid) -> int:
    return sum(1 for r in range(len(grid1.values)) for c in range(len(grid1.values[0])) if grid1.values[r][c] != grid2.values[r][c])

def swap_regions(grid: ColoredGrid, color1: int, color2: int):
    color1_cells = [(r, c) for r in range(len(grid.values)) for c in range(len(grid.values[0])) if grid.values[r][c] == color1]
    color2_cells = [(r, c) for r in range(len(grid.values)) for c in range(len(grid.values[0])) if grid.values[r][c] == color2]
    
    for (r1, c1), (r2, c2) in zip(color1_cells, color2_cells):
        grid.values[r1][c1] = color2
        grid.values[r2][c2] = color1

def ensure_connectedness(grid: ColoredGrid, color: int):
    regions = grid.find_connected_regions(color)
    if len(regions) <= 1:
        return
    
    main_region = max(regions, key=len)
    for region in regions:
        if region != main_region:
            for r, c in region:
                grid.values[r][c] = 0  # Set to black (empty)
            
            # Find the closest cell to connect
            closest_cell = min(region, key=lambda cell: min(abs(cell[0] - mr[0]) + abs(cell[1] - mr[1]) for mr in main_region))
            r, c = closest_cell
            grid.values[r][c] = color

def find_best_color_swap(grid: ColoredGrid, colors: List[int]) -> List[Tuple[int, int]]:
    color_pairs = list(combinations(colors, 2))
    pair_combinations = list(combinations(color_pairs, 2))
    
    best_swap = None
    min_difference = float('inf')
    
    for (color1, color2), (color3, color4) in pair_combinations:
        test_grid = grid.deep_copy()
        swap_regions(test_grid, color1, color2)
        swap_regions(test_grid, color3, color4)
        ensure_connectedness(test_grid, color1)
        ensure_connectedness(test_grid, color2)
        ensure_connectedness(test_grid, color3)
        ensure_connectedness(test_grid, color4)
        
        difference = calculate_difference(grid, test_grid)
        if difference < min_difference:
            min_difference = difference
            best_swap = [(color1, color2), (color3, color4)]
    
    return best_swap
