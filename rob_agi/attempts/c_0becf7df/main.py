from rob_agi.colored_grid import ColoredGrid
from itertools import combinations, product
from typing import List, Tuple, Dict, Set

def solve_0becf7df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by swapping two pairs of colors
    while maintaining the connectedness of regions. The solution involves:
    1. Preserving the top-left 2x2 square
    2. Identifying colors in the top-left 2x2 square and the rest of the grid
    3. Finding all possible combinations of color swaps, including one color from the top-left
    4. Trying each combination and checking if it produces the desired result
    5. Applying the successful swap combination to the input grid
    6. Ensuring the connectedness of the swapped regions
    7. Returning the transformed grid
    """
    top_left_colors = get_top_left_colors(input_grid)
    other_colors = get_other_colors(input_grid)
    
    for top_left_color in list(top_left_colors)[1:]:  # Exclude the corner color
        for other_color in other_colors:
            remaining_colors = other_colors - {other_color}
            for color1, color2 in combinations(remaining_colors, 2):
                new_grid = input_grid.deep_copy()
                swap_regions(new_grid, top_left_color, other_color)
                swap_regions(new_grid, color1, color2)
                
                if is_valid_solution(input_grid, new_grid):
                    return new_grid
    
    return input_grid

def get_top_left_colors(grid: ColoredGrid) -> Set[int]:
    return {grid.values[i][j] for i in range(2) for j in range(2)}

def get_other_colors(grid: ColoredGrid) -> Set[int]:
    return set(grid.values[r][c] for r in range(len(grid.values)) for c in range(len(grid.values[0]))) - get_top_left_colors(grid)

def swap_regions(grid: ColoredGrid, color1: int, color2: int):
    for r in range(len(grid.values)):
        for c in range(len(grid.values[0])):
            if r >= 2 or c >= 2:
                if grid.values[r][c] == color1:
                    grid.values[r][c] = color2
                elif grid.values[r][c] == color2:
                    grid.values[r][c] = color1

def is_valid_solution(input_grid: ColoredGrid, new_grid: ColoredGrid) -> bool:
    if not is_top_left_preserved(input_grid, new_grid):
        return False
    
    if not are_regions_connected(new_grid):
        return False
    
    return calculate_difference(input_grid, new_grid) > 0

def is_top_left_preserved(grid1: ColoredGrid, grid2: ColoredGrid) -> bool:
    return all(grid1.values[i][j] == grid2.values[i][j] for i in range(2) for j in range(2))

def are_regions_connected(grid: ColoredGrid) -> bool:
    for color in range(10):
        regions = grid.find_connected_regions(color)
        if len(regions) > 1:
            return False
    return True

def calculate_difference(grid1: ColoredGrid, grid2: ColoredGrid) -> int:
    return sum(1 for r in range(len(grid1.values)) for c in range(len(grid1.values[0])) 
               if (r >= 2 or c >= 2) and grid1.values[r][c] != grid2.values[r][c])
