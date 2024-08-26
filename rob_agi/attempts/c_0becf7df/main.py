from rob_agi.colored_grid import ColoredGrid
from itertools import combinations
from typing import Set, List, Tuple

def solve_0becf7df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by swapping two pairs of colors
    while maintaining the connectedness of regions. The solution involves:
    1. Preserving the top-left 2x2 square, except for one color
    2. Identifying colors in the top-left 2x2 square and the rest of the grid
    3. Finding all possible combinations of color swaps, including one color from the top-left
    4. Trying each combination and checking if it produces the desired result
    5. Applying the successful swap combination to the input grid
    6. Ensuring the connectedness of the swapped regions
    7. Returning the transformed grid
    """
    top_left_colors = get_top_left_colors(input_grid)
    other_colors = get_other_colors(input_grid)
    corner_color = input_grid.values[0][0]
    
    for top_left_color in top_left_colors - {corner_color}:
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
    all_colors = set(grid.values[r][c] for r in range(len(grid.values)) for c in range(len(grid.values[0])))
    return all_colors - get_top_left_colors(grid)

def swap_regions(grid: ColoredGrid, color1: int, color2: int):
    for r in range(len(grid.values)):
        for c in range(len(grid.values[0])):
            if grid.values[r][c] == color1:
                grid.values[r][c] = color2
            elif grid.values[r][c] == color2:
                grid.values[r][c] = color1

def is_valid_solution(input_grid: ColoredGrid, new_grid: ColoredGrid) -> bool:
    if not is_top_left_mostly_preserved(input_grid, new_grid):
        return False
    
    if not are_regions_connected(new_grid):
        return False
    
    return calculate_difference(input_grid, new_grid) > 0

def is_top_left_mostly_preserved(grid1: ColoredGrid, grid2: ColoredGrid) -> bool:
    differences = sum(1 for i in range(2) for j in range(2) if grid1.values[i][j] != grid2.values[i][j])
    return differences <= 1 and grid1.values[0][0] == grid2.values[0][0]

def are_regions_connected(grid: ColoredGrid) -> bool:
    for color in set(cell for row in grid.values for cell in row):
        if color != 0:  # Ignore black (0) regions
            regions = grid.find_connected_regions(color)
            if len(regions) > 1:
                return False
    return True

def calculate_difference(grid1: ColoredGrid, grid2: ColoredGrid) -> int:
    return sum(1 for r in range(len(grid1.values)) for c in range(len(grid1.values[0])) 
               if grid1.values[r][c] != grid2.values[r][c])
