from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_bbb1b8b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x9 grid into a 4x4 grid based on the following rules:
    
    1. Extract the left half (first 4 columns) of the input grid.
    2. Fill internal spaces with corresponding colors from the right half (last 4 columns).
    3. If the resulting grid doesn't touch all edges, revert to the left half.
    4. Fill any remaining empty spaces with the most common non-zero color.
    5. Return the resulting 4x4 grid as a ColoredGrid object.
    """
    left_half = extract_left_half(input_grid)
    right_half = extract_right_half(input_grid)
    
    completed_grid = complete_shape(left_half, right_half)
    
    if not touches_all_edges(completed_grid):
        completed_grid = left_half
    
    fill_empty_spaces(completed_grid)
    
    return ColoredGrid(values=completed_grid)

def extract_left_half(input_grid: ColoredGrid) -> List[List[int]]:
    return [row[:4] for row in input_grid.values]

def extract_right_half(input_grid: ColoredGrid) -> List[List[int]]:
    return [row[5:] for row in input_grid.values]

def touches_all_edges(grid: List[List[int]]) -> bool:
    return (
        any(grid[0][c] != 0 for c in range(4)) and  # top edge
        any(grid[3][c] != 0 for c in range(4)) and  # bottom edge
        any(grid[r][0] != 0 for r in range(4)) and  # left edge
        any(grid[r][3] != 0 for r in range(4))      # right edge
    )

def complete_shape(left_half: List[List[int]], right_half: List[List[int]]) -> List[List[int]]:
    completed = [row[:] for row in left_half]
    
    for r in range(4):
        for c in range(4):
            if completed[r][c] == 0:
                mirror_c = 3 - c
                if right_half[r][mirror_c] != 0:
                    completed[r][c] = right_half[r][mirror_c]
    
    return completed

def fill_empty_spaces(grid: List[List[int]]) -> None:
    main_color = get_main_color(grid)
    for r in range(4):
        for c in range(4):
            if grid[r][c] == 0:
                grid[r][c] = main_color

def get_main_color(grid: List[List[int]]) -> int:
    colors = [cell for row in grid for cell in row if cell != 0]
    return max(set(colors), key=colors.count) if colors else 1  # Default to blue if no colors
