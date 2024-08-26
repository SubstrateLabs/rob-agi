from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_c62e2108(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the c62e2108 challenge by replicating and expanding hollow square patterns.
    
    The function identifies hollow squares in the input grid, determines their color and bounds,
    then replicates the pattern across the width of the grid within the identified bounds.
    It preserves original frame lines and copies remaining elements from the input grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with replicated and expanded square patterns.
    """
    squares, top_row, bottom_row, square_color = find_squares_and_bounds(input_grid)
    new_grid = [[0 for _ in range(len(input_grid.values[0]))] for _ in range(len(input_grid.values))]
    replicate_pattern(new_grid, top_row, bottom_row, square_color)
    preserve_frame_lines(new_grid, input_grid, top_row, bottom_row)
    copy_remaining_elements(new_grid, input_grid, bottom_row)
    return ColoredGrid(values=new_grid)

def find_squares_and_bounds(grid: ColoredGrid) -> Tuple[List[Tuple[int, int]], int, int, int]:
    squares = []
    top_row = float('inf')
    bottom_row = -1
    square_color = None
    
    for i in range(len(grid.values) - 3):
        for j in range(len(grid.values[0]) - 3):
            if is_hollow_square(grid, i, j):
                squares.append((i, j))
                top_row = min(top_row, i)
                bottom_row = max(bottom_row, i + 3)
                square_color = grid.values[i][j]
    
    # Check for horizontal lines of the same color at the bottom
    for i in range(len(grid.values) - 1, bottom_row, -1):
        if any(grid.values[i][j] == square_color for j in range(len(grid.values[0]))):
            bottom_row = i
            break
    
    return squares, top_row, bottom_row, square_color

def is_hollow_square(grid: ColoredGrid, i: int, j: int) -> bool:
    color = grid.values[i][j]
    return (all(grid.values[i+di][j+dj] == color for di, dj in [(0,0), (0,1), (0,2), (0,3), (1,0), (1,3), (2,0), (2,3), (3,0), (3,1), (3,2), (3,3)]) and
            all(grid.values[i+di][j+dj] == 0 for di, dj in [(1,1), (1,2), (2,1), (2,2)]))

def replicate_pattern(new_grid: List[List[int]], top_row: int, bottom_row: int, square_color: int):
    width = len(new_grid[0])
    for i in range(top_row, bottom_row + 1):
        for j in range(0, width, 4):
            for di, dj in [(0,0), (0,1), (0,2), (0,3), (1,0), (1,3), (2,0), (2,3), (3,0), (3,1), (3,2), (3,3)]:
                if i + di <= bottom_row and j + dj < width:
                    new_grid[i+di][j+dj] = square_color

def preserve_frame_lines(new_grid: List[List[int]], original_grid: ColoredGrid, top_row: int, bottom_row: int):
    for i in range(top_row, bottom_row + 1):
        for j in range(len(original_grid.values[0])):
            if original_grid.values[i][j] != 0 and original_grid.values[i][j] != new_grid[i][j]:
                new_grid[i][j] = original_grid.values[i][j]

def copy_remaining_elements(new_grid: List[List[int]], original_grid: ColoredGrid, bottom_row: int):
    for i in range(bottom_row + 1, len(original_grid.values)):
        new_grid[i] = original_grid.values[i].copy()
