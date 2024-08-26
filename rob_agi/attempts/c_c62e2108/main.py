from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_c62e2108(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the c62e2108 challenge by expanding patterns across the grid.
    
    The function identifies non-black, non-blue patterns in the input grid,
    expands them horizontally and vertically, handles blue areas by filling
    them with nearby expanded colors, and preserves unchanged black areas.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with expanded patterns.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    patterns = find_patterns(input_grid)
    
    for color, r, c in patterns:
        expand_pattern(new_grid, color, r, c, rows, cols)
    
    handle_blue_areas(new_grid, input_grid)
    
    return ColoredGrid(values=new_grid)

def find_patterns(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    patterns = []
    for r, row in enumerate(grid.values):
        for c, color in enumerate(row):
            if color > 1:  # Non-black, non-blue
                patterns.append((color, r, c))
    return sorted(patterns, key=lambda x: (x[1], x[2]))  # Sort by row, then column

def expand_pattern(grid: List[List[int]], color: int, row: int, col: int, rows: int, cols: int):
    # Horizontal expansion
    grid[row] = [color] * cols
    # Vertical expansion
    for r in range(rows):
        grid[r][col] = color

def handle_blue_areas(new_grid: List[List[int]], original_grid: ColoredGrid):
    rows, cols = original_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if original_grid.values[r][c] == 1:  # Blue in original
                if new_grid[r][c] == 0:  # Black in new grid
                    new_color = find_nearest_color(new_grid, r, c, rows, cols)
                    if new_color:
                        new_grid[r][c] = new_color

def find_nearest_color(grid: List[List[int]], row: int, col: int, rows: int, cols: int) -> int:
    for d in range(1, max(rows, cols)):
        for r in range(max(0, row-d), min(rows, row+d+1)):
            for c in range(max(0, col-d), min(cols, col+d+1)):
                if grid[r][c] != 0:
                    return grid[r][c]
    return 0  # If no non-black color found
