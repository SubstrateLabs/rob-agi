from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function identifies red dots, creates a vertical spine through the second leftmost column
    containing red dots, and connects all red dots to the spine using horizontal green lines.
    The result is a tree-like structure connecting all red dots.
    
    Args:
    input_grid (ColoredGrid): The input grid containing red dots to be connected.
    
    Returns:
    ColoredGrid: A new grid with the red dots connected by green lines forming a tree-like structure.
    """
    result_grid = input_grid.deep_copy()
    red_dots = find_red_dots(input_grid)
    
    if not red_dots:
        return result_grid

    spine_col = find_spine_column(red_dots)
    create_vertical_spine(result_grid, red_dots, spine_col)
    connect_dots_to_spine(result_grid, red_dots, spine_col)

    return result_grid

def find_red_dots(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 2]

def find_spine_column(red_dots: List[Tuple[int, int]]) -> int:
    unique_columns = sorted(set(c for _, c in red_dots))
    return unique_columns[1] if len(unique_columns) > 1 else unique_columns[0]

def create_vertical_spine(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    min_row = min(r for r, _ in red_dots)
    max_row = max(r for r, _ in red_dots)
    for r in range(min_row, max_row + 1):
        if grid.values[r][spine_col] == 0:
            grid.values[r][spine_col] = 3

def connect_dots_to_spine(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    for r, c in red_dots:
        if c != spine_col:
            start, end = (c, spine_col) if c < spine_col else (spine_col, c)
            for x in range(start + 1, end):
                if grid.values[r][x] == 0:
                    grid.values[r][x] = 3
