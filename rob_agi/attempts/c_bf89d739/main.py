from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import statistics

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function identifies red dots, creates a vertical spine through the optimal column,
    and connects all red dots to the spine using the shortest path. It then optimizes
    the connections to form a minimal tree-like structure.
    
    Args:
    input_grid (ColoredGrid): The input grid containing red dots to be connected.
    
    Returns:
    ColoredGrid: A new grid with the red dots connected by green lines forming an optimized tree-like structure.
    """
    result_grid = input_grid.deep_copy()
    red_dots = find_red_dots(input_grid)
    
    if not red_dots:
        return result_grid

    spine_col = find_optimal_spine(red_dots)
    create_vertical_spine(result_grid, red_dots, spine_col)
    connect_dots_to_spine(result_grid, red_dots, spine_col)
    optimize_connections(result_grid, red_dots)

    return result_grid

def find_red_dots(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 2]

def find_optimal_spine(red_dots: List[Tuple[int, int]]) -> int:
    x_coords = [c for _, c in red_dots]
    return round(statistics.mean(x_coords))

def create_vertical_spine(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    min_row = min(r for r, _ in red_dots)
    max_row = max(r for r, _ in red_dots)
    for r in range(min_row, max_row + 1):
        if grid.values[r][spine_col] == 0:
            grid.values[r][spine_col] = 3

def connect_dots_to_spine(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    for r, c in red_dots:
        if c != spine_col:
            for x in range(min(c, spine_col), max(c, spine_col) + 1):
                if grid.values[r][x] == 0:
                    grid.values[r][x] = 3

def optimize_connections(grid: ColoredGrid, red_dots: List[Tuple[int, int]]):
    for i, (r1, c1) in enumerate(red_dots):
        for r2, c2 in red_dots[i+1:]:
            if abs(r1 - r2) + abs(c1 - c2) == 1:  # Adjacent dots
                grid.values[(r1 + r2) // 2][(c1 + c2) // 2] = 3  # Connect directly

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
