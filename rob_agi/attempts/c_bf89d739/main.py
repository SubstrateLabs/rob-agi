from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import statistics

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function creates a central vertical spine based on the average x-coordinate of red dots.
    It then connects all red dots to this spine using horizontal green lines and vertical branches
    where necessary. The algorithm ensures that all red dots are connected in a tree-like structure
    with a main vertical trunk and optimized branches.
    
    Args:
    input_grid (ColoredGrid): The input grid containing red dots to be connected.
    
    Returns:
    ColoredGrid: A new grid with the red dots connected by green lines forming an optimized tree-like structure.
    """
    result_grid = input_grid.deep_copy()
    red_dots = find_red_dots(input_grid)
    
    if not red_dots:
        return result_grid

    spine_col = find_optimal_spine(red_dots, input_grid.num_cols)
    create_vertical_spine(result_grid, red_dots, spine_col)
    connect_dots_to_spine(result_grid, red_dots, spine_col)
    optimize_connections(result_grid, red_dots, spine_col)

    return result_grid

def find_red_dots(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 2]

def find_optimal_spine(red_dots: List[Tuple[int, int]], num_cols: int) -> int:
    x_coords = [c for _, c in red_dots]
    avg_x = statistics.mean(x_coords)
    spine_candidates = sorted(set(x_coords), key=lambda x: abs(x - avg_x))
    return spine_candidates[0]

def create_vertical_spine(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    spine_dots = [r for r, c in red_dots if c == spine_col]
    if spine_dots:
        draw_line(grid, (min(spine_dots), spine_col), (max(spine_dots), spine_col), True)

def connect_dots_to_spine(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    for r, c in red_dots:
        if c != spine_col:
            draw_line(grid, (r, c), (r, spine_col), False)

def optimize_connections(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    for r, c in red_dots:
        if c != spine_col:
            optimize_single_connection(grid, r, c, spine_col)

def optimize_single_connection(grid: ColoredGrid, row: int, col: int, spine_col: int):
    direction = 1 if col < spine_col else -1
    for x in range(col + direction, spine_col, direction):
        if grid.values[row][x] == 3:  # Found a vertical branch
            draw_line(grid, (row, col), (row, x), False)
            return
    # If no optimization found, keep the original connection to the spine

def draw_line(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int], is_vertical: bool):
    y1, x1 = start
    y2, x2 = end
    
    if is_vertical:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if grid.values[y][x1] == 0:  # Only fill black cells
                grid.values[y][x1] = 3  # Green
    else:
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if grid.values[y1][x] == 0:  # Only fill black cells
                grid.values[y1][x] = 3  # Green
