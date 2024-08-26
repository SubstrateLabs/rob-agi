from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import statistics

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function identifies red dots, creates a vertical spine through the center of mass,
    connects all red dots to the spine, and optimizes the connections to form a tree-like structure.
    
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
    optimize_connections(result_grid, red_dots, spine_col)

    return result_grid

def find_red_dots(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 2]

def find_optimal_spine(red_dots: List[Tuple[int, int]]) -> int:
    x_coords = [c for _, c in red_dots]
    return round(statistics.mean(x_coords))

def create_vertical_spine(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    min_row = min(r for r, _ in red_dots)
    max_row = max(r for r, _ in red_dots)
    draw_line(grid, (min_row, spine_col), (max_row, spine_col), True)

def connect_dots_to_spine(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    for r, c in red_dots:
        if c != spine_col:
            draw_line(grid, (r, c), (r, spine_col), False)

def optimize_connections(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    sorted_dots = sorted(red_dots, key=lambda x: x[0])  # Sort by row
    for i in range(len(sorted_dots) - 1):
        current_dot = sorted_dots[i]
        next_dot = sorted_dots[i + 1]
        if current_dot[1] == next_dot[1]:  # Same column
            draw_line(grid, current_dot, next_dot, True)
        elif abs(current_dot[1] - spine_col) > abs(next_dot[1] - spine_col):
            # If the next dot is closer to the spine, connect through it
            draw_line(grid, current_dot, (current_dot[0], next_dot[1]), False)
            draw_line(grid, (current_dot[0], next_dot[1]), next_dot, True)

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
